using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime;

namespace DiffSingerForTuneLab;

// MLRuntime 客户端（逻辑层 + 崩溃处理）：经传输层发 LoadModel/Run/Release，并处理子进程崩溃。
//   · 传输由工厂按 provider 惰性现建（factory(provider)）；子进程崩后丢弃连接，下次 Load/Run 惰性重建并按需重载模型。
//   · 崩溃策略（管道断/子进程崩）：丢弃死连接、本次任务直接抛错上报——宿主侧据此把这次合成标失败并报错给用户。
//     不自动重试同任务（同输入必然再崩 → 自动重启+自动重跑=死循环），不自动切 CPU。
//     下次任务分配时惰性重启子进程、重载模型，后续片段照常合成；崩掉的那次是否重试完全由用户决定。
//   · DML 加载失败（host 报错、exe 存活）：抛可读错误提示用户手动把执行设备改为 CPU——不自动切换（同「崩了不暴力切 CPU」）。
//   · 模型以 path 为稳定键（sessionId 随重建失效），RemoteModelSession 跨重建透明有效。
//   loopback（canRespawn=false）：host 在插件进程内、不会崩，故不启用重建/错误包装。
internal sealed class RuntimeClient : IDisposable
{
    readonly string mProvider;
    readonly Func<string, IRuntimeTransport> mFactory;
    readonly bool mCanRespawn;
    readonly Action<string>? mLog;
    readonly object mLock = new();
    readonly Dictionary<string, (int Id, IReadOnlyDictionary<string, int[]> Inputs)> mLoaded = new(StringComparer.Ordinal);

    IRuntimeTransport? mTransport;

    public RuntimeClient(string provider, Func<string, IRuntimeTransport> transportFactory, bool canRespawn,
        Action<string>? log = null, IRuntimeTransport? initialTransport = null)
    {
        mProvider = provider;
        mFactory = transportFactory;
        mCanRespawn = canRespawn;
        mLog = log;
        mTransport = initialTransport;   // 启动探测已建的传输，首用直接复用（不再经工厂 spawn）
    }

    // 惰性传输：null（首次 / 崩溃丢弃后）时经工厂现建子进程。
    IRuntimeTransport Transport => mTransport ??= mFactory(mProvider);

    // 加载模型，返回输入元数据（名→形状）。幂等（已载即返缓存）。
    public IReadOnlyDictionary<string, int[]> Load(string path)
    {
        lock (mLock)
        {
            if (mLoaded.TryGetValue(path, out var cached))
                return cached.Inputs;

            try
            {
                return LoadInner(path);
            }
            catch (RuntimeHostException ex) when (mCanRespawn && mProvider != "cpu")
            {
                // DML 加载失败（host 报错、exe 存活、非崩溃）：不自动切 CPU，抛可读错误让用户手动改执行设备。
                throw new InvalidOperationException(
                    $"MLRuntime 用 DirectML 加载 {Path.GetFileName(path)} 失败。请在插件设置里把「执行设备」改为 CPU 并重启 TuneLab。原始错误：{ex.Message}", ex);
            }
            catch (Exception ex) when (mCanRespawn && IsTransportFailure(ex))
            {
                DiscardRuntime();
                mLog?.Invoke($"DiffSinger：MLRuntime 加载 {Path.GetFileName(path)} 时崩溃（{ex.Message}）；已丢弃连接，下次合成自动重启。");
                throw new InvalidOperationException(
                    $"MLRuntime 子进程在加载 {Path.GetFileName(path)} 时崩溃，本次合成失败（子进程将在下次合成时自动重启）。", ex);
            }
        }
    }

    public IReadOnlyList<NamedOnnxValue> Run(string path, IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        lock (mLock)
        {
            try
            {
                return RunInner(path, inputs);
            }
            catch (Exception ex) when (mCanRespawn && IsTransportFailure(ex))
            {
                // 子进程崩溃：丢弃死连接、本次片段抛错上报（宿主标失败并报错）——不自动重试（同输入必然再崩 → 死循环）。
                //   下次任务分配时惰性重启子进程、重载模型，后续片段照常；崩掉的那次是否重试由用户决定。
                DiscardRuntime();
                mLog?.Invoke($"DiffSinger：MLRuntime 推理 {Path.GetFileName(path)} 时崩溃（{ex.Message}）；已丢弃连接，下次合成自动重启。");
                throw new InvalidOperationException(
                    $"MLRuntime 子进程在推理 {Path.GetFileName(path)} 时崩溃，本次合成失败（子进程将在下次合成时自动重启）。若重试仍崩，可能是该模型/输入触发原生崩溃。", ex);
            }
        }
    }

    // 释放 host 侧会话（尽力而为）：子进程已崩则会话随进程消亡，无需显式释放。
    public void Release(string path)
    {
        lock (mLock)
        {
            if (!mLoaded.TryGetValue(path, out var e))
                return;
            mLoaded.Remove(path);
            try { Transport.Send(RuntimeProtocol.EncodeRelease(e.Id)); }
            catch { }
        }
    }

    IReadOnlyDictionary<string, int[]> LoadInner(string path)
    {
        var resp = Transport.Send(RuntimeProtocol.EncodeLoadModel(path));
        var (id, inputs) = RuntimeProtocol.DecodeLoadModelResult(resp);
        var map = new Dictionary<string, int[]>(StringComparer.Ordinal);
        foreach (var (name, dims) in inputs)
            map[name] = dims;
        mLoaded[path] = (id, map);
        return map;
    }

    IReadOnlyList<NamedOnnxValue> RunInner(string path, IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        if (!mLoaded.TryGetValue(path, out var e))   // 崩溃丢弃后首次访问：惰性重载（Transport 现建新子进程）
        {
            LoadInner(path);
            e = mLoaded[path];
        }
        var resp = Transport.Send(RuntimeProtocol.EncodeRun(e.Id, inputs));
        return RuntimeProtocol.DecodeRunResult(resp);
    }

    // 丢弃当前（已崩的）传输与失效的 sessionId；下次 Load/Run 经工厂惰性重建子进程并按需重载模型。
    void DiscardRuntime()
    {
        try { mTransport?.Dispose(); } catch { }
        mTransport = null;
        mLoaded.Clear();
    }

    static bool IsTransportFailure(Exception ex)
        => ex is IOException or EndOfStreamException or ObjectDisposedException;

    public void Dispose()
    {
        lock (mLock)
            DiscardRuntime();
    }
}

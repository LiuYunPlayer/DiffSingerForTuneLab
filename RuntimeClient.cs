using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;

namespace DiffSingerForTuneLab;

// MLRuntime 弹性客户端（逻辑层 + 生命周期）：经传输层发 LoadModel/Run/Release，并在子进程崩溃 / DML 失败时自愈。
//   · 传输由工厂按当前 provider 现建（factory(provider)）；respawn = 弃旧传输（杀旧子进程）+ 按新 provider 重建 + 重载已知模型。
//   · 崩溃重启：Run/Load 遇传输级故障（管道断/子进程崩）→ 同 provider 重开、重载、重试一次；再失败则抛出（不无限重启）。
//   · DML→CPU：Load 遇 host 明确报错（多为 DML 算子不支持/设备起不来）→ 整个 runtime 重开为 CPU（进程级同质），重载。
//   · 模型以「路径」为稳定键：sessionId 随子进程重建而失效，故本地缓存 path→(id, 输入元数据)，respawn 后按 path 重载取新 id。
//   loopback（canRespawn=false）不自愈：其 host 在插件进程内，DML→CPU 会触发 AccessViolation；仅子进程模式开启自愈。
internal sealed class RuntimeClient : IDisposable
{
    readonly Func<string, IRuntimeTransport> mFactory;
    readonly bool mCanRespawn;
    readonly Action<string>? mLog;
    readonly object mLock = new();
    readonly Dictionary<string, (int Id, IReadOnlyDictionary<string, int[]> Inputs)> mLoaded = new(StringComparer.Ordinal);

    string mProvider;
    IRuntimeTransport? mTransport;

    public RuntimeClient(string provider, Func<string, IRuntimeTransport> transportFactory, bool canRespawn, Action<string>? log = null)
    {
        mProvider = provider;
        mFactory = transportFactory;
        mCanRespawn = canRespawn;
        mLog = log;
    }

    IRuntimeTransport Transport => mTransport ??= mFactory(mProvider);

    // 加载模型，返回输入元数据（名→形状）。幂等（已载即返缓存）；含 DML→CPU 与崩溃重载自愈。
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
                mLog?.Invoke($"DiffSinger：MLRuntime DirectML 加载 {Path.GetFileName(path)} 失败，重开为 CPU：{ex.Message}");
                RespawnAs("cpu");
                return LoadInner(path);
            }
            catch (Exception ex) when (mCanRespawn && IsTransportFailure(ex))
            {
                mLog?.Invoke($"DiffSinger：MLRuntime 连接中断（{ex.Message}），重启并重载 {Path.GetFileName(path)}。");
                RespawnAs(mProvider);
                return LoadInner(path);
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
                mLog?.Invoke($"DiffSinger：MLRuntime 推理中断（{ex.Message}），重启子进程并重试一次。");
                RespawnAs(mProvider);
                return RunInner(path, inputs);   // 重试一次；再失败则抛出，不无限重启
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
        if (!mLoaded.TryGetValue(path, out var e))
        {
            LoadInner(path);
            e = mLoaded[path];
        }
        var resp = Transport.Send(RuntimeProtocol.EncodeRun(e.Id, inputs));
        return RuntimeProtocol.DecodeRunResult(resp);
    }

    // 重开 runtime：换 provider、弃旧传输（杀旧子进程）、清失效的 sessionId、按 path 重载所有先前已载模型（取新 id）。
    void RespawnAs(string provider)
    {
        mProvider = provider;
        try { mTransport?.Dispose(); } catch { }
        mTransport = null;   // 下次 Transport 访问经 factory(mProvider) 重建
        var paths = mLoaded.Keys.ToArray();
        mLoaded.Clear();
        foreach (var p in paths)
            LoadInner(p);
    }

    static bool IsTransportFailure(Exception ex)
        => ex is IOException or EndOfStreamException or ObjectDisposedException;

    public void Dispose()
    {
        lock (mLock)
        {
            try { mTransport?.Dispose(); } catch { }
            mTransport = null;
            mLoaded.Clear();
        }
    }
}

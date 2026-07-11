using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.OnnxRuntime;

namespace DiffSingerForTuneLab;

// MLRuntime 服务核（传输无关）：持有本进程加载的 InferenceSession，处理 LoadModel/Run 请求帧、回响应帧。
//   P2 经 LoopbackTransport 在插件进程内跑（验证协议+逻辑）；P3 将由 MLRuntime.exe 包裹、经命名管道驱动，核逻辑不变。
//   这是未来子进程内唯一持有原生会话者：provider 决策与设备级串行锁都在此。
//   任何请求处理异常都收敛成 Error 响应帧回传（不崩 host）——真正的原生崩溃（AccessViolation）才是进程级、由子进程隔离兜底。
internal sealed class RuntimeHost : IDisposable
{
    readonly string mProvider;
    readonly object mRunLock = new();          // 设备级串行：DirectML 下 Run 不可并发（跨会话亦然）
    readonly Dictionary<string, int> mByPath = new(StringComparer.Ordinal);
    readonly Dictionary<int, InferenceSession> mSessions = new();
    int mNextId = 1;

    public RuntimeHost(string provider) => mProvider = provider;

    // 处理一个请求帧、返回一个响应帧。任何异常 → Error 响应（携 message）。
    public byte[] Handle(byte[] request)
    {
        try
        {
            using var ms = new MemoryStream(request);
            using var r = new BinaryReader(ms, Encoding.UTF8);
            var op = RuntimeProtocol.PeekOp(r);
            return op switch
            {
                RuntimeOp.LoadModel => HandleLoadModel(RuntimeProtocol.DecodeLoadModel(r)),
                RuntimeOp.Run => HandleRun(RuntimeProtocol.DecodeRun(r)),
                RuntimeOp.Release => HandleRelease(RuntimeProtocol.DecodeRelease(r)),
                _ => RuntimeProtocol.EncodeError($"未知操作码 {(byte)op}"),
            };
        }
        catch (Exception ex)
        {
            return RuntimeProtocol.EncodeError(ex.ToString());
        }
    }

    byte[] HandleLoadModel(string modelPath)
    {
        InferenceSession session;
        int id;
        lock (mRunLock)
        {
            if (mByPath.TryGetValue(modelPath, out id))
            {
                session = mSessions[id];
            }
            else
            {
                session = LoadSession(modelPath);
                id = mNextId++;
                mByPath[modelPath] = id;
                mSessions[id] = session;
            }
        }

        var inputs = new List<(string, int[])>();
        foreach (var kv in session.InputMetadata)
            inputs.Add((kv.Key, kv.Value.Dimensions));
        return RuntimeProtocol.EncodeLoadModelOk(id, inputs);
    }

    byte[] HandleRun((int SessionId, List<NamedOnnxValue> Inputs) req)
    {
        List<NamedOnnxValue> outputs;
        lock (mRunLock)   // 锁只罩 Run + Clone（脱离原生内存）；编码在锁外
        {
            if (!mSessions.TryGetValue(req.SessionId, out var session))
                return RuntimeProtocol.EncodeError($"会话 {req.SessionId} 不存在");
            using var raw = session.Run(req.Inputs);
            outputs = TensorCodec.Clone(raw);
        }
        return RuntimeProtocol.EncodeRunOk(outputs);
    }

    byte[] HandleRelease(int sessionId)
    {
        lock (mRunLock)
        {
            if (mSessions.TryGetValue(sessionId, out var session))
            {
                mSessions.Remove(sessionId);
                foreach (var kv in mByPath)
                    if (kv.Value == sessionId) { mByPath.Remove(kv.Key); break; }
                session.Dispose();
            }
        }
        return RuntimeProtocol.EncodeAck();
    }

    // provider 决策：cpu 只建 CPU；directml 建 DML、失败即抛（不就地回退——同进程 DML/CPU 混用会崩，见止血补丁）。
    //   子进程模型下回退是「杀本进程、按 CPU 目标重开」，故此处失败上抛即可（P4 由客户端侧处理重开）。
    InferenceSession LoadSession(string modelPath)
    {
        if (mProvider == "cpu")
            return new InferenceSession(modelPath);
        var options = new SessionOptions();
        options.AppendExecutionProvider_DML(0);
        return new InferenceSession(modelPath, options);
    }

    public void Dispose()
    {
        lock (mRunLock)
        {
            foreach (var s in mSessions.Values)
                s.Dispose();
            mSessions.Clear();
            mByPath.Clear();
        }
    }
}

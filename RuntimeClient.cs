using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;

namespace DiffSingerForTuneLab;

// MLRuntime 客户端（逻辑层）：经传输层发 LoadModel/Run、解码响应，不碰字节以下细节。
//   一个 client 对应一个 runtime（一个执行目标）；多个 RemoteModelSession 共享同一 client。
//   线程安全：传输为同步请求/响应，推理天然串行；如需并发派发由上层排队（设备本就要求串行）。
internal sealed class RuntimeClient : IDisposable
{
    readonly IRuntimeTransport mTransport;

    public RuntimeClient(IRuntimeTransport transport) => mTransport = transport;

    // 加载模型，返回会话句柄 + 输入口元数据（名→形状）。失败经 Error 响应抛 RuntimeHostException。
    public (int SessionId, IReadOnlyDictionary<string, int[]> Inputs) LoadModel(string modelPath)
    {
        var resp = mTransport.Send(RuntimeProtocol.EncodeLoadModel(modelPath));
        var (id, inputs) = RuntimeProtocol.DecodeLoadModelResult(resp);
        var map = new Dictionary<string, int[]>(StringComparer.Ordinal);
        foreach (var (name, dims) in inputs)
            map[name] = dims;
        return (id, map);
    }

    public IReadOnlyList<NamedOnnxValue> Run(int sessionId, IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        var resp = mTransport.Send(RuntimeProtocol.EncodeRun(sessionId, inputs));
        return RuntimeProtocol.DecodeRunResult(resp);
    }

    // 释放 client 即释放其传输（loopback 下连带释放 host；子进程下断管道、由生命周期管理杀进程）。
    public void Dispose() => mTransport.Dispose();
}

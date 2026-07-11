using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.OnnxRuntime;

namespace DiffSingerForTuneLab;

// MLRuntime IPC 线协议（编解码层）：请求/响应 ↔ 字节帧。张量编组复用 DiffSingerTensorCache 的序列化（单一来源，
//   与磁盘缓存同款、已被缓存往返证明无损）。与传输无关——传输层只管把一个字节帧送达对端、取回一个字节帧。
//   帧布局：请求 [op:byte] payload；响应 [status:byte] payload（Ok=0 / Error=1）。
internal enum RuntimeOp : byte { LoadModel = 1, Run = 2 }
internal enum RuntimeStatus : byte { Ok = 0, Error = 1 }

internal static class RuntimeProtocol
{
    // —— 请求编码（客户端）——
    public static byte[] EncodeLoadModel(string modelPath)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8);
        w.Write((byte)RuntimeOp.LoadModel);
        w.Write(modelPath);
        w.Flush();
        return ms.ToArray();
    }

    public static byte[] EncodeRun(int sessionId, IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8);
        w.Write((byte)RuntimeOp.Run);
        w.Write(sessionId);
        DiffSingerTensorCache.WriteValues(w, inputs);
        w.Flush();
        return ms.ToArray();
    }

    // —— 请求解码（服务端）——
    public static RuntimeOp PeekOp(BinaryReader r) => (RuntimeOp)r.ReadByte();
    public static string DecodeLoadModel(BinaryReader r) => r.ReadString();
    public static (int SessionId, List<NamedOnnxValue> Inputs) DecodeRun(BinaryReader r)
        => (r.ReadInt32(), DiffSingerTensorCache.ReadValues(r));

    // —— 响应编码（服务端）——
    public static byte[] EncodeLoadModelOk(int sessionId, IReadOnlyList<(string Name, int[] Dims)> inputs)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8);
        w.Write((byte)RuntimeStatus.Ok);
        w.Write(sessionId);
        w.Write(inputs.Count);
        foreach (var (name, dims) in inputs)
        {
            w.Write(name);
            w.Write(dims.Length);
            foreach (var d in dims)
                w.Write(d);
        }
        w.Flush();
        return ms.ToArray();
    }

    public static byte[] EncodeRunOk(IReadOnlyCollection<NamedOnnxValue> outputs)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8);
        w.Write((byte)RuntimeStatus.Ok);
        DiffSingerTensorCache.WriteValues(w, outputs);
        w.Flush();
        return ms.ToArray();
    }

    public static byte[] EncodeError(string message)
    {
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms, Encoding.UTF8);
        w.Write((byte)RuntimeStatus.Error);
        w.Write(message);
        w.Flush();
        return ms.ToArray();
    }

    // —— 响应解码（客户端）——：Error 状态即抛 RuntimeHostException。
    public static (int SessionId, List<(string Name, int[] Dims)> Inputs) DecodeLoadModelResult(byte[] frame)
    {
        using var ms = new MemoryStream(frame);
        using var r = new BinaryReader(ms, Encoding.UTF8);
        ThrowIfError(r);
        int id = r.ReadInt32();
        int n = r.ReadInt32();
        var inputs = new List<(string, int[])>(n);
        for (int i = 0; i < n; i++)
        {
            var name = r.ReadString();
            int rank = r.ReadInt32();
            var dims = new int[rank];
            for (int k = 0; k < rank; k++)
                dims[k] = r.ReadInt32();
            inputs.Add((name, dims));
        }
        return (id, inputs);
    }

    public static List<NamedOnnxValue> DecodeRunResult(byte[] frame)
    {
        using var ms = new MemoryStream(frame);
        using var r = new BinaryReader(ms, Encoding.UTF8);
        ThrowIfError(r);
        return DiffSingerTensorCache.ReadValues(r);
    }

    static void ThrowIfError(BinaryReader r)
    {
        if ((RuntimeStatus)r.ReadByte() == RuntimeStatus.Error)
            throw new RuntimeHostException(r.ReadString());
    }
}

// runtime 侧处理请求时抛出的错误，经 Error 响应帧回传客户端后在此重抛（携原始 message）。
internal sealed class RuntimeHostException : Exception
{
    public RuntimeHostException(string message) : base(message) { }
}

using System;
using System.Buffers.Binary;
using System.IO;

namespace DiffSingerForTuneLab;

// 命名管道消息分帧：管道是字节流，需自带长度前缀切出「一个请求/响应字节帧」。格式 [int32 LE 长度][payload]。
//   插件（客户端）与 MLRuntime.exe（服务端）经链接文件共用同一份分帧，保证两端字节对齐。
internal static class PipeFraming
{
    public static void WriteFrame(Stream stream, byte[] payload)
    {
        Span<byte> len = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(len, payload.Length);
        stream.Write(len);
        stream.Write(payload, 0, payload.Length);
        stream.Flush();
    }

    // 读一帧；对端在帧边界干净关闭（EOF）返回 null（正常收尾），帧中途断裂抛 EndOfStreamException（异常收尾）。
    public static byte[]? ReadFrame(Stream stream)
    {
        var lenBuf = ReadExactly(stream, 4);
        if (lenBuf == null)
            return null;
        int len = BinaryPrimitives.ReadInt32LittleEndian(lenBuf);
        if (len < 0)
            throw new InvalidDataException($"管道帧长度非法：{len}");
        return ReadExactly(stream, len) ?? throw new EndOfStreamException("管道帧不完整（对端中途断裂）");
    }

    // 读满 count 字节；起始即 EOF 返回 null（帧边界干净关闭），读到一半 EOF 抛。
    static byte[]? ReadExactly(Stream stream, int count)
    {
        if (count == 0)
            return Array.Empty<byte>();
        var buf = new byte[count];
        int off = 0;
        while (off < count)
        {
            int n = stream.Read(buf, off, count - off);
            if (n == 0)
                return off == 0 ? null : throw new EndOfStreamException("管道帧不完整（对端中途断裂）");
            off += n;
        }
        return buf;
    }
}

using System;

namespace DiffSingerForTuneLab;

// 传输层：把一个请求字节帧送达 runtime、取回一个响应字节帧。同步请求/响应（推理本就串行，无需并发）。
//   实现可换、逻辑层/编解码层不依赖具体实现：
//   · LoopbackTransport（P2）——进程内直连，验证协议与 host 逻辑；
//   · PipeTransport（P3）——命名管道驱动 MLRuntime.exe 子进程；
//   · SharedMemoryTransport（将来）——大帧走 MMF、管道只传控制帧。换传输只改本层。
internal interface IRuntimeTransport : IDisposable
{
    byte[] Send(byte[] request);
}

// 进程内直连：请求字节帧直接交给同进程的 RuntimeHost 处理、取回响应字节帧。
//   仍完整走「编码→字节帧→解码→执行→编码→字节帧→解码」全链，故能验证协议编解码往返无损、host 逻辑正确，
//   只是不牵扯任何跨进程 I/O。子进程落地后（P3）换成 PipeTransport 即可，上层一行不改。
internal sealed class LoopbackTransport : IRuntimeTransport
{
    readonly RuntimeHost mHost;

    public LoopbackTransport(RuntimeHost host) => mHost = host;

    public byte[] Send(byte[] request) => mHost.Handle(request);

    public void Dispose() => mHost.Dispose();
}

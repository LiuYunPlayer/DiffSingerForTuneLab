using System;
using System.IO;
using System.IO.Pipes;
using DiffSingerForTuneLab;

// MLRuntime.exe：独立进程跑 ONNX 推理，经命名管道被插件驱动。
//   用法：MLRuntime <pipeName> <provider>
//   · 隔离 onnxruntime 原生崩溃（DML/CPU 混用 AccessViolation、并发 Run 设备崩、dispose use-after-free 等）于本进程，
//     崩了只崩自己、宿主 TuneLab 不受牵连；插件侧检测管道断即重启本进程。
//   · 孤儿治理由插件侧 Job Object 兜底（宿主消亡 → OS 杀本进程），本进程再以管道 EOF 自杀作快速副保险。
if (args.Length < 2)
{
    Console.Error.WriteLine("用法: MLRuntime <pipeName> <provider>");
    return 2;
}

string pipeName = args[0];
string provider = args[1];

try
{
    using var host = new RuntimeHost(provider);
    using var server = new NamedPipeServerStream(
        pipeName, PipeDirection.InOut, maxNumberOfServerInstances: 1, PipeTransmissionMode.Byte);

    server.WaitForConnection();   // 等插件连上（插件死则 Job Object 兜底杀本进程）

    // 请求循环：读一帧 → host.Handle → 回一帧。管道断（插件退出/崩溃）= EOF/异常 → 退出进程（自杀副保险）。
    while (true)
    {
        byte[]? request;
        try { request = PipeFraming.ReadFrame(server); }
        catch { break; }              // 管道破损
        if (request == null) break;   // 帧边界干净 EOF：插件正常关闭
        var response = host.Handle(request);
        try { PipeFraming.WriteFrame(server, response); }
        catch { break; }              // 回写失败：对端已断
    }
    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"MLRuntime 致命错误：{ex}");
    return 1;
}

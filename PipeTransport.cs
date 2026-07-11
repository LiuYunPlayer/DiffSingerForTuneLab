using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;

namespace DiffSingerForTuneLab;

// 命名管道传输：spawn MLRuntime.exe，经命名管道同步请求/响应。子进程隔离 onnxruntime 原生崩溃于自身。
//   生命周期：构造即 spawn + 连接；Dispose 关管道（触发 exe 读到 EOF 自杀）+ Job Object 兜底杀（治孤儿）。
//   崩溃重启、DML 失败→重开 CPU 等更完整的生命周期策略见 P4（由 DiffSingerModelCache 侧编排）。
internal sealed class PipeTransport : IRuntimeTransport
{
    readonly Process mProcess;
    readonly NamedPipeClientStream mPipe;
    readonly JobObject? mJob;

    public PipeTransport(string exePath, string provider)
    {
        if (!File.Exists(exePath))
            throw new FileNotFoundException($"MLRuntime 可执行文件不存在：{exePath}", exePath);

        var pipeName = "diffsinger-mlruntime-" + Guid.NewGuid().ToString("N");

        var psi = new ProcessStartInfo
        {
            FileName = exePath,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardError = true,
            WorkingDirectory = Path.GetDirectoryName(exePath)!,
        };
        psi.ArgumentList.Add(pipeName);
        psi.ArgumentList.Add(provider);

        mProcess = Process.Start(psi) ?? throw new InvalidOperationException($"无法启动 MLRuntime：{exePath}");

        // 进 Job：宿主消亡 → OS 杀本子进程（治孤儿主保险）。仅 Windows；建 Job 失败不致命（尚有管道 EOF 副保险）。
        if (OperatingSystem.IsWindows())
        {
            try
            {
                mJob = new JobObject();
                mJob.AssignProcess(mProcess.Handle);
            }
            catch { mJob = null; }
        }

        mPipe = new NamedPipeClientStream(".", pipeName, PipeDirection.InOut);
        try
        {
            mPipe.Connect(10000);   // 等 exe 起管道服务（10s 超时）
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    public byte[] Send(byte[] request)
    {
        PipeFraming.WriteFrame(mPipe, request);
        return PipeFraming.ReadFrame(mPipe) ?? throw new IOException("MLRuntime 连接已关闭（对端退出）");
    }

    public void Dispose()
    {
        try { mPipe.Dispose(); } catch { }              // 关管道 → exe 读到 EOF 自杀
        try { if (!mProcess.WaitForExit(2000)) mProcess.Kill(); } catch { }
        try { mProcess.Dispose(); } catch { }
        if (OperatingSystem.IsWindows())
            mJob?.Dispose();                            // 关 Job（KILL_ON_JOB_CLOSE 兜底）
    }
}

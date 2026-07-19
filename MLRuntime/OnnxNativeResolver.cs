using System;
using System.IO;
using System.Runtime.InteropServices;

namespace DiffSingerForTuneLab;

// 让 MLRuntime.exe 子进程复用【父插件目录】的 onnxruntime 原生库，不再随 mlruntime/ 自带一份（去重 ~15MB×平台）。
//   打包/部署时删掉 mlruntime/runtimes/，本解析器改从 <plugin>/runtimes/<rid>/native/onnxruntime.dll 加载。
//   MLRuntime.exe 恒位于 <plugin>/mlruntime/（pack/deploy 布局保证），故父目录即插件根、带整树 runtimes/。
//   找不到（如独立运行的 raw build，其副本在自身 runtimes/ 而非父目录）→ 返回 Zero 回退默认 deps.json 解析，故安全。
//   仅拦截托管 P/Invoke 的 "onnxruntime"；onnxruntime.dll 内部再 LoadLibrary 的 DirectML.dll 走系统 System32（本就不随包）。
internal static class OnnxNativeResolver
{
    public static void Register()
    {
        var baseDir = AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var pluginDir = Path.GetDirectoryName(baseDir);   // <plugin>/mlruntime → <plugin>
        if (pluginDir == null)
            return;

        string rid = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.X86 => "win-x86",
            Architecture.Arm64 => "win-arm64",
            _ => "win-x64",
        };
        var candidate = Path.Combine(pluginDir, "runtimes", rid, "native", "onnxruntime.dll");

        NativeLibrary.SetDllImportResolver(typeof(Microsoft.ML.OnnxRuntime.SessionOptions).Assembly, (name, _, _) =>
            name == "onnxruntime" && File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out var handle)
                ? handle
                : IntPtr.Zero);   // 回退默认解析（自带 runtimes/ 仍在时可载）

        // 预载随包 DirectML.dll（否则 DML EP delay-load 会走到 System32 的旧版而失败，见 DirectMlNative 说明）。
        DirectMlNative.Preload(pluginDir);
    }
}

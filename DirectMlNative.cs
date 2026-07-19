using System;
using System.IO;
using System.Runtime.InteropServices;

namespace DiffSingerForTuneLab;

// 显式预载随包分发的 DirectML.dll，令 onnxruntime 的 DML EP delay-load 命中已加载模块。
//   为何必须：onnxruntime.dll 对 DirectML.dll 是 delay-load，AppendExecutionProvider_DML 时才按【标准搜索】
//   （进程 exe 目录 → System32 → PATH）去找它——恰恰不含 onnxruntime.dll 自身所在的 runtimes/<rid>/native/。
//   于是即便我们把 DirectML.dll 放进 native/，子进程/宿主也只会捞到 System32 那份：旧 Windows（如 Win10 19045）
//   自带的是 2020 年的 DirectML 1.0，喂不动 onnxruntime 1.20 的 DML EP → AppendDML 直接失败。
//   Windows 按【模块基名】匹配已加载模块，故只要在 AppendDML 前用全路径把我们的 DirectML.dll 先 Load 进来，
//   后续 onnxruntime 内部 LoadLibrary("DirectML.dll") 就会复用它、不再走搜索。（实测生效，见提交说明。）
//   幂等、尽力而为：文件不在或加载失败都不抛——大不了退回 System32 那份，行为不劣于预载前。
internal static class DirectMlNative
{
    public static void Preload(string pluginRootDir)
    {
        try
        {
            string rid = RuntimeInformation.ProcessArchitecture switch
            {
                Architecture.X86 => "win-x86",
                Architecture.Arm64 => "win-arm64",
                _ => "win-x64",
            };
            var dml = Path.Combine(pluginRootDir, "runtimes", rid, "native", "DirectML.dll");
            if (File.Exists(dml))
                NativeLibrary.TryLoad(dml, out _);
        }
        catch { /* 尽力而为：失败则退回默认搜索（System32），不比修复前更差 */ }
    }
}

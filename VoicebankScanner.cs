using System;
using System.Collections.Generic;
using System.IO;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 在给定的一组根目录下发现 DiffSinger 物理模型包（返回去重后的根目录绝对路径列表）。
// 包判别：含声学主配置 dsconfig.yaml，且含 character 元数据文件。
// 预测器子目录（dsdur / dspitch / dsvariance）也各自带 dsconfig.yaml，但无 character.*——据此排除；
// 探到包后即不再下钻其子目录。合并（按 model id / voice id）与展示元数据交由 VoiceRegistry。
internal static class VoicebankScanner
{
    const int MaxDepth = 6;

    public static List<string> Scan(IEnumerable<string> roots, ILogger logger)
    {
        var result = new List<string>();
        var seenPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var root in roots)
        {
            if (string.IsNullOrWhiteSpace(root))
                continue;

            string full;
            try { full = Path.GetFullPath(root); }
            catch { continue; }

            if (!Directory.Exists(full))
                continue;

            try { Walk(full, 0, result, seenPaths); }
            catch (Exception ex) { logger.Warning($"扫描声库根目录失败 {full}: {ex.Message}"); }
        }

        return result;
    }

    static void Walk(string dir, int depth, List<string> result, HashSet<string> seenPaths)
    {
        if (IsVoicebank(dir))
        {
            if (seenPaths.Add(Path.GetFullPath(dir)))
                result.Add(Path.GetFullPath(dir));
            return;   // 不下钻：子目录是 dsdur / dspitch / dsvariance 等预测器
        }

        if (depth >= MaxDepth)
            return;

        string[] subDirs;
        try { subDirs = Directory.GetDirectories(dir); }
        catch { return; }

        foreach (var sub in subDirs)
            Walk(sub, depth + 1, result, seenPaths);
    }

    static bool IsVoicebank(string dir)
    {
        if (!File.Exists(Path.Combine(dir, "dsconfig.yaml")))
            return false;

        return File.Exists(Path.Combine(dir, "character.yaml"))
            || File.Exists(Path.Combine(dir, "character.txt"));
    }
}

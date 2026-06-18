using System;
using System.Collections.Generic;
using System.IO;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 在给定的一组根目录下发现 DiffSinger 声库。
// 声库根判别：含声学主配置 dsconfig.yaml，且含 character 元数据文件。
// 预测器子目录（dsdur / dspitch / dsvariance）也各自带 dsconfig.yaml，但无 character.*——据此排除；
// 探到声库后即不再下钻其子目录。按解析后的全路径去重（重叠根重复命中只取一次），
// voiceId 取声库文件夹名、冲突时追加序号保唯一。
internal static class VoicebankScanner
{
    const int MaxDepth = 6;

    public static List<DiscoveredVoicebank> Scan(IEnumerable<string> roots, ILogger logger)
    {
        var result = new List<DiscoveredVoicebank>();
        var seenPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var usedIds = new HashSet<string>(StringComparer.Ordinal);

        foreach (var root in roots)
        {
            if (string.IsNullOrWhiteSpace(root))
                continue;

            string full;
            try { full = Path.GetFullPath(root); }
            catch { continue; }

            if (!Directory.Exists(full))
                continue;

            try { Walk(full, 0, result, seenPaths, usedIds); }
            catch (Exception ex) { logger.Warning($"扫描声库根目录失败 {full}: {ex.Message}"); }
        }

        return result;
    }

    static void Walk(string dir, int depth, List<DiscoveredVoicebank> result,
        HashSet<string> seenPaths, HashSet<string> usedIds)
    {
        if (IsVoicebank(dir))
        {
            if (seenPaths.Add(Path.GetFullPath(dir)))
                result.Add(Build(dir, usedIds));
            return;   // 不下钻：子目录是 dsdur / dspitch / dsvariance 等预测器
        }

        if (depth >= MaxDepth)
            return;

        string[] subDirs;
        try { subDirs = Directory.GetDirectories(dir); }
        catch { return; }

        foreach (var sub in subDirs)
            Walk(sub, depth + 1, result, seenPaths, usedIds);
    }

    static bool IsVoicebank(string dir)
    {
        if (!File.Exists(Path.Combine(dir, "dsconfig.yaml")))
            return false;

        return File.Exists(Path.Combine(dir, "character.yaml"))
            || File.Exists(Path.Combine(dir, "character.txt"));
    }

    static DiscoveredVoicebank Build(string bankDir, HashSet<string> usedIds)
    {
        var folderName = new DirectoryInfo(bankDir).Name;
        var meta = CharacterMetadata.Read(bankDir);
        var voiceId = UniqueId(folderName, usedIds);

        ImageResource? portrait = null;
        if (!string.IsNullOrWhiteSpace(meta.ImageFile))
        {
            var imagePath = Path.Combine(bankDir, meta.ImageFile);
            if (File.Exists(imagePath))
                portrait = new FileImageResource(imagePath);
        }

        var info = new VoiceSourceInfo
        {
            Name = string.IsNullOrWhiteSpace(meta.Name) ? folderName : meta.Name!,
            Description = meta.Author ?? string.Empty,
            Portrait = portrait,
        };

        return new DiscoveredVoicebank(voiceId, Path.GetFullPath(bankDir), info);
    }

    static string UniqueId(string baseName, HashSet<string> usedIds)
    {
        var id = baseName;
        int n = 2;
        while (!usedIds.Add(id))
            id = $"{baseName} ({n++})";
        return id;
    }
}

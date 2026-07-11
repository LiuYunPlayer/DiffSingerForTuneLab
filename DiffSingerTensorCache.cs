using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Hashing;
using System.Linq;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// DiffSinger 张量缓存：把一个 ONNX 模型调用的输出按「模型文件哈希 + 序列化输入」为键缓存到磁盘，
//   反复合成（撤销重做、重开工程、改动不影响某块、跨块/跨说话人共享 linguistic 等）时直接复用、免重算。
// 忠实移植 OpenUtau DiffSingerCache（序列化/反序列化、文件格式与按 name 排序求键一致），差异仅在：
//   · 哈希用 System.IO.Hashing.XxHash64（OpenUtau 用 K4os.Hash.xxHash），数值不同但本插件不需跨工具兼容；
//   · 缓存目录取插件独立用户数据根 UserDataRoot/Cache（OpenUtau 用 PathManager.Inst.CachePath）。
// 另附编排封装：Run（建键→Load→未命中则模型 Run + Save，返回脱离原生内存的托管张量）、
//   Clone（把模型原生输出深拷成托管，供未命中/禁用时安全返回）、HashFile（模型 identifier）、EnforceSizeLimit（LRU 逐出）。
public sealed class DiffSingerTensorCache
{
    const string FormatHeader = "TENSORCACHE";

    readonly ulong mHash;
    readonly string mFilename;

    public ulong Hash => mHash;
    public string Filename => mFilename;

    // 缓存目录：插件用户数据根下的 Cache（与 Voices/Vocoders 并列）。
    public static string CacheDirectory => Path.Combine(DiffSingerDeclarations.UserDataRoot, "Cache");

    public DiffSingerTensorCache(ulong identifier, IReadOnlyCollection<NamedOnnxValue> inputs)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write(identifier);
            foreach (var onnxValue in inputs.OrderBy(v => v.Name, StringComparer.InvariantCulture))
                TensorCodec.WriteValue(writer, onnxValue);
        }

        mHash = XxHash64.HashToUInt64(stream.ToArray());
        mFilename = $"ds-{mHash:x16}.tensorcache";
    }

    // —— 编排封装 ——
    // 推理串行化 + 会话退役机制已内聚进 InProcessModelSession（设备级约束随「推理在哪跑」走）；
    //   本类只管缓存编排与张量序列化，不再持有原生会话或全局锁。

    // 跑一个模型并经缓存：命中直接返回缓存输出；未命中则 Run + Save。model.Run 已返回脱离原生内存的托管张量。
    //   enabled=false 时跳过磁盘缓存，直接返回 model.Run 的托管输出。
    public static IReadOnlyList<NamedOnnxValue> Run(
        IModelSession model, ulong identifier, IReadOnlyCollection<NamedOnnxValue> inputs, bool enabled)
    {
        if (!enabled)
            return model.Run(inputs);

        var cache = new DiffSingerTensorCache(identifier, inputs);
        var loaded = cache.Load();
        if (loaded != null)
            return loaded;

        var result = model.Run(inputs);
        cache.Save(result);
        return result;
    }

    // 模型 identifier：.onnx 文件内容的 XxHash64（流式、不整体载入内存）。加载时算一次缓存进字段，
    //   用作缓存键的一部分，区分不同模型（同输入不同权重不撞键），且模型文件更换即自动失效。
    public static ulong HashFile(string path)
    {
        var h = new XxHash64();
        using var fs = File.OpenRead(path);
        h.Append(fs);
        return h.GetCurrentHashAsUInt64();
    }

    // LRU 体积上限逐出：缓存目录超过上限时，按最近访问时间删最旧的 .tensorcache 直到回落。maxSizeMb<=0 视作不限制。
    //   尽力而为，任何 IO 异常吞掉（逐出失败不应影响合成）。
    public static void EnforceSizeLimit(long maxSizeMb)
    {
        if (maxSizeMb <= 0)
            return;
        try
        {
            var dir = CacheDirectory;
            if (!Directory.Exists(dir))
                return;
            var files = new DirectoryInfo(dir).GetFiles("*.tensorcache");
            long total = files.Sum(f => f.Length);
            long limit = maxSizeMb * 1024L * 1024L;
            if (total <= limit)
                return;
            foreach (var f in files.OrderBy(f => f.LastAccessTimeUtc))
            {
                try { long len = f.Length; f.Delete(); total -= len; } catch { }
                if (total <= limit)
                    break;
            }
        }
        catch { }
    }

    public IReadOnlyList<NamedOnnxValue>? Load()
    {
        var cachePath = Path.Join(CacheDirectory, mFilename);
        if (!File.Exists(cachePath))
            return null;

        var result = new List<NamedOnnxValue>();
        try
        {
            using (var stream = new FileStream(cachePath, FileMode.Open, FileAccess.Read))
            using (var reader = new BinaryReader(stream))
            {
                if (reader.ReadString() != FormatHeader)
                    throw new InvalidDataException($"[TensorCache] 缓存文件头异常：{mFilename}。");
                result = TensorCodec.ReadValues(reader);
            }
        }
        catch (Exception e)
        {
            TuneLabContext.Global.GetLogger().Warning($"DiffSinger：反序列化缓存 {mFilename} 失败、丢弃重算：{e.Message}");
            Delete();
            return null;
        }

        // 命中即「访问」：显式刷新访问时间，令 LRU 逐出以真实使用近度排序（不依赖 NTFS 自动 last-access 策略）。
        try { File.SetLastAccessTimeUtc(cachePath, DateTime.UtcNow); } catch { }
        return result;
    }

    public void Delete()
    {
        var cachePath = Path.Join(CacheDirectory, mFilename);
        if (File.Exists(cachePath))
        {
            try { File.Delete(cachePath); } catch { }
        }
    }

    public void Save(IReadOnlyCollection<NamedOnnxValue> outputs)
    {
        Directory.CreateDirectory(CacheDirectory);
        var cachePath = Path.Join(CacheDirectory, mFilename);
        using var stream = new FileStream(cachePath, FileMode.Create, FileAccess.Write);
        using var writer = new BinaryWriter(stream);
        writer.Write(FormatHeader);
        TensorCodec.WriteValues(writer, outputs);
    }

}

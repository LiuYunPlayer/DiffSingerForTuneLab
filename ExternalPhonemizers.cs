using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Loader;
using OpenUtau.Api;
using OpenUtau.Core.DiffSinger;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 声库自带 OpenUtau 音素器程序集的发现、装载与按语言注册。
// 忠实对齐 OpenUtau DocManager.SearchAllPlugins：递归扫 *.dll → 托管程序集判定（IsManagedAssembly，
//   逐字节移植其 PE 解析）→ 加载 → 找音素器子类实例化，逐文件 try/catch、坏件记日志跳过。
// 与 OpenUtau 的差异：
//   · OpenUtau 扫全局 Plugins 目录（声库作者在说明书里让用户手动拷 DLL）；本插件直接扫声库目录，开箱即用；
//   · OpenUtau 收 Phonemizer 全家（含 UTAU 系）；本插件只认 DiffSingerBasePhonemizer 子类——这类 DLL 是
//     纯 G2P 配置壳（词典名/语言码/算法引擎包），歌词→音素编排走本插件自己的管线，无需 OpenUtau 其余 API；
//   · DLL 对 "OpenUtau.Core, 1.0.0.0" 的引用由随包的门面程序集满足（专用 ALC 的 Load 显式共享，
//     OnnxRuntime 同享插件自带副本；其余系统程序集回落默认解析）。
// 注意：这是执行声库携带的第三方代码（与 OpenUtau 同一信任模型——装谁的声库就是信谁）。
public sealed class ExternalPhonemizerSet
{
    // 按声库根目录（声学 dsconfig 所在目录）缓存；扫描/装载一次，跨会话共享。
    static readonly ConcurrentDictionary<string, ExternalPhonemizerSet> Cache = new(StringComparer.OrdinalIgnoreCase);

    public static ExternalPhonemizerSet For(string rootPath, ILogger logger)
        => Cache.GetOrAdd(Path.GetFullPath(rootPath), p => new ExternalPhonemizerSet(p, logger));

    readonly Dictionary<string, ExternalPhonemizerEntry> mByLang = new(StringComparer.OrdinalIgnoreCase);

    public ExternalPhonemizerEntry? ForLang(string lang)
        => !string.IsNullOrEmpty(lang) && mByLang.TryGetValue(lang, out var e) ? e : null;

    ExternalPhonemizerSet(string rootPath, ILogger logger)
    {
        List<string> files;
        try
        {
            // 扫声库根（LYSE 型：DLL 在 <root>/phonemizers/）；configs/ 型布局下声学根是子目录，
            // 补扫父级 phonemizers/（DLL 惯例放包顶层）。
            var dirs = new List<string> { rootPath };
            var parent = Path.GetDirectoryName(rootPath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
            if (parent != null)
            {
                var sibling = Path.Combine(parent, "phonemizers");
                if (Directory.Exists(sibling)) dirs.Add(sibling);
            }
            files = dirs
                .SelectMany(d => Directory.EnumerateFiles(d, "*.dll", SearchOption.AllDirectories))
                .Select(Path.GetFullPath)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
        }
        catch (Exception e)
        {
            logger.Warning($"DiffSinger：扫描声库音素器目录失败：{e.Message}");
            return;
        }
        if (files.Count == 0) return;

        var alc = new PhonemizerLoadContext(rootPath, files.Select(Path.GetDirectoryName).OfType<string>().Distinct(StringComparer.OrdinalIgnoreCase).ToArray());
        foreach (var file in files)
        {
            try
            {
                if (!IsManagedAssembly(file)) continue;   // 原生库（onnxruntime 之类）静默跳过
                var assembly = alc.LoadFromAssemblyPath(file);
                foreach (var type in assembly.GetExportedTypes())
                {
                    if (type.IsAbstract || !type.IsSubclassOf(typeof(DiffSingerBasePhonemizer))) continue;
                    var instance = (DiffSingerBasePhonemizer)Activator.CreateInstance(type)!;

                    // 语言归属：GetLangCode() 优先；未声明再落 [Phonemizer(language:)] 特性。都无则没法自动挂语言，跳过。
                    var attr = type.GetCustomAttribute<PhonemizerAttribute>();
                    var lang = instance.LangCode;
                    if (string.IsNullOrEmpty(lang)) lang = attr?.Language?.ToLowerInvariant() ?? string.Empty;
                    if (string.IsNullOrEmpty(lang))
                    {
                        logger.Warning($"DiffSinger：声库音素器 {type.Name}（{Path.GetFileName(file)}）未声明语言码，跳过");
                        continue;
                    }

                    var entry = new ExternalPhonemizerEntry(lang, instance, attr?.Name ?? type.Name, logger);
                    // 同语言多实现：带算法 G2P 引擎的优先；同级先到先得。
                    if (!mByLang.TryGetValue(lang, out var existing) || (!existing.HasEngine && entry.HasEngine))
                        mByLang[lang] = entry;
                    logger.Info($"DiffSinger：装载声库音素器 {entry.Name}（{lang}，{Path.GetFileName(file)}）");
                }
            }
            catch (Exception e)
            {
                // 对齐 OpenUtau：单个 DLL 装载失败不连坐（引用面超出门面 API 的重型音素器也落到这里降级）。
                logger.Warning($"DiffSinger：装载声库音素器 {Path.GetFileName(file)} 失败：{e.Message}");
            }
        }
    }

    // 专用装载上下文：OpenUtau.Core → 门面、OnnxRuntime → 插件自带副本、同目录旁邻 DLL → 就地解析；
    //   其余（System.* 等）返回 null 走默认回落。每声库一个上下文（不同声库同名程序集互不串）。
    sealed class PhonemizerLoadContext : AssemblyLoadContext
    {
        readonly string[] mProbeDirs;

        public PhonemizerLoadContext(string name, string[] probeDirs) : base($"ds-phonemizers:{name}")
        {
            mProbeDirs = probeDirs;
        }

        protected override Assembly? Load(AssemblyName assemblyName)
        {
            if (string.Equals(assemblyName.Name, "OpenUtau.Core", StringComparison.OrdinalIgnoreCase))
                return typeof(DiffSingerBasePhonemizer).Assembly;
            if (string.Equals(assemblyName.Name, "Microsoft.ML.OnnxRuntime", StringComparison.OrdinalIgnoreCase))
                return typeof(Microsoft.ML.OnnxRuntime.InferenceSession).Assembly;
            foreach (var dir in mProbeDirs)
            {
                var candidate = Path.Combine(dir, assemblyName.Name + ".dll");
                if (File.Exists(candidate))
                    return LoadFromAssemblyPath(candidate);
            }
            return null;
        }
    }

    // 忠实移植自 OpenUtau（MIT）——OpenUtau.Core/Util/LibraryLoader.IsManagedAssembly。
    static bool IsManagedAssembly(string fileName)
    {
        using var fileStream = new FileStream(fileName, FileMode.Open, FileAccess.Read);
        using var binaryReader = new BinaryReader(fileStream);
        if (fileStream.Length < 64)
            return false;

        // PE Header 指针在 0x3C（4 字节）。
        fileStream.Position = 0x3C;
        uint peHeaderPointer = binaryReader.ReadUInt32();
        if (peHeaderPointer == 0)
            peHeaderPointer = 0x80;

        // 确保后续结构（PE 签名/头 + 标准域 + NT 域 + 数据目录表）在文件内。
        if (peHeaderPointer > fileStream.Length - 256)
            return false;

        // PE 签名应为 'PE\0\0'。
        fileStream.Position = peHeaderPointer;
        uint peHeaderSignature = binaryReader.ReadUInt32();
        if (peHeaderSignature != 0x00004550)
            return false;

        // 跳过 PEHeader 各域。
        fileStream.Position += 20;

        const ushort PE32 = 0x10b;
        const ushort PE32Plus = 0x20b;

        var peFormat = binaryReader.ReadUInt16();
        if (peFormat != PE32 && peFormat != PE32Plus)
            return false;

        // 第 15 个数据目录 RVA = CLI header RVA；非零即托管程序集。
        ushort dataDictionaryStart = (ushort)(peHeaderPointer + (peFormat == PE32 ? 232 : 248));
        fileStream.Position = dataDictionaryStart;

        uint cliHeaderRva = binaryReader.ReadUInt32();
        return cliHeaderRva != 0;
    }
}

// 已装载的单个声库音素器：语言码 + 词典名 + （可选）算法 G2P 引擎（懒初始化——
//   LoadBaseG2p 首调会解嵌入资源包并建 ONNX 会话，只在该语言真被使用时发生；失败记日志、降级纯词典）。
public sealed class ExternalPhonemizerEntry
{
    readonly DiffSingerBasePhonemizer mInstance;
    readonly ILogger mLogger;
    readonly object mLock = new();
    IG2p? mEngine;
    bool mEngineTried;

    public string Lang { get; }
    public string Name { get; }
    public string DictionaryName { get; }     // 如 "dsdict-fr-millefeuille.yaml"（相对预测器目录）
    public string[] Vowels { get; }
    public string[] Consonants { get; }
    public bool HasEngine => mInstance is DiffSingerG2pPhonemizer;

    internal ExternalPhonemizerEntry(string lang, DiffSingerBasePhonemizer instance, string name, ILogger logger)
    {
        Lang = lang;
        Name = name;
        mInstance = instance;
        mLogger = logger;
        DictionaryName = instance.DictionaryName;
        Vowels = (instance as DiffSingerG2pPhonemizer)?.BaseG2pVowels ?? Array.Empty<string>();
        Consonants = (instance as DiffSingerG2pPhonemizer)?.BaseG2pConsonants ?? Array.Empty<string>();
    }

    public IG2p? Engine()
    {
        lock (mLock)
        {
            if (mEngineTried) return mEngine;
            mEngineTried = true;
            try
            {
                mEngine = (mInstance as DiffSingerG2pPhonemizer)?.CreateBaseG2p();
            }
            catch (Exception e)
            {
                mLogger.Warning($"DiffSinger：声库音素器 {Name}（{Lang}）算法引擎初始化失败，降级纯词典：{e.Message}");
                mEngine = null;
            }
            return mEngine;
        }
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// DiffSinger voice 引擎入口。
// 职责：
//   - 经扩展设置接收「声库根目录」与「执行设备」（随宿主持久化、与工程无关）；
//   - 扫描物理模型包、合并为 voice→model→version 注册表（VoiceRegistry），缓存 VoiceSourceInfos；
//   - 解析 (voiceId + part 属性 model/version) → 具体物理包能力集（PartContext），声明面与会话据此工作；
//   - 为每条 part 建一个合成会话。
// 模型以原生 DiffSinger 目录形态存在、与本插件版本解耦：插件不打包模型，只按结构特征扫描发现。
// 设计见 docs/tunelab-voicebank-schema.md。
public sealed class DiffSingerVoiceEngine : IVoiceSynthesisEngine, IExtensionSettings
{
    const string KeyVoicebankDirs = "voicebank_dirs";
    const string KeyExecutionProvider = "execution_provider";
    const string KeySamplingSteps = "sampling_steps";
    const string KeyTensorCache = "tensor_cache";
    const string KeyCacheMaxSizeMb = "cache_max_size_mb";

    public IReadOnlyOrderedMap<string, VoiceSourceInfo> VoiceSourceInfos => mRegistry.Infos;

    public void Init()
    {
        EnsureDefaultDirectory();
        Rescan();
    }

    public void Destroy()
    {
        mModelCache?.Dispose();
        mModelCache = null;
    }

    public IVoiceSynthesisSession CreateSession(IVoiceSynthesisContext context)
    {
        var voiceId = context.VoiceId;
        if (!mRegistry.Contains(voiceId))
            throw new ArgumentException($"未知 voice: {voiceId}");

        var samplingSteps = mSettings.GetInt(KeySamplingSteps, 20);
        var tensorCache = mSettings.GetBool(KeyTensorCache, true);
        var cacheMaxSizeMb = mSettings.GetInt(KeyCacheMaxSizeMb, 4096);
        // 会话按需（每次合成）从 part 属性解析到具体物理包——故传解析委托而非固定 config。
        return new DiffSingerSynthesisSession(
            voiceId, context, rp => Resolve(voiceId, rp), EnsureModelCache(),
            samplingSteps, tensorCache, cacheMaxSizeMb);
    }

    // —— 声明（引擎层、纯函数 of (part 声库 + part 值)；宿主在每次 part 参数 commit 时按当前值重算 diff 到 UI）——
    //   多选 part 跨声库的契约：只暴露所有选中 part 的解析包**共有**的 config 项（按 key 取交集）。
    //   单选即退化为该包全量；无选中 / 全未知 → 空声明（不抛）。
    public IReadOnlyOrderedMap<PropertyKey, AutomationConfig> GetAutomationConfigs(IVoiceSynthesisPartPropertyContext context)
    {
        var merged = context.Parts.Select(p => p.PartProperties).Merge();
        return CommonItems(SelectedContexts(context).Select(pc =>
            (IReadOnlyOrderedMap<PropertyKey, AutomationConfig>)DiffSingerDeclarations.BuildAutomationConfigs(pc, merged)));
    }

    public IReadOnlyOrderedMap<PropertyKey, AutomationConfig> GetSynthesizedParameterConfigs(IVoiceSynthesisPartPropertyContext context)
        => CommonItems(SelectedContexts(context).Select(pc =>
            (IReadOnlyOrderedMap<PropertyKey, AutomationConfig>)DiffSingerDeclarations.BuildReadbackConfigs(pc.Config)));

    public ObjectConfig GetPartPropertyConfig(IVoiceSynthesisPartPropertyContext context)
    {
        var merged = context.Parts.Select(p => p.PartProperties).Merge();
        return new ObjectConfig { Properties = CommonItems(SelectedContexts(context).Select(pc => DiffSingerDeclarations.BuildPartConfig(pc, merged).Properties)) };
    }

    public ObjectConfig GetNotePropertyConfig(IVoiceSynthesisNotePropertyContext context)
        => Resolve(context.Part.VoiceId, context.Part.PartProperties) is { } pc ? DiffSingerDeclarations.BuildNoteConfig(pc, context) : EmptyConfig;

    public IReadOnlyList<ObjectConfig> GetPhonemePropertyConfigs(IVoiceSynthesisNotePropertyContext context)
    {
        if (Resolve(context.Part.VoiceId, context.Part.PartProperties) is not { } pc || !DiffSingerDeclarations.HasLanguageChoice(pc))
            return [];
        var phonemeConfig = DiffSingerDeclarations.BuildPhonemeConfig(pc);
        return context.Notes.SelectMany(n => n.Phonemes).Select(_ => phonemeConfig).ToList();
    }

    // 选中 part 去重后的已知解析上下文集（未知/解析失败过滤掉）。
    IEnumerable<PartContext> SelectedContexts(IVoiceSynthesisPartPropertyContext context)
        => context.Parts.Select(p => Resolve(p.VoiceId, p.PartProperties)).OfType<PartContext>();

    // 解析：voiceId + (model/version 选择) → 物理包能力集。未知 voice / 解析失败 → null。
    PartContext? Resolve(string voiceId, ResolveProps props)
        => mRegistry.Resolve(voiceId, props) is { } resolved
            ? new PartContext(ConfigForRoot(resolved.RootPath), resolved, mRegistry, voiceId)
            : null;

    // 声明面（part 视图给 PropertyObject）用：从属性抽 model/version 再解析。
    PartContext? Resolve(string voiceId, PropertyObject props)
        => Resolve(voiceId, new ResolveProps(
            props.GetString(DiffSingerDeclarations.KeyModel, string.Empty) is { Length: > 0 } m ? m : null,
            props.GetString(DiffSingerDeclarations.KeyVersion, string.Empty) is { Length: > 0 } v ? v : null));

    // 按 key 取多张有序 map 的交集（只保留每张都含的键，顺序与 value 取首张）；零张 → 空 map。
    static OrderedMap<PropertyKey, T> CommonItems<T>(IEnumerable<IReadOnlyOrderedMap<PropertyKey, T>> maps)
    {
        var list = maps.ToList();
        var result = new OrderedMap<PropertyKey, T>();
        if (list.Count == 0)
            return result;
        foreach (var kvp in list[0])
            if (list.All(m => m.ContainsKey(kvp.Key)))
                result.Add(kvp.Key, kvp.Value);
        return result;
    }

    // 声库能力集按物理 RootPath 缓存（解析每次 commit 都调，避免重复解析 dsconfig）；config 随包不可变，扫描重建时清空。
    VoicebankConfig ConfigForRoot(string rootPath)
    {
        if (mConfigCache.TryGetValue(rootPath, out var cached))
            return cached;
        var config = VoicebankConfig.Load(rootPath, TuneLabContext.Global.GetLogger());
        mConfigCache[rootPath] = config;
        return config;
    }

    // 模型缓存按当前执行设备设置懒建；provider 变更则弃旧建新（旧缓存 Dispose 释放原生会话）。
    DiffSingerModelCache EnsureModelCache()
    {
        var provider = mSettings.GetString(KeyExecutionProvider, "directml");
        if (mModelCache == null || mProviderInUse != provider)
        {
            mModelCache?.Dispose();
            mModelCache = new DiffSingerModelCache(provider, TuneLabContext.Global.GetLogger());
            mProviderInUse = provider;
        }
        return mModelCache;
    }

    // —— 扩展设置（设置 > 扩展 面板，随宿主持久化、跨工程共享）——
    public ObjectConfig GetSettingsConfig(IExtensionSettingsContext context)
    {
        var properties = new OrderedMap<PropertyKey, IControllerConfig>
        {
            {
                (KeyVoicebankDirs, L.Tr("Voicebank directories (separate with ;)")),
                new TextBoxConfig { DefaultValue = "" }
            },
            {
                (KeyExecutionProvider, L.Tr("Execution device")),
                new ComboBoxConfig
                {
                    Options = new List<ComboBoxOption>
                    {
                        new("directml", L.Tr("GPU (DirectML)")),
                        new("cpu", L.Tr("CPU")),
                    },
                }
            },
            {
                (KeySamplingSteps, L.Tr("Sampling steps")),
                SliderConfig.Integer(20, 1, 1000)
            },
            {
                (KeyTensorCache, L.Tr("Tensor cache")),
                new CheckBoxConfig { DefaultValue = true }
            },
            {
                (KeyCacheMaxSizeMb, L.Tr("Cache size limit (MB, 0 = unlimited)")),
                SliderConfig.Integer(4096, 0, 102400)
            },
        };
        return new ObjectConfig { Properties = properties };
    }

    public void ApplySettings(PropertyObject settings)
    {
        mSettings = settings;
        Rescan();
    }

    // —— 内部：扫描与状态 ——
    void Rescan()
    {
        var logger = TuneLabContext.Global.GetLogger();
        var roots = CollectRoots();
        var packages = VoicebankScanner.Scan(roots, logger);
        mRegistry = VoiceRegistry.Build(packages, TuneLabContext.Global.Language, logger);
        mConfigCache.Clear();   // 包集变更：弃旧声明缓存（RootPath 可能变）
        logger.Info($"DiffSinger：在 {roots.Count} 个根目录下发现 {packages.Count} 个模型包。");
    }

    List<string> CollectRoots()
    {
        var roots = new List<string> { DefaultVoicebankDirectory };
        var configured = mSettings.GetString(KeyVoicebankDirs, string.Empty);
        roots.AddRange(configured.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
        return roots;
    }

    static string DefaultVoicebankDirectory => Path.Combine(DiffSingerDeclarations.UserDataRoot, "Voices");

    static void EnsureDefaultDirectory()
    {
        try { Directory.CreateDirectory(DefaultVoicebankDirectory); }
        catch { }
    }

    PropertyObject mSettings = PropertyObject.Empty;

    volatile VoiceRegistry mRegistry = VoiceRegistry.Empty;

    readonly Dictionary<string, VoicebankConfig> mConfigCache = new(StringComparer.OrdinalIgnoreCase);
    static readonly ObjectConfig EmptyConfig = new() { Properties = new OrderedMap<PropertyKey, IControllerConfig>() };

    DiffSingerModelCache? mModelCache;
    string mProviderInUse = string.Empty;
}

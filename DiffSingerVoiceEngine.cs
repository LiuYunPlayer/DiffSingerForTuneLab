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
    const string KeyVocoderDirs = "vocoder_dirs";
    const string KeyExecutionProvider = "execution_provider";
    const string KeyRuntimeMode = "runtime_mode";
    const string KeySamplingSteps = "sampling_steps";
    const string KeyTensorCache = "tensor_cache";
    const string KeyCacheMaxSizeMb = "cache_max_size_mb";

    public IReadOnlyOrderedMap<string, VoiceSourceInfo> VoiceSourceInfos => mRegistry.Infos;

    // 选择器分组布局：legacy 多说话人包（一个物理包 spk_embed 展开出的多个 voice）收进一个以声库名命名的组，
    //   不再平铺到顶层；manifest 声库与单说话人 legacy 各自是独立身份、留在顶层（宿主对未引用 id 兜底平铺）。
    public IReadOnlyList<VoiceSourceLayoutItem> VoiceSourceLayout => mRegistry.Layout;

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

        var samplingSteps = (int)mSettings.GetDouble(KeySamplingSteps, 20);
        var tensorCache = mSettings.GetBoolean(KeyTensorCache, true);
        var cacheMaxSizeMb = (int)mSettings.GetDouble(KeyCacheMaxSizeMb, 4096);
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
        return ObjectConfig.Create(CommonItems(SelectedContexts(context).Select(pc => DiffSingerDeclarations.BuildPartConfig(pc, merged).Properties)));
    }

    public ObjectConfig GetNotePropertyConfig(IVoiceSynthesisNotePropertyContext context)
        => Resolve(context.Part.VoiceId, context.Part.PartProperties) is { } pc ? DiffSingerDeclarations.BuildNoteConfig(pc, context) : EmptyConfig;

    // 音素属性声明（核相对 slot 键控 map，口径 = SDK PhonemeSlots）：本插件 schema 只是 part 形状的函数
    //（语言覆盖 + N 组音素混合目标，N = part 属性 phoneme_mix_slots），不依赖任一音素/slot 的当前值——
    //   故所有 slot 授同一份 config，无需按 slot 合并成员值。多语言 / 能力声库才有内容，否则空 map。
    public IReadOnlyMap<int, ObjectConfig> GetPhonemePropertyConfigs(IVoiceSynthesisNotePropertyContext context)
    {
        var map = new Map<int, ObjectConfig>();
        if (Resolve(context.Part.VoiceId, context.Part.PartProperties) is not { } pc)
            return map;
        int slots = DiffSingerDeclarations.HasPhonemeMix(pc) ? DiffSingerDeclarations.MixSlots(context.Part.PartProperties) : 0;
        var phonemeConfig = DiffSingerDeclarations.BuildPhonemeConfig(pc, slots);
        if (phonemeConfig.Properties.Count == 0)
            return map;
        foreach (int slot in context.Notes.UnionSlots())
            map.Add(slot, phonemeConfig);
        return map;
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

    // 声库能力集按物理 RootPath 缓存（解析每次 commit 都调，避免重复解析 dsconfig）；config 随包不可变，扫描重建时清空
    //   （「声码器目录」设置变更 → ApplySettings → Rescan 清缓存 ⇒ pitch_controllable 判定随新目录即时重算）。
    VoicebankConfig ConfigForRoot(string rootPath)
    {
        if (mConfigCache.TryGetValue(rootPath, out var cached))
            return cached;
        var config = VoicebankConfig.Load(rootPath, TuneLabContext.Global.GetLogger(), CollectVocoderRoots());
        mConfigCache[rootPath] = config;
        return config;
    }

    // 模型缓存按当前执行设备设置懒建；provider 变更则弃旧建新（旧缓存 Dispose 释放原生会话）。
    DiffSingerModelCache EnsureModelCache()
    {
        var provider = mSettings.GetString(KeyExecutionProvider, "directml");
        var runtimeMode = mSettings.GetString(KeyRuntimeMode, "subprocess");
        if (mModelCache == null || mProviderInUse != provider || mRuntimeModeInUse != runtimeMode)
        {
            mModelCache?.Dispose();
            mModelCache = new DiffSingerModelCache(provider, runtimeMode, TuneLabContext.Global.GetLogger());
            mProviderInUse = provider;
            mRuntimeModeInUse = runtimeMode;
        }
        mModelCache.VocoderRoots = CollectVocoderRoots();   // 每次刷入：改「声码器目录」设置即时生效，无需重建缓存/重启
        return mModelCache;
    }

    // —— 扩展设置（设置 > 扩展 面板，随宿主持久化、跨工程共享）——
    public ObjectConfig GetSettingsConfig(IExtensionSettingsContext context)
    {
        var properties = new OrderedMap<PropertyKey, IControllerConfig>
        {
            {
                (KeyVoicebankDirs, L.Tr("Voicebank directories")),
                DirListConfig(context.Settings, KeyVoicebankDirs)
            },
            {
                (KeyVocoderDirs, L.Tr("Vocoder directories")),
                DirListConfig(context.Settings, KeyVocoderDirs)
            },
            {
                (KeyExecutionProvider, L.Tr("Execution device")),
                ComboBoxConfig.Create(new List<ComboBoxItem>
                {
                    new(PropertyValue.Create("directml"), L.Tr("GPU (DirectML)")),
                    new(PropertyValue.Create("cpu"), L.Tr("CPU")),
                })
            },
            {
                (KeyRuntimeMode, L.Tr("Inference mode")),
                ComboBoxConfig.Create(new List<ComboBoxItem>
                {
                    new(PropertyValue.Create("subprocess"), L.Tr("Isolated process (recommended)")),
                    new(PropertyValue.Create("inprocess"), L.Tr("In-process")),
                })
            },
            {
                (KeySamplingSteps, L.Tr("Sampling steps")),
                SliderConfig.Integer(20, 1, 1000)
            },
            {
                (KeyTensorCache, L.Tr("Tensor cache")),
                CheckBoxConfig.Create(true)
            },
            {
                (KeyCacheMaxSizeMb, L.Tr("Cache size limit (MB, 0 = unlimited)")),
                SliderConfig.Integer(4096, 0, 102400)
            },
        };
        return ObjectConfig.Create(properties);
    }

    // 目录列表（变长）：每行一个路径 TextBox、+ 追加空行。行数按当前已存值算——
    // 缺席（从未配置）= 0 行、不 seed（默认目录本就隐式生效，无需列在此）。
    static ListConfig DirListConfig(PropertyObject settings, string key)
    {
        int count = settings.GetValue(key, PropertyArray.Empty).Count;
        var elements = Enumerable.Range(0, count)
            .Select(_ => (IControllerConfig)TextBoxConfig.Create())
            .ToList();
        return ListConfig.Create(elements, [new AddableElement(TextBoxConfig.Create(), L.Tr("Directory"))]);
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
        roots.AddRange(ConfiguredDirs(KeyVoicebankDirs));
        return roots;
    }

    // 声码器搜索根 = 默认 Vocoders 目录 + 用户追加。与声库目录同构：默认始终生效、追加项在后。
    List<string> CollectVocoderRoots()
    {
        var roots = new List<string> { DiffSingerModelCache.VocodersDirectory };
        roots.AddRange(ConfiguredDirs(KeyVocoderDirs));
        return roots;
    }

    // 从设置读用户配置的目录列表（PropertyArray of string）：跳过空白行、去首尾空白。
    IEnumerable<string> ConfiguredDirs(string key)
    {
        foreach (var v in mSettings.GetValue(key, PropertyArray.Empty))
            if (v.ToString(out var s) && !string.IsNullOrWhiteSpace(s))
                yield return s.Trim();
    }

    static string DefaultVoicebankDirectory => Path.Combine(DiffSingerDeclarations.UserDataRoot, "Voices");

    static void EnsureDefaultDirectory()
    {
        try { Directory.CreateDirectory(DefaultVoicebankDirectory); }
        catch { }
        try { Directory.CreateDirectory(DiffSingerModelCache.VocodersDirectory); }
        catch { }
    }

    PropertyObject mSettings = PropertyObject.Empty;

    volatile VoiceRegistry mRegistry = VoiceRegistry.Empty;

    readonly Dictionary<string, VoicebankConfig> mConfigCache = new(StringComparer.OrdinalIgnoreCase);
    static readonly ObjectConfig EmptyConfig = ObjectConfig.Create(new OrderedMap<PropertyKey, IControllerConfig>());

    DiffSingerModelCache? mModelCache;
    string mProviderInUse = string.Empty;
    string mRuntimeModeInUse = string.Empty;
}

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 声明面（解析上下文 PartContext 的纯函数）：自动化轨 / 回显轨 / part 面板 / note 面板。
//   据 use_*_embed 暴露可编辑曲线、据 predict_* 暴露只读回显轨；据 manifest 暴露 voice→model→version 与白名单。
//   无 manifest（legacy）→ 退化为今天行为：整模型 1 voice、speaker 下拉 + 混音容器。
//   有 manifest → voice 即说话人、暴露 model/version 下拉、混音候选 = 同包其它暴露 voice、seed 轨按 retake 声明 gating。
// 规格集中在此、单一真相源；会话运行时复用本类的轨 key 与 variance/gender/speed/speaker 集合。设计见 docs/tunelab-voicebank-schema.md。
public static class DiffSingerDeclarations
{
    // 插件独立用户数据根：Voices（默认声库扫描根）/ Vocoders（声码器）/ Cache（张量缓存）均由此派生。
    public static string UserDataRoot => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "DiffSingerForTuneLab");

    // —— 暴露给用户的参数键（避开宿主保留名 Volume / VibratoEnvelope）——
    public const string KeyGender = "gender";
    public const string KeySpeed = "speed";
    public const string KeySpeaker = "speaker";    // legacy：part 默认说话人下拉
    public const string KeyLanguage = "language";
    public const string KeyModel = "model";        // manifest：模型下拉（part 属性）
    public const string KeyVersion = "version";    // manifest：版本下拉（part 属性，"" = 最新跟随）
    // 随机种子轨（逐帧 = 时间维×值维 → 区域独立 take）：按 manifest retake 三位 gating。
    public const string KeySeedPitch = "seed_pitch";
    public const string KeySeedVariance = "seed_variance";
    public const string KeySeedAcoustic = "seed_acoustic";
    // seed 轨归一化为 [0,1]（最通用刻度，别的 retake 引擎天然兼容、撞键也不越界）；
    //   合成期 round(v·uint.MaxValue) 放大到 uint32 喂位置寻址哈希（哈希白化 → 刻度不影响噪声质量，见 DiffSingerNoise）。
    public const double SeedCurveMax = 1;
    public const string KeyMix = "speaker_mix";    // part 属性：说话人混合变长键控容器，条目键 = suffix
    public const string KeyMixPrefix = "mix:";     // 说话人混合自动化轨 key 前缀：mix:<suffix>

    // 归一化刻度：gender 中性 0、量程 [-1,1]；speed 中性 1（=原速）、量程 [0,2]（百分比小数化）。
    public const double GenderBaseline = 0, GenderMin = -1, GenderMax = 1;
    public const double SpeedBaseline = 1, SpeedMin = 0, SpeedMax = 2;

    public readonly record struct VarianceSpec(
        string Key, string Display, string Color,
        Func<VoicebankConfig, bool> Use, Func<VoicebankConfig, bool> Predict,
        double EditMin, double EditMax, double Neutral,
        double AcousticMin, double AcousticMax,
        Func<float, float, float> Delta);

    // 编辑轨（delta 语义）归一化到小数：energy/breath/tension 中性 0、量程 [-1,1]；voicing 中性 1、量程 [0,1]。
    //   Delta(x=预测声学值, y=用户归一化值) 系数随之 ×100：y=1 等价旧 y=100（energy/breath ±12dB、voicing ±12dB、tension ±5）。
    //   AcousticMin/Max 为回显轨的真实声学单位（dB）值域，保持不变。
    public static readonly VarianceSpec[] Variances =
    {
        new("energy",      "Energy",      "#E573A5", c => c.UseEnergyEmbed,      c => c.PredictEnergy,      -1, 1, 0, -96, 0, (x, y) => x + y * 12),
        new("breathiness", "Breathiness", "#73E5C2", c => c.UseBreathinessEmbed, c => c.PredictBreathiness, -1, 1, 0, -96, 0, (x, y) => x + y * 12),
        new("voicing",     "Voicing",     "#C2E573", c => c.UseVoicingEmbed,     c => c.PredictVoicing,      0, 1, 1, -96, 0, (x, y) => x + (y - 1) * 12),
        new("tension",     "Tension",     "#A573E5", c => c.UseTensionEmbed,     c => c.PredictTension,     -1, 1, 0, -10, 10, (x, y) => x + y * 5),
    };

    // manifest retake 三位（legacy → 全 false ⇒ 不暴露任何 seed 轨）。
    public static (bool Acoustic, bool Pitch, bool Variance) RetakeOf(ResolvedVoice resolved)
        => resolved.Manifest is { } m ? (m.RetakeAcoustic, m.RetakePitch, m.RetakeVariance) : (false, false, false);

    // —— 自动化轨（可编辑曲线）= 固定轨 + 已启用的说话人混合轨 ——
    public static OrderedMap<PropertyKey, AutomationConfig> BuildAutomationConfigs(PartContext pc, PropertyObject partProperties)
    {
        var map = BuildFixedAutomationConfigs(pc.Config, RetakeOf(pc.Resolved));
        var set = SpeakerSet.Compute(pc.Resolved);
        foreach (var (suffix, display, color) in SelectedMixTracks(set, partProperties))
            map.Add((KeyMixPrefix + suffix, display), Continuous(color, 0, 0, 1));   // 混合权重归一化 [0,1]（0=不混入）
        return map;
    }

    // 固定轨（与 part 属性无关）：variance（按声库能力）+ Gender/Speed + seed（按 retake gating）。
    public static OrderedMap<PropertyKey, AutomationConfig> BuildFixedAutomationConfigs(
        VoicebankConfig config, (bool Acoustic, bool Pitch, bool Variance) retake)
    {
        var map = new OrderedMap<PropertyKey, AutomationConfig>();
        foreach (var v in Variances)
            if (v.Use(config))
                map.Add((v.Key, L.Tr(v.Display)), Continuous(v.Color, v.Neutral, v.EditMin, v.EditMax));

        if (config.UseKeyShiftEmbed)
            map.Add((KeyGender, L.Tr("Gender")), Continuous("#E5A573", GenderBaseline, GenderMin, GenderMax));
        if (config.UseSpeedEmbed)
            map.Add((KeySpeed, L.Tr("Speed")), Continuous("#73B5E5", SpeedBaseline, SpeedMin, SpeedMax));

        // seed 轨：仅当 manifest 声明对应 retake 能力时暴露（连续、基线 0、量程 [0,SeedCurveMax]）。
        if (retake.Pitch)
            map.Add((KeySeedPitch, L.Tr("Pitch seed")), Continuous("#9E9E9E", 0, 0, SeedCurveMax, randomizable: true));
        if (retake.Variance)
            map.Add((KeySeedVariance, L.Tr("Variance seed")), Continuous("#9E9E9E", 0, 0, SeedCurveMax, randomizable: true));
        if (retake.Acoustic)
            map.Add((KeySeedAcoustic, L.Tr("Timbre seed")), Continuous("#9E9E9E", 0, 0, SeedCurveMax, randomizable: true));
        return map;
    }

    // 已启用混合的说话人 (suffix, 显示名, 稳定配色)：遍历说话人集合，排除默认 speaker，
    //   过滤出 part 属性 speaker_mix 容器里已存在的键。配色按固定索引轮转（manifest voice 自带 color 则优先）。
    public static IEnumerable<(string Suffix, string Display, string Color)> SelectedMixTracks(SpeakerSet set, PropertyObject partProperties)
    {
        var selected = partProperties.GetObject(KeyMix);
        int i = 0;
        foreach (var opt in set.Options)
        {
            string color = opt.Color ?? MixColors[i % MixColors.Length];
            i++;
            if (opt.Suffix == set.DefaultSuffix)
                continue;
            if (selected.Map.ContainsKey(opt.Suffix))
                yield return (opt.Suffix, opt.Display, color);
        }
    }

    // 只读回显轨：仅当声学接受该量为输入且方差器能产基线时——显示方差器纯预测。
    public static OrderedMap<PropertyKey, AutomationConfig> BuildReadbackConfigs(VoicebankConfig config)
    {
        var map = new OrderedMap<PropertyKey, AutomationConfig>();
        foreach (var v in Variances)
            if (v.Use(config) && v.Predict(config))
                map.Add((v.Key, L.Tr(v.Display)), Piecewise(v.Color, v.AcousticMin, v.AcousticMax));
        return map;
    }

    // part 级面板：manifest 暴露 model/version 下拉；legacy 暴露 speaker 下拉；两者皆可有混音容器 + 默认语言。
    public static ObjectConfig BuildPartConfig(PartContext pc, PropertyObject mergedProps)
    {
        var properties = new OrderedMap<PropertyKey, IControllerConfig>();
        var registry = pc.Registry;

        // 模型下拉（含此 voice 的模型 >1 时）：默认 = released 最新（浮动）。
        if (registry.ModelOptions(pc.VoiceId) is { } modelOptions)
        {
            properties.Add((KeyModel, L.Tr("Model")), ComboBoxConfig
                .Create(modelOptions.Select(o => new ComboBoxItem(PropertyValue.Create(o.Value), o.Display)).ToList())
                .WithDefault(PropertyValue.Create(string.Empty)));   // "" = 最新 sentinel（浮动跟随最新模型）
        }
        // 版本下拉（当前选中模型版本 >1 时）：首项 "" = 最新（跟随）。
        string? selectedModel = mergedProps.GetString(KeyModel, string.Empty) is { Length: > 0 } sm ? sm : null;
        if (registry.VersionOptions(pc.VoiceId, selectedModel) is { } versionOptions)
        {
            properties.Add((KeyVersion, L.Tr("Version")), ComboBoxConfig
                .Create(versionOptions.Select(o => new ComboBoxItem(PropertyValue.Create(o.Value), o.Display)).ToList())
                .WithDefault(PropertyValue.Create(string.Empty)));
        }

        // 默认语言：多语言切换属高频操作，排在进阶的混音容器之上。
        if (HasLanguageChoice(pc))
            properties.Add((KeyLanguage, L.Tr("Language")), LanguageCombo(EffectiveLanguages(pc), DefaultLanguageId(pc.Config, pc.Resolved)));

        // 说话人混合容器：候选 = 同包暴露 voice 中除当前 voice 外的（legacy 即旧 speaker 下拉里的其他人）。
        var set = SpeakerSet.Compute(pc.Resolved);
        if (set.Options.Any(o => o.Suffix != set.DefaultSuffix))
            properties.Add((KeyMix, L.Tr("Speaker mix")), BuildSpeakerMixConfig(set, mergedProps));

        return ObjectConfig.Create(properties);
    }

    // 说话人混合容器：Properties = 当前已选（present 键）；AddableElements = 候选（除默认）。条目皆纯 presence。
    static ExtensibleObjectConfig BuildSpeakerMixConfig(SpeakerSet set, PropertyObject partProperties)
    {
        var selected = partProperties.GetObject(KeyMix);
        var props = new OrderedMap<PropertyKey, IControllerConfig>();
        var addable = new List<AddableKey>();
        foreach (var opt in set.Options)
        {
            if (opt.Suffix == set.DefaultSuffix)
                continue;
            if (selected.Map.ContainsKey(opt.Suffix))
                props.Add((opt.Suffix, opt.Display), EmptyEntry());
            addable.Add(new AddableKey((opt.Suffix, opt.Display), EmptyEntry()));
        }
        return ExtensibleObjectConfig.Create(props, addable);
    }

    static ObjectConfig EmptyEntry() => ObjectConfig.Create(new OrderedMap<PropertyKey, IControllerConfig>());

    // note 级面板：多语言暴露 per-note 语言覆盖；默认值取 part 当前默认语言。
    public static ObjectConfig BuildNoteConfig(PartContext pc, IVoiceSynthesisNotePropertyContext context)
    {
        var properties = new OrderedMap<PropertyKey, IControllerConfig>();
        if (HasLanguageChoice(pc))
        {
            var partDefault = context.Part.PartProperties.GetString(KeyLanguage, DefaultLanguageId(pc.Config, pc.Resolved));
            properties.Add((KeyLanguage, L.Tr("Language")), LanguageCombo(EffectiveLanguages(pc), partDefault));
        }
        return ObjectConfig.Create(properties);
    }

    // per-phoneme 语言：每个钉死音素一个语言下拉，覆盖该音素语种；默认空 = 跟随 note。
    public static ObjectConfig BuildPhonemeConfig(PartContext pc)
    {
        var options = new List<ComboBoxItem> { new(PropertyValue.Create(string.Empty), L.Tr("(follow note)")) };
        foreach (var (id, display) in EffectiveLanguages(pc))
            options.Add(new ComboBoxItem(PropertyValue.Create(id), display));
        var props = new OrderedMap<PropertyKey, IControllerConfig>();
        props.Add((KeyLanguage, L.Tr("Language")), ComboBoxConfig.Create(options).WithDefault(PropertyValue.Create(string.Empty)));
        return ObjectConfig.Create(props);
    }

    public static bool HasLanguageChoice(PartContext pc)
        => pc.Config.UseLanguageId && EffectiveLanguages(pc).Count > 1;

    // 有效语言 (id, 显示名)：id 来自 dsconfig 语言表；manifest 的 languages.expose 非空则按其白名单 + 顺序 + 显示名/i18n 叠加。
    public static List<(string Id, string Display)> EffectiveLanguages(PartContext pc)
    {
        var configLangs = pc.Config.Languages;
        var manifest = pc.Resolved.Manifest;
        if (manifest is not null && manifest.Languages.Count > 0)
        {
            var set = new HashSet<string>(configLangs, StringComparer.Ordinal);
            var result = manifest.Languages
                .Where(l => set.Contains(l.Id))
                .Select(l => (l.Id, I18n.Resolve(string.IsNullOrWhiteSpace(l.Name) ? l.Id : l.Name!, l.NameI18n, HostLang)))
                .ToList();
            if (result.Count > 0)
                return result;
        }
        return configLangs.Select(id => (id, id)).ToList();
    }

    // 默认语言：manifest 模型级 default → 当前 voice 的 default → 有效语言首项。
    public static string DefaultLanguageId(VoicebankConfig config, ResolvedVoice resolved)
    {
        var ids = new HashSet<string>(config.Languages, StringComparer.Ordinal);
        if (resolved.Manifest?.DefaultLanguage is { } md && ids.Contains(md))
            return md;
        if (resolved.CurrentVoice?.DefaultLanguage is { } vd && ids.Contains(vd))
            return vd;
        return config.Languages.Count > 0 ? config.Languages[0] : string.Empty;
    }

    static string? HostLang => TuneLabContext.Global.Language;

    static ComboBoxConfig LanguageCombo(List<(string Id, string Display)> langs, string defaultValue)
        => ComboBoxConfig
            .Create(langs.Select(l => new ComboBoxItem(PropertyValue.Create(l.Id), l.Display)).ToList())
            .WithDefault(PropertyValue.Create(defaultValue));


    // 说话人混合轨 (key=mix:<suffix>, suffix)：每候选一条；同 suffix 去重。会话据此订阅 + 渲染期解析。
    public static IEnumerable<(string Key, string Suffix)> SpeakerMixTracks(SpeakerSet set)
    {
        foreach (var opt in set.Options)
            yield return (KeyMixPrefix + opt.Suffix, opt.Suffix);
    }

    // 全部候选 mix 轨 key（从解析包的 ExposedVoices）——供会话订阅期使用（彼时只有实时属性、无 PropertyObject 快照）。
    public static IEnumerable<(string Key, string Suffix)> MixTrackKeys(ResolvedVoice resolved)
    {
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var v in resolved.ExposedVoices)
        {
            var suffix = Suffix(v.Speaker);
            if (!string.IsNullOrEmpty(suffix) && seen.Add(suffix))
                yield return (KeyMixPrefix + suffix, suffix);
        }
    }

    // dsconfig 说话人条目 → 显示/匹配用 suffix（"260509a.Miku" → "Miku"；无点则原样）。
    public static string Suffix(string speaker)
    {
        int dot = speaker.LastIndexOf('.');
        return dot >= 0 && dot < speaker.Length - 1 ? speaker[(dot + 1)..] : speaker;
    }

    static readonly string[] MixColors =
    {
        "#E57373", "#F06292", "#BA68C8", "#9575CD", "#7986CB", "#64B5F6",
        "#4DD0E1", "#4DB6AC", "#81C784", "#DCE775", "#FFD54F", "#FFB74D",
    };

    // 分段轨：不调 WithDefault → DefaultValue 保持 NaN（= 分段、无基线）。
    static AutomationConfig Piecewise(string color, double min, double max)
        => AutomationConfig.Create(min, max).WithColor(color);

    // 连续轨：WithDefault 给实数基线 → 连续；按需 WithRandomizable。
    static AutomationConfig Continuous(string color, double baseline, double min, double max, bool randomizable = false)
        => AutomationConfig.Create(min, max).WithDefault(baseline).WithColor(color).WithRandomizable(randomizable);
}

// 一个 part 当前的“可用说话人集合”：驱动默认说话人、混音候选/轨、嵌入解析。
//   统一从解析包的 ExposedVoices 取（manifest 显式声明；legacy 由 dsconfig speakers 自动派生）——无 legacy 特例分支。
//   Options = 同包暴露 voices（显示=voice 名，可 i18n / 自带 color）；默认 = 当前 voice 的 speaker。
public sealed class SpeakerSet
{
    public IReadOnlyList<SpeakerOption> Options { get; }
    public string DefaultSuffix { get; }

    SpeakerSet(IReadOnlyList<SpeakerOption> options, string defaultSuffix)
    {
        Options = options;
        DefaultSuffix = defaultSuffix;
    }

    public static SpeakerSet Compute(ResolvedVoice resolved)
    {
        string? host = TuneLabContext.Global.Language;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var options = new List<SpeakerOption>();
        foreach (var v in resolved.ExposedVoices)
        {
            var suffix = DiffSingerDeclarations.Suffix(v.Speaker);
            if (string.IsNullOrEmpty(suffix) || !seen.Add(suffix))
                continue;
            var display = I18n.Resolve(string.IsNullOrWhiteSpace(v.Name) ? suffix : v.Name!, v.NameI18n, host);
            options.Add(new SpeakerOption(suffix, display, v.Color));
        }
        return new SpeakerSet(options, DiffSingerDeclarations.Suffix(resolved.VoiceSpeaker ?? string.Empty));
    }
}

public sealed record SpeakerOption(string Suffix, string Display, string? Color);

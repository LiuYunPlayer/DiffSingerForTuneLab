using System;
using System.Collections.Generic;
using System.IO;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 声明面（声库能力集 VoicebankConfig 的纯函数）：自动化轨 / 回显轨 / part 面板 / note 面板。
//   据 use_*_embed 暴露可编辑曲线、据 predict_* 暴露只读回显轨、据 speakers/languages 暴露 part/note 属性。
// SDK 把声明上移到 IVoiceEngine（不依赖会话实例、宿主在「建会话之前」即填好 Voice.AutomationConfigs）：
//   引擎据 context.VoiceId 取 config 调本类建声明；会话在运行时复用此处的轨 key 与 variance/gender/speed 规格
//   （故规格集中在此、单一真相源）。见 IVoiceEngine 注释与 MidiPart 的 RefreshDeclarations→CreateSession 时序。
public static class DiffSingerDeclarations
{
    // 插件独立用户数据根：在 TuneLab 宿主目录之外（宿主目录只存宿主自己的数据，外来插件应有独立用户目录）。
    //   Voices（默认声库扫描根）/ Vocoders（声码器）/ Cache（张量缓存）三处目录单一来源、均由此派生。
    public static string UserDataRoot => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "DiffSingerForTuneLab");

    // —— 暴露给用户的参数键（避开宿主保留名 Volume / VibratoEnvelope）——
    public const string KeyGender = "gender";
    public const string KeySpeed = "speed";
    public const string KeySpeaker = "speaker";
    public const string KeyLanguage = "language";
    public const string KeyMix = "speaker_mix";  // part 属性：说话人混合变长键控容器（ExtensibleObjectConfig），条目键 = suffix
    public const string KeyMixPrefix = "mix:";   // 说话人混合自动化轨 key 前缀：mix:<suffix>

    // Gender(GENC) / Speed(VELC) 连续轨：忠实采 OpenUtau 原生 UI 量程（非半音/倍率），convert 据此逐字移植。
    //   · GENC ∈ [-100,100]，基线 0 = 不移位，正 = formant 下移；增广范围 KeyShift* 仅入 convert 缩放，不做轨边界。
    //   · VELC ∈ [0,200]，基线 100 = 原速，对数（每 +100 速度 ×2）；纯帧级声学条件，不改 durations / 帧数。
    public const double GenderBaseline = 0, GenderMin = -100, GenderMax = 100;
    public const double SpeedBaseline = 100, SpeedMin = 0, SpeedMax = 200;

    // 四个 variance 量的声明规格 + 合成语义（单一真相源，忠实移植 OpenUtau）：
    //   · 可编辑轨 = 增量 delta（OpenUtau UI 单位 [EditMin,EditMax]）；连续轨、基线=Neutral（处处有值）。
    //     Use(config) ⇒ 暴露此轨。日后若加「绝对覆盖式实参」，在 Render 的合成处另开分支即可，量程表不动。
    //   · 合成喂声学 = clamp(Delta(预测 x, 用户 y), AcousticMin, AcousticMax)；Neutral 代入 Delta 恒得纯预测。
    //   · 回显轨 = 纯预测值（同声学单位 [AcousticMin,AcousticMax]）；Use && Predict ⇒ 附此只读轨。
    // 注：voicing 的中性值为 100（OpenUtau 语义：默认 100、量程 [0,100]、只降不升），故其 NaN 自由区对应 100=纯预测。
    public readonly record struct VarianceSpec(
        string Key, string Display, string Color,
        Func<VoicebankConfig, bool> Use, Func<VoicebankConfig, bool> Predict,
        double EditMin, double EditMax, double Neutral,
        double AcousticMin, double AcousticMax,
        Func<float, float, float> Delta);

    public static readonly VarianceSpec[] Variances =
    {
        new("energy",      "Energy",      "#E573A5", c => c.UseEnergyEmbed,      c => c.PredictEnergy,      -100, 100,   0, -96, 0, (x, y) => x + y * 12 / 100),
        new("breathiness", "Breathiness", "#73E5C2", c => c.UseBreathinessEmbed, c => c.PredictBreathiness, -100, 100,   0, -96, 0, (x, y) => x + y * 12 / 100),
        new("voicing",     "Voicing",     "#C2E573", c => c.UseVoicingEmbed,     c => c.PredictVoicing,        0, 100, 100, -96, 0, (x, y) => x + (y - 100) * 12 / 100),
        new("tension",     "Tension",     "#A573E5", c => c.UseTensionEmbed,     c => c.PredictTension,     -100, 100,   0, -10, 10, (x, y) => x + y / 20),
    };

    // 可编辑曲线 = 固定轨（variance / Gender / Speed，与 part 属性无关）+ 已启用的说话人混合轨（f(part 属性)）。
    //   必须连续（非分段）：宿主仅把连续轨接进合成；中性基线上画偏移正是 delta 的天然形态。
    //   mix 轨是动态集：用户在 part 面板 speaker_mix 容器（ExtensibleObjectConfig）+ 一个 speaker 才出现其曲线、删除即消失；
    //   宿主既有 OnPartPropertiesModified→RebuildAutomationConfigs→AutomationConfigsModified 链驱动曲线按钮自动增减（免新增 wiring）。
    public static OrderedMap<PropertyKey, AutomationConfig> BuildAutomationConfigs(VoicebankConfig config, PropertyObject partProperties)
    {
        var map = BuildFixedAutomationConfigs(config);
        foreach (var (suffix, color) in SelectedMixTracks(config, partProperties))
            map.Add((KeyMixPrefix + suffix, suffix), Continuous(color, 0, 0, 100));   // [0,100]、基线 0；轨名 = suffix（非 UI 文案、不过 L.Tr）
        return map;
    }

    // 固定轨（与 part 属性无关）：variance（按声库能力）+ Gender/Speed。会话据此 key 集在构造期一次性订阅（恒定不随属性变）。
    public static OrderedMap<PropertyKey, AutomationConfig> BuildFixedAutomationConfigs(VoicebankConfig config)
    {
        var map = new OrderedMap<PropertyKey, AutomationConfig>();
        foreach (var v in Variances)
            if (v.Use(config))
                map.Add((v.Key, L.Tr(v.Display)), Continuous(v.Color, v.Neutral, v.EditMin, v.EditMax));

        if (config.UseKeyShiftEmbed)
            map.Add((KeyGender, L.Tr("Gender")), Continuous("#E5A573", GenderBaseline, GenderMin, GenderMax));
        if (config.UseSpeedEmbed)
            map.Add((KeySpeed, L.Tr("Speed")), Continuous("#73B5E5", SpeedBaseline, SpeedMin, SpeedMax));
        return map;
    }

    // 主/兜底 speaker 的 suffix（part 属性 KeySpeaker，缺省取声库首个）。它恒占混合一席（defaultWeight）、
    //   不作混合候选——故下方候选/已选/automation 轨一律排除它，避免双重计入与自混冗余。可变：随 KeySpeaker 改。
    public static string DefaultSuffix(VoicebankConfig config, PropertyObject partProperties)
        => Suffix(partProperties.GetString(KeySpeaker, config.Speakers.Count > 0 ? config.Speakers[0] : string.Empty));

    // 已启用混合的说话人 (suffix, 稳定配色)：遍历全量去重 speaker 表（配色按其固定索引轮转，保证同一 speaker
    //   不论何时加入颜色一致），排除主/兜底 speaker，过滤出 part 属性 speaker_mix 容器里已存在的键（present = 用户已 + 该 speaker）。
    public static IEnumerable<(string Suffix, string Color)> SelectedMixTracks(VoicebankConfig config, PropertyObject partProperties)
    {
        var selected = partProperties.GetObject(KeyMix);
        var defaultSuffix = DefaultSuffix(config, partProperties);
        int i = 0;
        foreach (var (key, suffix) in SpeakerMixTracks(config))
        {
            var color = MixColors[i++ % MixColors.Length];   // i 覆盖全量（含默认）→ 颜色不随默认变更漂移
            if (suffix == defaultSuffix)
                continue;
            if (selected.Map.ContainsKey(suffix))
                yield return (suffix, color);
        }
    }

    // 只读回显轨：仅当声学接受该量为输入且方差器能产基线时——显示方差器纯预测（内容感知基线，真实声学单位）。
    public static OrderedMap<PropertyKey, AutomationConfig> BuildReadbackConfigs(VoicebankConfig config)
    {
        var map = new OrderedMap<PropertyKey, AutomationConfig>();
        foreach (var v in Variances)
            if (v.Use(config) && v.Predict(config))
                map.Add((v.Key, L.Tr(v.Display)), Piecewise(v.Color, v.AcousticMin, v.AcousticMax));
        return map;
    }

    // part 级面板：多说话人暴露「主/兜底 speaker 选择 + 说话人混合容器」、多语言暴露 part 默认语言。
    public static ObjectConfig BuildPartConfig(VoicebankConfig config, IPartPropertyContext context)
    {
        var properties = new OrderedMap<PropertyKey, IControllerConfig>();

        if (config.Speakers.Count > 1)
        {
            // 主/兜底 speaker：不画任何混合时的单说话人，且逐帧混合权重和不足 1 时由它补足（见 DiffSingerSpeakerMix）。
            properties.Add((KeySpeaker, L.Tr("Speaker")), new ComboBoxConfig
            {
                Options = SpeakerOptions(config.Speakers),
            });
            // 说话人混合：变长键控容器（ExtensibleObjectConfig）。用户从 + 菜单挑 speaker 加入（纯开关、空对象条目），
            //   加入即出现该 speaker 的 [0,100] 逐帧混合曲线、删除即消失——免一次平铺所有 speaker 的曲线。
            properties.Add((KeyMix, L.Tr("Speaker mix")), BuildSpeakerMixConfig(config, context.PartProperties));
        }

        if (HasLanguageChoice(config))
            properties.Add((KeyLanguage, L.Tr("Language")), LanguageCombo(config, config.Languages[0]));

        return new ObjectConfig { Properties = properties };
    }

    // 说话人混合容器：Properties = 当前已选 speaker（读 part 属性、present 键）；AddableElements = 全量去重候选
    //   （宿主在 + 菜单隐藏已存在的键）。条目皆纯 presence（空 ObjectConfig）：加入=启用混合、删除=禁用，权重靠曲线。
    static ExtensibleObjectConfig BuildSpeakerMixConfig(VoicebankConfig config, PropertyObject partProperties)
    {
        var selected = partProperties.GetObject(KeyMix);
        var defaultSuffix = DefaultSuffix(config, partProperties);
        var props = new OrderedMap<PropertyKey, IControllerConfig>();
        var addable = new List<AddableKey>();
        foreach (var (key, suffix) in SpeakerMixTracks(config))   // 全量去重，suffix 既作条目键 Id 又作显示文本
        {
            if (suffix == defaultSuffix)   // 主/兜底 speaker 不作混合候选（已恒占一席）
                continue;
            if (selected.Map.ContainsKey(suffix))   // 容器里若残留它（曾选后又被设为默认），一并排除、不渲染
                props.Add((suffix, suffix), EmptyEntry());
            addable.Add(new AddableKey((suffix, suffix), EmptyEntry()));
        }
        return new ExtensibleObjectConfig { Properties = props, AddableElements = addable };
    }

    static ObjectConfig EmptyEntry() => new() { Properties = new OrderedMap<PropertyKey, IControllerConfig>() };

    // note 级面板：多语言声库暴露 per-note 语言覆盖；默认值取 part 当前默认语言（依赖 part 值 ⇒ 逐次构建）。
    public static ObjectConfig BuildNoteConfig(VoicebankConfig config, INotePropertyContext context)
    {
        var properties = new OrderedMap<PropertyKey, IControllerConfig>();
        if (HasLanguageChoice(config))
        {
            var partDefault = context.PartProperties.GetString(KeyLanguage, config.Languages[0]);
            properties.Add((KeyLanguage, L.Tr("Language")), LanguageCombo(config, partDefault));
        }
        return new ObjectConfig { Properties = properties };
    }

    public static bool HasLanguageChoice(VoicebankConfig config) => config.UseLanguageId && config.Languages.Count > 1;

    static ComboBoxConfig LanguageCombo(VoicebankConfig config, string defaultValue) => new()
    {
        Options = ToOptions(config.Languages),
        DefaultOption = PropertyValue.Create(defaultValue),
    };

    static List<ComboBoxOption> LanguageOptions(IReadOnlyList<string> languages)
    {
        var options = new List<ComboBoxOption> { new(PropertyValue.Create(string.Empty), "default") };
        foreach (var lang in languages)
            options.Add(lang);
        return options;
    }

    // 说话人：值保留 dsconfig 原始条目（下游据此选 .emb），显示去模型名前缀（"260509a.Miku" → "Miku"）。
    static List<ComboBoxOption> SpeakerOptions(IReadOnlyList<string> speakers)
    {
        var options = new List<ComboBoxOption>(speakers.Count);
        foreach (var speaker in speakers)
            options.Add(new ComboBoxOption(PropertyValue.Create(speaker), Suffix(speaker)));
        return options;
    }

    // 说话人混合轨 (key=mix:<suffix>, suffix)：多说话人时每 speaker 一条；同 suffix 去重（取首个，对齐 OpenUtau 按 suffix 解析）。
    public static IEnumerable<(string Key, string Suffix)> SpeakerMixTracks(VoicebankConfig config)
    {
        if (config.Speakers.Count <= 1)
            yield break;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var speaker in config.Speakers)
        {
            var suffix = Suffix(speaker);
            if (seen.Add(suffix))
                yield return (KeyMixPrefix + suffix, suffix);
        }
    }

    // dsconfig 说话人条目 → 显示/匹配用 suffix（"260509a.Miku" → "Miku"；无点则原样）。
    public static string Suffix(string speaker)
    {
        int dot = speaker.LastIndexOf('.');
        return dot >= 0 && dot < speaker.Length - 1 ? speaker[(dot + 1)..] : speaker;
    }

    // 说话人混合轨配色（按声明顺序轮转）。
    static readonly string[] MixColors =
    {
        "#E57373", "#F06292", "#BA68C8", "#9575CD", "#7986CB", "#64B5F6",
        "#4DD0E1", "#4DB6AC", "#81C784", "#DCE775", "#FFD54F", "#FFB74D",
    };

    // 轨名（DisplayText）随 PropertyKey 走、不属 config 本身（SDK 改动）——故此处只产纯量程/配色 config。
    static AutomationConfig Piecewise(string color, double min, double max) => new()
    {
        DefaultValue = double.NaN,   // 分段：无基线、段间断开（NaN 自由区）
        MinValue = min,
        MaxValue = max,
        Color = color,
    };

    static AutomationConfig Continuous(string color, double baseline, double min, double max) => new()
    {
        DefaultValue = baseline,
        MinValue = min,
        MaxValue = max,
        Color = color,
    };
}

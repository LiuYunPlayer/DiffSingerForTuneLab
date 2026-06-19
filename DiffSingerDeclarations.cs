using System;
using System.Collections.Generic;
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
    // —— 暴露给用户的参数键（避开宿主保留名 Volume / VibratoEnvelope）——
    public const string KeyGender = "gender";
    public const string KeySpeed = "speed";
    public const string KeySpeaker = "speaker";
    public const string KeyLanguage = "language";
    public const string KeyMixPrefix = "mix:";   // 说话人混合轨 key 前缀：mix:<suffix>

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

    // 可编辑曲线：variance 量（连续 delta，基线=中性值=纯预测）+ Gender/Speed（连续，基线为中性值）。
    //   必须连续（非分段）：宿主仅把连续轨接进合成；中性基线上画偏移正是 delta 的天然形态。
    public static OrderedMap<string, AutomationConfig> BuildAutomationConfigs(VoicebankConfig config)
    {
        var map = new OrderedMap<string, AutomationConfig>();
        foreach (var v in Variances)
            if (v.Use(config))
                map.Add(v.Key, Continuous(v.Display, v.Color, v.Neutral, v.EditMin, v.EditMax));

        if (config.UseKeyShiftEmbed)
            map.Add(KeyGender, Continuous("Gender", "#E5A573", GenderBaseline, GenderMin, GenderMax));
        if (config.UseSpeedEmbed)
            map.Add(KeySpeed, Continuous("Speed", "#73B5E5", SpeedBaseline, SpeedMin, SpeedMax));

        // 说话人混合：多说话人时每 speaker 一条逐帧权重轨（连续、[0,100]、基线 0），mix:<suffix>。
        //   不画时全权重落到 part 级 KeySpeaker（默认 suffix）⇒ 等价单说话人广播；画上即逐帧混入该 speaker。
        int mixColorIndex = 0;
        foreach (var (key, suffix) in SpeakerMixTracks(config))
            map.Add(key, Continuous(suffix, MixColors[mixColorIndex++ % MixColors.Length], 0, 0, 100));
        return map;
    }

    // 只读回显轨：仅当声学接受该量为输入且方差器能产基线时——显示方差器纯预测（内容感知基线，真实声学单位）。
    public static OrderedMap<string, AutomationConfig> BuildReadbackConfigs(VoicebankConfig config)
    {
        var map = new OrderedMap<string, AutomationConfig>();
        foreach (var v in Variances)
            if (v.Use(config) && v.Predict(config))
                map.Add(v.Key, Piecewise(v.Display, v.Color, v.AcousticMin, v.AcousticMax));
        return map;
    }

    // part 级面板：多说话人暴露说话人选择、多语言暴露 part 默认语言。
    public static ObjectConfig BuildPartConfig(VoicebankConfig config)
    {
        var properties = new OrderedMap<string, IControllerConfig>();

        if (config.Speakers.Count > 1)
            properties.Add(KeySpeaker, new ComboBoxConfig
            {
                DisplayText = L.Tr("Speaker"),
                Options = SpeakerOptions(config.Speakers),
            });

        if (HasLanguageChoice(config))
            properties.Add(KeyLanguage, LanguageCombo(config, config.Languages[0]));

        return new ObjectConfig { Properties = properties };
    }

    // note 级面板：多语言声库暴露 per-note 语言覆盖；默认值取 part 当前默认语言（依赖 part 值 ⇒ 逐次构建）。
    public static ObjectConfig BuildNoteConfig(VoicebankConfig config, INotePropertyContext context)
    {
        var properties = new OrderedMap<string, IControllerConfig>();
        if (HasLanguageChoice(config))
        {
            var partDefault = context.PartProperties.GetString(KeyLanguage, config.Languages[0]);
            properties.Add(KeyLanguage, LanguageCombo(config, partDefault));
        }
        return new ObjectConfig { Properties = properties };
    }

    public static bool HasLanguageChoice(VoicebankConfig config) => config.UseLanguageId && config.Languages.Count > 1;

    static ComboBoxConfig LanguageCombo(VoicebankConfig config, string defaultValue) => new()
    {
        DisplayText = L.Tr("Language"),
        Options = ToOptions(config.Languages),
        DefaultOption = PropertyValue.Create(defaultValue),
    };

    static List<ComboBoxOption> ToOptions(IReadOnlyList<string> values)
    {
        var options = new List<ComboBoxOption>(values.Count);
        foreach (var value in values)
            options.Add(value);   // 隐式转换：string → ComboBoxOption（值即显示文本）
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

    static AutomationConfig Piecewise(string display, string color, double min, double max) => new()
    {
        DisplayText = L.Tr(display),
        DefaultValue = double.NaN,   // 分段：无基线、段间断开（NaN 自由区）
        MinValue = min,
        MaxValue = max,
        Color = color,
    };

    static AutomationConfig Continuous(string display, string color, double baseline, double min, double max) => new()
    {
        DisplayText = L.Tr(display),
        DefaultValue = baseline,
        MinValue = min,
        MaxValue = max,
        Color = color,
    };
}

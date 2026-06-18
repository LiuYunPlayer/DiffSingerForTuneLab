using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 一条 part 的合成会话。本阶段实现「声明面」：四个声明方法是选中声库能力集（VoicebankConfig）的纯函数——
// 据 use_*_embed 暴露可编辑曲线、据 predict_* 暴露只读回显轨、据 speakers/languages 暴露 part/note 属性。
// 调度与 6 级合成管线、产物发布为后续阶段：GetNextSegment 暂报「无待合成」，故宿主不驱动 SynthesizeNext，
// 会话呈现属性面板与轨但不产音——诚实的中间态。
public sealed class DiffSingerSynthesisSession : ISynthesisSession
{
    // —— 暴露给用户的参数键（避开宿主保留名 Volume / VibratoEnvelope）——
    const string KeyGender = "gender";
    const string KeySpeed = "speed";
    const string KeySpeaker = "speaker";
    const string KeyLanguage = "language";

    // Gender 连续轨基线 0 = 不移位；量程取声库增广范围（VoicebankConfig.KeyShift*）。
    const double GenderBaseline = 0;
    // —— 占位区间：曲线形态（连续 / 分段）已按声学语义定死，Min/Max/基线为占位值，
    //    待声学阶段按模型实际值域校准（此前无合成、无用户数据，调整无损）——
    const double SpeedMin = 0.5, SpeedMax = 2.0, SpeedBaseline = 1.0;     // 连续，基线 1.0 = 不变速（语义待声学阶段确认）
    const double VarianceMin = -1.0, VarianceMax = 1.0;                   // 分段（NaN 自由），值域占位

    // 四个 variance 量的声明规格：Use(config) ⇒ 暴露可编辑分段轨；Use && Predict ⇒ 附只读回显轨。
    readonly record struct VarianceSpec(
        string Key, string Display, string Color,
        Func<VoicebankConfig, bool> Use, Func<VoicebankConfig, bool> Predict);

    static readonly VarianceSpec[] Variances =
    {
        new("energy",      "Energy",      "#E573A5", c => c.UseEnergyEmbed,      c => c.PredictEnergy),
        new("breathiness", "Breathiness", "#73E5C2", c => c.UseBreathinessEmbed, c => c.PredictBreathiness),
        new("voicing",     "Voicing",     "#C2E573", c => c.UseVoicingEmbed,     c => c.PredictVoicing),
        new("tension",     "Tension",     "#A573E5", c => c.UseTensionEmbed,     c => c.PredictTension),
    };

    readonly VoicebankConfig mConfig;
    readonly ISynthesisContext mContext;

    // 轨集合与 part 面板只依赖声库能力集（每会话固定），构造期算一次即可；note 面板默认值随 part 当前值变，逐次构建。
    readonly OrderedMap<string, AutomationConfig> mAutomationConfigs = new();
    readonly OrderedMap<string, AutomationConfig> mReadbackConfigs = new();
    readonly ObjectConfig mPartConfig;

    public DiffSingerSynthesisSession(VoicebankConfig config, ISynthesisContext context)
    {
        mConfig = config;
        mContext = context;

        // 可编辑曲线：variance 量（分段，NaN 自由区由方差器/中性默认填充）+ Gender/Speed（连续，基线为中性值）。
        foreach (var v in Variances)
            if (v.Use(config))
                mAutomationConfigs.Add(v.Key, Piecewise(v.Display, v.Color));

        if (config.UseKeyShiftEmbed)
            mAutomationConfigs.Add(KeyGender, Continuous("Gender", "#E5A573", GenderBaseline, config.KeyShiftMin, config.KeyShiftMax));
        if (config.UseSpeedEmbed)
            mAutomationConfigs.Add(KeySpeed, Continuous("Speed", "#73B5E5", SpeedBaseline, SpeedMin, SpeedMax));

        // 只读回显轨：仅当声学接受该量为输入且方差器能产基线时——自由区显示方差器的内容感知输出（这正是新信息）。
        foreach (var v in Variances)
            if (v.Use(config) && v.Predict(config))
                mReadbackConfigs.Add(v.Key, Piecewise(v.Display, v.Color));

        mPartConfig = BuildPartConfig();
    }

    // 新建 note 的默认歌词：中性占位，待词典 G2P 阶段按声库词典择一有效词细化。
    public string DefaultLyric => "a";

    public IReadOnlyOrderedMap<string, AutomationConfig> GetAutomationConfigs(IPartPropertyContext context) => mAutomationConfigs;
    public IReadOnlyOrderedMap<string, AutomationConfig> GetSynthesizedParameterConfigs(IPartPropertyContext context) => mReadbackConfigs;
    public ObjectConfig GetPartPropertyConfig(IPartPropertyContext context) => mPartConfig;

    // note 级：多语言声库暴露 per-note 语言覆盖；默认值取 part 当前默认语言（依赖 part 值 ⇒ 逐次构建）。
    public ObjectConfig GetNotePropertyConfig(INotePropertyContext context)
    {
        var properties = new OrderedMap<string, IControllerConfig>();
        if (HasLanguageChoice)
        {
            var partDefault = context.PartProperties.GetString(KeyLanguage, mConfig.Languages[0]);
            properties.Add(KeyLanguage, LanguageCombo(partDefault));
        }
        return new ObjectConfig { Properties = properties };
    }

    // —— 调度 / 产物：管线未接入，呈「无待合成、无产物」 ——
    public SynthesisSegment? GetNextSegment(double startTime, double endTime) => null;
    public Task SynthesizeNext(SynthesisSegment segment, CancellationToken cancellation = default) => Task.CompletedTask;

    public IReadOnlyList<IReadOnlyList<Point>> SynthesizedPitch => [];
    public IReadOnlyMap<string, SynthesizedParameter> SynthesizedParameters => mEmptyParameters;
    public IReadOnlyList<SynthesizedPhoneme> Phonemes => [];
    public IReadOnlyList<SynthesisStatusSegment> GetStatus() => [];

    public event Action? StatusChanged;

    public void Dispose() { }   // 本阶段未订阅活视图；接入调度时在此退订。

    // —— 构建辅助 ——
    bool HasLanguageChoice => mConfig.UseLanguageId && mConfig.Languages.Count > 1;

    ObjectConfig BuildPartConfig()
    {
        var properties = new OrderedMap<string, IControllerConfig>();

        if (mConfig.Speakers.Count > 1)
            properties.Add(KeySpeaker, new ComboBoxConfig
            {
                DisplayText = L.Tr("Speaker"),
                Options = SpeakerOptions(mConfig.Speakers),
            });

        if (HasLanguageChoice)
            properties.Add(KeyLanguage, LanguageCombo(mConfig.Languages[0]));

        return new ObjectConfig { Properties = properties };
    }

    ComboBoxConfig LanguageCombo(string defaultValue) => new()
    {
        DisplayText = L.Tr("Language"),
        Options = ToOptions(mConfig.Languages),
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
    // 注：character.yaml 的 subbanks 带更友好的本地化名（如「01: 初音未来」），按 suffix 关联可作后续增强。
    static List<ComboBoxOption> SpeakerOptions(IReadOnlyList<string> speakers)
    {
        var options = new List<ComboBoxOption>(speakers.Count);
        foreach (var speaker in speakers)
        {
            int dot = speaker.LastIndexOf('.');
            var display = dot >= 0 && dot < speaker.Length - 1 ? speaker[(dot + 1)..] : speaker;
            options.Add(new ComboBoxOption(PropertyValue.Create(speaker), display));
        }
        return options;
    }

    static AutomationConfig Piecewise(string display, string color) => new()
    {
        DisplayText = L.Tr(display),
        DefaultValue = double.NaN,   // 分段：无基线、段间断开（NaN 自由区）
        MinValue = VarianceMin,
        MaxValue = VarianceMax,
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

    static readonly Map<string, SynthesizedParameter> mEmptyParameters = new();
}

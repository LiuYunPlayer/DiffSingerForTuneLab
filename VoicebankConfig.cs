using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TuneLab.Foundation;
using TuneLab.SDK;
using YamlDotNet.Serialization;

namespace DiffSingerForTuneLab;

// 从一个声库的 DiffSinger 配置解析出「声明面所需的能力集」——属性面板与自动化轨完全是它的纯函数。
// 两对正交标志决定每个声学量如何暴露：
//   · use_*_embed（声学接受为帧级输入）⇒ 暴露该量为可编辑曲线才有意义（声学不读则编辑无效）；
//   · predict_*（方差器能自动产出该量）⇒ 自由区有内容感知的基线可回显。
// 声学主配置在声库根 dsconfig.yaml；方差预测器配置在 dsvariance/dsconfig.yaml（predict_* 在此，
// 缺该目录即视作无方差器、各量无基线）。字段缺失或解析失败一律降级到安全默认，不抛。
public sealed class VoicebankConfig
{
    // Gender(key_shift) 量程缺省回退（半音）：声库未声明增广范围时的中性兜底。
    const double DefaultKeyShiftRange = 5.0;

    public int SampleRate { get; private init; } = 44100;

    // —— 推理所需的资源定位与声学超参（从声学 dsconfig 读出；文件名相对 RootPath）——
    public string RootPath { get; private init; } = string.Empty;
    public string AcousticFileName { get; private init; } = string.Empty;   // dsconfig `acoustic`
    public string VocoderName { get; private init; } = string.Empty;        // dsconfig `vocoder`（→ Vocoders/<名>）
    public string PhonemesFileName { get; private init; } = string.Empty;   // dsconfig `phonemes`（音素→id 表 JSON）
    public string LanguagesFileName { get; private init; } = string.Empty;  // dsconfig `languages`（语言→id 表 JSON）
    public int HiddenSize { get; private init; } = 256;                     // spk_embed 维度
    public int HopSize { get; private init; } = 512;
    public int NumMelBins { get; private init; } = 128;
    public double MaxDepth { get; private init; } = 1.0;                    // 浅扩散最大深度（use_variable_depth）

    // 多说话人：>1 时声明 part 级说话人选择（值为 dsconfig 原始条目，如 "260509a.Miku"）。
    public IReadOnlyList<string> Speakers { get; private init; } = [];
    // 多语言：use_lang_id 且 languages>1 时声明 part 默认语言 + per-note 语言。
    public bool UseLanguageId { get; private init; }
    public IReadOnlyList<string> Languages { get; private init; } = [];

    // 帧级声学条件输入（use_*_embed）：true ⇒ 可暴露为可编辑曲线。
    public bool UseKeyShiftEmbed { get; private init; }   // Gender
    public bool UseSpeedEmbed { get; private init; }      // Speed
    public bool UseEnergyEmbed { get; private init; }
    public bool UseBreathinessEmbed { get; private init; }
    public bool UseVoicingEmbed { get; private init; }
    public bool UseTensionEmbed { get; private init; }

    // Gender 轨量程（半音）：取声学增广 augmentation_args.random_pitch_shifting.range，缺省回退 ±DefaultKeyShiftRange。
    public double KeyShiftMin { get; private init; } = -DefaultKeyShiftRange;
    public double KeyShiftMax { get; private init; } = DefaultKeyShiftRange;

    // 方差器能否自动产出该量（predict_*）：true ⇒ 自由区有基线，回显有意义。
    // 能量/气声默认 true、发声/张力默认 false——仅当方差配置存在时这些默认才适用。
    public bool PredictEnergy { get; private init; }
    public bool PredictBreathiness { get; private init; }
    public bool PredictVoicing { get; private init; }
    public bool PredictTension { get; private init; }

    public static VoicebankConfig Load(string rootPath, ILogger logger)
    {
        var acoustic = ReadYaml(Path.Combine(rootPath, "dsconfig.yaml"), logger);

        // 方差预测器配置（原生 DiffSinger 约定子目录）；缺失则各量无基线（predict_* 全 false）。
        var variancePath = Path.Combine(rootPath, "dsvariance", "dsconfig.yaml");
        var variance = File.Exists(variancePath) ? ReadYaml(variancePath, logger) : null;

        var (keyShiftMin, keyShiftMax) = ReadRange(
            Dig(acoustic, "augmentation_args", "random_pitch_shifting", "range"))
            ?? (-DefaultKeyShiftRange, DefaultKeyShiftRange);

        return new VoicebankConfig
        {
            SampleRate = GetInt(acoustic, "sample_rate", 44100),
            RootPath = rootPath,
            AcousticFileName = GetString(acoustic, "acoustic"),
            VocoderName = GetString(acoustic, "vocoder"),
            PhonemesFileName = GetString(acoustic, "phonemes"),
            LanguagesFileName = GetString(acoustic, "languages"),
            HiddenSize = GetInt(acoustic, "hidden_size", 256),
            HopSize = GetInt(acoustic, "hop_size", 512),
            NumMelBins = GetInt(acoustic, "num_mel_bins", 128),
            MaxDepth = GetFloat(acoustic, "max_depth", 1.0),
            Speakers = GetStringList(acoustic, "speakers"),
            UseLanguageId = GetBool(acoustic, "use_lang_id", false),
            Languages = ResolveLanguages(acoustic, rootPath, logger),

            UseKeyShiftEmbed = GetBool(acoustic, "use_key_shift_embed", false),
            UseSpeedEmbed = GetBool(acoustic, "use_speed_embed", false),
            UseEnergyEmbed = GetBool(acoustic, "use_energy_embed", false),
            UseBreathinessEmbed = GetBool(acoustic, "use_breathiness_embed", false),
            UseVoicingEmbed = GetBool(acoustic, "use_voicing_embed", false),
            UseTensionEmbed = GetBool(acoustic, "use_tension_embed", false),
            KeyShiftMin = keyShiftMin,
            KeyShiftMax = keyShiftMax,

            PredictEnergy = variance is not null && GetBool(variance, "predict_energy", true),
            PredictBreathiness = variance is not null && GetBool(variance, "predict_breathiness", true),
            PredictVoicing = variance is not null && GetBool(variance, "predict_voicing", false),
            PredictTension = variance is not null && GetBool(variance, "predict_tension", false),
        };
    }

    // languages 在 dsconfig 里通常是引用的 JSON 文件名（{"en":1,...}，键即语言）；也兼容内联序列/映射或单个语言名。
    static IReadOnlyList<string> ResolveLanguages(IReadOnlyDictionary<string, object?> map, string rootPath, ILogger logger)
    {
        if (!map.TryGetValue("languages", out var v) || v is null)
            return [];

        if (v is string s)
        {
            var path = Path.Combine(rootPath, s);
            if (!File.Exists(path))
                return [s];   // 非文件引用：视作单个语言名

            try
            {
                // JSON 是 YAML 的子集，复用同一反序列化器即可；键即语言名，保留文档顺序。
                var langs = new DeserializerBuilder().Build()
                    .Deserialize<Dictionary<string, object?>>(File.ReadAllText(path));
                return langs is null ? [] : langs.Keys.ToList();
            }
            catch (System.Exception ex)
            {
                logger.Warning($"DiffSinger：解析语言表失败 {path}: {ex.Message}");
                return [];
            }
        }

        return GetStringList(map, "languages");
    }

    // —— YAML 读取（标量反序列化为 string，序列为 List，映射为 Dictionary；逐键取值带默认）——
    static IReadOnlyDictionary<string, object?> ReadYaml(string path, ILogger logger)
    {
        try
        {
            var map = new DeserializerBuilder().Build()
                .Deserialize<Dictionary<string, object?>>(File.ReadAllText(path));
            return map ?? [];
        }
        catch (System.Exception ex)
        {
            logger.Warning($"DiffSinger：解析配置失败 {path}: {ex.Message}");
            return new Dictionary<string, object?>();
        }
    }

    static bool GetBool(IReadOnlyDictionary<string, object?> map, string key, bool def)
        => map.TryGetValue(key, out var v) && v is string s && bool.TryParse(s, out var b) ? b : def;

    static int GetInt(IReadOnlyDictionary<string, object?> map, string key, int def)
        => map.TryGetValue(key, out var v) && v is string s && int.TryParse(s, out var i) ? i : def;

    static double GetFloat(IReadOnlyDictionary<string, object?> map, string key, double def)
        => map.TryGetValue(key, out var v) && v is string s
            && double.TryParse(s, System.Globalization.CultureInfo.InvariantCulture, out var d) ? d : def;

    static string GetString(IReadOnlyDictionary<string, object?> map, string key)
        => map.TryGetValue(key, out var v) && v is string s ? s : string.Empty;

    static IReadOnlyList<string> GetStringList(IReadOnlyDictionary<string, object?> map, string key)
        => map.TryGetValue(key, out var v) ? ToStringList(v) : [];

    static IReadOnlyList<string> ToStringList(object? v)
    {
        IEnumerable<object?>? items = v switch
        {
            IDictionary dict => dict.Keys.Cast<object?>(),
            IEnumerable seq and not string => seq.Cast<object?>(),
            _ => null,
        };
        if (items is null)
            return [];

        return items.Select(x => x?.ToString())
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Select(s => s!)
            .ToList();
    }

    // 沿键路径下钻嵌套映射（顶层与 YamlDotNet 嵌套映射均实现非泛型 IDictionary）。
    static object? Dig(object? node, params string[] path)
    {
        foreach (var key in path)
        {
            if (node is not IDictionary d || !d.Contains(key))
                return null;
            node = d[key];
        }
        return node;
    }

    // 取序列前两个可解析为数的元素作 (min, max)；不足则返回 null（由调用方回退）。
    static (double, double)? ReadRange(object? node)
    {
        if (node is not IEnumerable seq || node is string)
            return null;

        var nums = seq.Cast<object?>()
            .Select(x => double.TryParse(x?.ToString(), out var d) ? (double?)d : null)
            .Where(d => d.HasValue)
            .Select(d => d!.Value)
            .ToList();
        return nums.Count >= 2 ? (nums[0], nums[1]) : null;
    }
}

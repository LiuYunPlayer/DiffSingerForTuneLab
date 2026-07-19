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
// 缺该目录即视作无方差器、各量无基线）；pitch 预测器配置在 dspitch/dsconfig.yaml（use_expr/use_note_rest 在此）；
// 声码器能力（pitch_controllable）在解析出的 vocoder.yaml。字段缺失或解析失败一律降级到安全默认，不抛。
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
    public int WinSize { get; private init; } = 2048;
    public int FftSize { get; private init; } = 2048;
    public int NumMelBins { get; private init; } = 128;
    public double MelFmin { get; private init; } = 40;
    public double MelFmax { get; private init; } = 16000;
    public string MelBase { get; private init; } = "e";                     // "10" 或 "e"
    public string MelScale { get; private init; } = "slaney";               // "slaney" 或 "htk"

    // 采样加速形态：连续加速 ⇒ 传 depth+steps；否则 maxDepth 要 /1000、传 speedup（见 OpenUtau DiffSingerRenderer）。
    public bool UseContinuousAcceleration { get; private init; }
    public bool UseVariableDepth { get; private init; }   // use_variable_depth || use_shallow_diffusion
    // 浅扩散最大深度：连续加速取原值，否则 /1000（OpenUtau DsConfig.maxDepth）。
    public double RawMaxDepth { get; private init; } = 1.0;
    public double MaxDepth => UseContinuousAcceleration ? RawMaxDepth : RawMaxDepth / 1000.0;

    // pitch 预测器相关（来源 dspitch/dsconfig.yaml，对齐 OpenUtau DsPitch 读自己目录的配置；缺该目录即 false）。
    //   UseExpr ⇒ pitch 模型有 expr 口（表现力 0~1 帧级混合），声明期据此暴露 expressiveness 轨。
    public bool UseExpr { get; private init; }
    public bool UseNoteRest { get; private init; }

    // 声码器 pitch_controllable（解析出的 vocoder.yaml；查无声码器 → false）：
    //   true ⇒ 声学吃移调 f0 而声码器吃原始 f0 可行 ⇒ 声明期暴露 tone_shift 轨（对齐 OpenUtau SHFC 语义）。
    public bool VocoderPitchControllable { get; private init; }

    // 多说话人：>1 时声明 part 级说话人选择（值为 dsconfig 原始条目，如 "260509a.Miku"）。
    public IReadOnlyList<string> Speakers { get; private init; } = [];
    // 多语言：languages>1 即声明 part 默认语言 + per-note 语言（前缀音素库 use_lang_id=false 也算多语言）。
    //   use_lang_id 只标记模型有无 lang embed；喂不喂 languages 张量由 ONNX HasInput 决定，此标志现仅作参考。
    public bool UseLanguageId { get; private init; }
    public IReadOnlyList<string> Languages { get; private init; } = [];

    // 帧级声学条件输入（use_*_embed）：true ⇒ 可暴露为可编辑曲线。
    public bool UseKeyShiftEmbed { get; private init; }   // Gender
    public bool UseSpeedEmbed { get; private init; }      // Speed
    public bool UseEnergyEmbed { get; private init; }
    public bool UseBreathinessEmbed { get; private init; }
    public bool UseVoicingEmbed { get; private init; }
    public bool UseTensionEmbed { get; private init; }
    // SHMC（fork PR#2）：声学接受帧级口型偏移 alpha ∈ [-1,1]（相对隐式基线，0 = 不干预）。
    // 冻结导出（--freeze_shift_mouth_opening）时导出器写 false ⇒ 不暴露轨。
    public bool UseShiftMouthOpeningEmbed { get; private init; }

    // Gender 轨量程（半音）：取声学增广 augmentation_args.random_pitch_shifting.range，缺省回退 ±DefaultKeyShiftRange。
    public double KeyShiftMin { get; private init; } = -DefaultKeyShiftRange;
    public double KeyShiftMax { get; private init; } = DefaultKeyShiftRange;

    // 方差器能否自动产出该量（predict_*）：true ⇒ 自由区有基线，回显有意义。
    // 能量/气声默认 true、发声/张力默认 false——仅当方差配置存在时这些默认才适用。
    // mouth_opening 默认 false（与 use_mouth_opening_embed 解耦，仅方差器自身输出通道）。
    public bool PredictEnergy { get; private init; }
    public bool PredictBreathiness { get; private init; }
    public bool PredictVoicing { get; private init; }
    public bool PredictTension { get; private init; }
    public bool PredictMouthOpening { get; private init; }

    // —— voicing 线上编码域（fork feat/mulaw-voicing 实验声库）——
    // "db"（缺省 = 历史声库）或 "mulaw"（binarize 期谐波 RMS 压 mu-law 后线性映射到 [-96,0] 伪装 dB）。
    // 消费点 VoicingDomainCodec：Delta 公式/回显/clamp 恒 dB 语义、仅模型边界转换；μ 仅 mulaw 有意义。
    public string VoicingDomain { get; private init; } = "db";
    public float VoicingMu { get; private init; } = 255f;

    // vocoderRoots = 全局声码器搜索根（引擎设置）；null → 仅默认目录（注册表扫描等不关心声码器的调用点）。
    public static VoicebankConfig Load(string rootPath, ILogger logger, IReadOnlyList<string>? vocoderRoots = null)
    {
        var acoustic = ReadYaml(Path.Combine(rootPath, "dsconfig.yaml"), logger);

        // 方差预测器配置（原生 DiffSinger 约定子目录）；缺失则各量无基线（predict_* 全 false）。
        var variancePath = Path.Combine(rootPath, "dsvariance", "dsconfig.yaml");
        var variance = File.Exists(variancePath) ? ReadYaml(variancePath, logger) : null;

        // pitch 预测器配置（use_expr/use_note_rest 属 dspitch 自己的 dsconfig，非声库根）；缺失即无 pitch 侧能力。
        var pitchPath = Path.Combine(rootPath, "dspitch", "dsconfig.yaml");
        var pitch = File.Exists(pitchPath) ? ReadYaml(pitchPath, logger) : null;

        var (keyShiftMin, keyShiftMax) = ReadRange(
            Dig(acoustic, "augmentation_args", "random_pitch_shifting", "range"))
            ?? (-DefaultKeyShiftRange, DefaultKeyShiftRange);

        bool useVoicingEmbed = GetBool(acoustic, "use_voicing_embed", false);
        bool predictVoicing = variance is not null && GetBool(variance, "predict_voicing", false);
        var (voicingDomain, voicingMu) = ReadVoicingDomain(acoustic, variance, useVoicingEmbed, predictVoicing, logger);

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
            WinSize = GetInt(acoustic, "win_size", 2048),
            FftSize = GetInt(acoustic, "fft_size", 2048),
            NumMelBins = GetInt(acoustic, "num_mel_bins", 128),
            MelFmin = GetFloat(acoustic, "mel_fmin", 40),
            MelFmax = GetFloat(acoustic, "mel_fmax", 16000),
            MelBase = GetStringOr(acoustic, "mel_base", "e"),
            MelScale = GetStringOr(acoustic, "mel_scale", "slaney"),
            UseContinuousAcceleration = GetBool(acoustic, "use_continuous_acceleration", false),
            UseVariableDepth = GetBool(acoustic, "use_variable_depth", GetBool(acoustic, "use_shallow_diffusion", false)),
            RawMaxDepth = GetFloat(acoustic, "max_depth", 1.0),
            UseExpr = pitch is not null && GetBool(pitch, "use_expr", false),
            UseNoteRest = pitch is not null && GetBool(pitch, "use_note_rest", false),
            VocoderPitchControllable = ReadVocoderPitchControllable(rootPath, GetString(acoustic, "vocoder"), vocoderRoots, logger),
            Speakers = GetStringList(acoustic, "speakers"),
            UseLanguageId = GetBool(acoustic, "use_lang_id", false),
            Languages = ResolveLanguages(acoustic, rootPath, logger),

            UseKeyShiftEmbed = GetBool(acoustic, "use_key_shift_embed", false),
            UseSpeedEmbed = GetBool(acoustic, "use_speed_embed", false),
            UseEnergyEmbed = GetBool(acoustic, "use_energy_embed", false),
            UseBreathinessEmbed = GetBool(acoustic, "use_breathiness_embed", false),
            UseVoicingEmbed = useVoicingEmbed,
            UseTensionEmbed = GetBool(acoustic, "use_tension_embed", false),
            UseShiftMouthOpeningEmbed = GetBool(acoustic, "use_shift_mouth_opening_embed", false),
            KeyShiftMin = keyShiftMin,
            KeyShiftMax = keyShiftMax,

            PredictEnergy = variance is not null && GetBool(variance, "predict_energy", true),
            PredictBreathiness = variance is not null && GetBool(variance, "predict_breathiness", true),
            PredictVoicing = predictVoicing,
            PredictTension = variance is not null && GetBool(variance, "predict_tension", false),
            PredictMouthOpening = variance is not null && GetBool(variance, "predict_mouth_opening", false),
            VoicingDomain = voicingDomain,
            VoicingMu = voicingMu,
        };
    }

    // voicing 域解析：声学与 dsvariance 的 dsconfig 各自可带 voicing_domain / voicing_mu（缺省 db / 255）。
    //   域只在该模型实际消费 voicing 时有意义；两侧同时在用却异域/异 μ = 坏导出（成对训练是导出契约），
    //   log 后按 acoustic 侧处理（终端消费者）。未知域名不猜：log 后回退 db（与"降级到安全默认"总原则一致）。
    static (string Domain, float Mu) ReadVoicingDomain(
        IReadOnlyDictionary<string, object?> acoustic, IReadOnlyDictionary<string, object?>? variance,
        bool useEmbed, bool predict, ILogger logger)
    {
        (string Domain, float Mu) Of(IReadOnlyDictionary<string, object?> map)
        {
            string d = GetStringOr(map, "voicing_domain", "db");
            if (d is not ("db" or "mulaw"))
            {
                logger.Warning($"DiffSinger：未知 voicing_domain '{d}'，按 db 处理");
                d = "db";
            }
            return (d, (float)GetFloat(map, "voicing_mu", 255));
        }

        var acu = Of(acoustic);
        var vari = variance is null ? acu : Of(variance);
        if (useEmbed && predict && (acu.Domain != vari.Domain || (acu.Domain == "mulaw" && acu.Mu != vari.Mu)))
            logger.Warning($"DiffSinger：acoustic 与 variance 的 voicing 域不一致" +
                $"（{acu.Domain}/μ{acu.Mu} vs {vari.Domain}/μ{vari.Mu}），按 acoustic 侧处理——声库导出损坏，请重导");
        return useEmbed ? acu : vari;
    }

    // 声码器 pitch_controllable：按与加载期同一解析链（bundled dsvocoder 优先 → 全局目录按名，见
    //   DiffSingerModelCache.ResolveVocoderDir）定位 vocoder.yaml 后读布尔。查无声码器 / 解析失败 → false（轨不暴露，
    //   合成期加载声码器会另行报错）。roots 为 null 时用默认 Vocoders 目录兜底。
    static bool ReadVocoderPitchControllable(string rootPath, string vocoderName, IReadOnlyList<string>? roots, ILogger logger)
    {
        var dir = DiffSingerModelCache.ResolveVocoderDir(rootPath, vocoderName,
            roots ?? [DiffSingerModelCache.VocodersDirectory]);
        if (dir is null)
            return false;
        var yaml = ReadYaml(Path.Combine(dir, "vocoder.yaml"), logger);
        return GetBool(yaml, "pitch_controllable", false);
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

    static string GetStringOr(IReadOnlyDictionary<string, object?> map, string key, string def)
        => map.TryGetValue(key, out var v) && v is string s && s.Length > 0 ? s : def;

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

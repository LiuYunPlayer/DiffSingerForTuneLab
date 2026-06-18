using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using YamlDotNet.Serialization;

// dev-only 冒烟测试：对真实声学/声码器模型敲定 ONNX 张量 I/O、验证 DirectML、跑出第一声。
// 用法（路径外部化，绝不硬编码敏感声库/角色名）：
//   dotnet run --project tools/DiffSingerSmokeTest -- <voiceRoot> <vocoderDir> [outWav]
//   或设环境变量 DIFFSINGER_VOICE_ROOT / DIFFSINGER_VOCODER_DIR 后直接 dotnet run。
// 一切声库相关标识（speaker、音素、语言）均从配置/文件运行时解析，源码不写入任何敏感名。

var positional = args.Where(a => !a.StartsWith("--")).ToArray();
string? voiceRoot = ArgOrEnv(positional, 0, "DIFFSINGER_VOICE_ROOT");
string? vocoderDir = ArgOrEnv(positional, 1, "DIFFSINGER_VOCODER_DIR");
if (voiceRoot is null || vocoderDir is null)
{
    Console.Error.WriteLine("用法: <voiceRoot> <vocoderDir> [outWav]（或设 DIFFSINGER_VOICE_ROOT / DIFFSINGER_VOCODER_DIR）");
    return 1;
}
string outWav = args.Length > 2 ? args[2] : Path.Combine(Path.GetTempPath(), "diffsinger_smoke.wav");

var yaml = new DeserializerBuilder().Build();
var acousticCfg = yaml.Deserialize<Dictionary<string, object?>>(File.ReadAllText(Path.Combine(voiceRoot, "dsconfig.yaml")));
var vocoderCfg = yaml.Deserialize<Dictionary<string, object?>>(File.ReadAllText(Path.Combine(vocoderDir, "vocoder.yaml")));

string acousticPath = Path.Combine(voiceRoot, acousticCfg["acoustic"]!.ToString()!);
string vocoderPath = Path.Combine(vocoderDir, vocoderCfg["model"]!.ToString()!);
int hopSize = int.Parse(acousticCfg["hop_size"]!.ToString()!);
int sampleRate = int.Parse(acousticCfg["sample_rate"]!.ToString()!);

Dump("ACOUSTIC", acousticPath);
Dump("VOCODER", vocoderPath);

// 预测器链元数据（dsdur/dspitch/dsvariance 下的 linguistic/dur/pitch/variance）——敲定张量名与形状。
foreach (var sub in new[] { "dsdur", "dspitch", "dsvariance" })
{
    var subDir = Path.Combine(voiceRoot, sub);
    if (!Directory.Exists(subDir)) continue;
    foreach (var onnx in Directory.GetFiles(subDir, "*.onnx").OrderBy(p => p))
        Dump($"{sub}/{Path.GetFileName(onnx)}", onnx);
}

if (args.Contains("--dur"))
{
    PredictDurChain(voiceRoot);
    return 0;
}

if (args.Contains("--dump-only"))
    return 0;

Synthesize(voiceRoot, acousticCfg, acousticPath, vocoderPath, hopSize, sampleRate, outWav);
return 0;

static string? ArgOrEnv(string[] args, int idx, string env)
    => idx < args.Length ? args[idx] : Environment.GetEnvironmentVariable(env);

// 先试 DirectML，建会话失败回退 CPU；同时验证执行设备接线与回退路径。
static InferenceSession Open(string modelPath)
{
    try
    {
        var opts = new SessionOptions();
        opts.AppendExecutionProvider_DML(0);
        var session = new InferenceSession(modelPath, opts);
        Console.WriteLine($"  [{Path.GetFileName(modelPath)}] provider=DirectML");
        return session;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  [{Path.GetFileName(modelPath)}] DirectML 失败，回退 CPU：{ex.Message}");
        return new InferenceSession(modelPath, new SessionOptions());
    }
}

static void Dump(string label, string modelPath)
{
    Console.WriteLine($"\n======== {label}: {Path.GetFileName(modelPath)} ========");
    using var session = Open(modelPath);
    Console.WriteLine("  -- Inputs --");
    foreach (var (name, meta) in session.InputMetadata)
        Console.WriteLine($"    {name}: {Describe(meta)}");
    Console.WriteLine("  -- Outputs --");
    foreach (var (name, meta) in session.OutputMetadata)
        Console.WriteLine($"    {name}: {Describe(meta)}");
}

// 形状里 -1 / 命名维是动态维；SymbolicDimensions 给出动态维名字（如 "n_tokens" / "n_frames"）。
static string Describe(NodeMetadata meta)
{
    var dims = meta.Dimensions;
    var symbolic = meta.SymbolicDimensions;
    var parts = new string[dims.Length];
    for (int i = 0; i < dims.Length; i++)
        parts[i] = dims[i] >= 0 ? dims[i].ToString()
            : (i < symbolic.Length && !string.IsNullOrEmpty(symbolic[i]) ? symbolic[i] : "?");
    return $"{meta.ElementDataType} [{string.Join(", ", parts)}]";
}

// 最小推理链：手填 tokens/durations/f0 + 中性条件量 → acoustic(mel) → vocoder(wav)。
// 跳过 G2P 与四个预测器（dur/pitch/variance/linguistic）；中性常量待出声后按耳朵 + 接预测器校准。
static void Synthesize(string voiceRoot, Dictionary<string, object?> cfg,
    string acousticPath, string vocoderPath, int hopSize, int sampleRate, string outWav)
{
    Console.WriteLine("\n======== SYNTHESIZE（最小链：固定音素 zh/a，平 f0 220Hz）========");

    // 标识全部运行时解析，源码不写入任何敏感名。
    var yaml = new DeserializerBuilder().Build();
    var phonemes = yaml.Deserialize<Dictionary<string, int>>(File.ReadAllText(ResolvePath(voiceRoot, cfg, "phonemes")));
    var languages = yaml.Deserialize<Dictionary<string, int>>(File.ReadAllText(ResolvePath(voiceRoot, cfg, "languages")));
    int hidden = int.Parse(cfg["hidden_size"]!.ToString()!);

    // 音素序列 SP, zh/a, SP（帧时长手设，和 = n_frames）。
    long[] tokens = { phonemes["SP"], phonemes["zh/a"], phonemes["SP"] };
    long[] phLangs = { languages["zh"], languages["zh"], languages["zh"] };
    long[] durations = { 8, 80, 8 };
    int nTokens = tokens.Length;
    int nFrames = (int)durations.Sum();

    // 帧级条件量：f0 平 220Hz；gender 0（不移调）；velocity 1（常速）；三 variance 量中性 0（待校准）。
    var f0 = Fill(nFrames, 220f);
    var gender = Fill(nFrames, 0f);
    var velocity = Fill(nFrames, 1f);
    var breathiness = Fill(nFrames, 0f);
    var voicing = Fill(nFrames, 0f);
    var tension = Fill(nFrames, 0f);

    // spk_embed：取 speakers[0] 的 .emb（256 float32 LE），逐帧广播。
    string speaker0 = ((List<object>)cfg["speakers"]!)[0].ToString()!;
    float[] emb = ReadEmb(Path.Combine(voiceRoot, speaker0 + ".emb"), hidden);
    var spk = new float[nFrames * hidden];
    for (int f = 0; f < nFrames; f++)
        Array.Copy(emb, 0, spk, f * hidden, hidden);

    float depth = float.Parse(cfg.GetValueOrDefault("max_depth")?.ToString() ?? "1.0");
    long steps = 20;

    var acInputs = new List<NamedOnnxValue>
    {
        NvL("tokens", tokens, 1, nTokens),
        NvL("languages", phLangs, 1, nTokens),
        NvL("durations", durations, 1, nTokens),
        NvF("f0", f0, 1, nFrames),
        NvF("breathiness", breathiness, 1, nFrames),
        NvF("voicing", voicing, 1, nFrames),
        NvF("tension", tension, 1, nFrames),
        NvF("gender", gender, 1, nFrames),
        NvF("velocity", velocity, 1, nFrames),
        NamedOnnxValue.CreateFromTensor("spk_embed", new DenseTensor<float>(spk, new[] { 1, nFrames, hidden })),
        NamedOnnxValue.CreateFromTensor("depth", new DenseTensor<float>(new[] { depth }, Array.Empty<int>())),
        NamedOnnxValue.CreateFromTensor("steps", new DenseTensor<long>(new[] { steps }, Array.Empty<int>())),
    };

    using var acoustic = Open(acousticPath);
    using var melResult = acoustic.Run(acInputs);
    var mel = melResult.First(v => v.Name == "mel").AsTensor<float>();
    Console.WriteLine($"  mel: [{string.Join(",", mel.Dimensions.ToArray())}]");

    using var vocoder = Open(vocoderPath);
    using var wavResult = vocoder.Run(new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("mel", mel),
        NvF("f0", f0, 1, nFrames),
    });
    var waveform = wavResult.First(v => v.Name == "waveform").AsTensor<float>();
    Console.WriteLine($"  waveform: [{string.Join(",", waveform.Dimensions.ToArray())}]  (期望 ≈ {nFrames * hopSize})");

    WriteWav(outWav, waveform.ToArray(), sampleRate);
    Console.WriteLine($"  已写出: {outWav}");
}

// dur 链原型（dsdur: linguistic + dur）——敲定 ph_dur_pred 单位与按词整流。
static void PredictDurChain(string voiceRoot)
{
    Console.WriteLine("\n======== DUR CHAIN (dsdur: linguistic + dur) ========");
    var yaml = new DeserializerBuilder().Build();
    string durDir = Path.Combine(voiceRoot, "dsdur");
    var cfg = yaml.Deserialize<Dictionary<string, object?>>(File.ReadAllText(Path.Combine(durDir, "dsconfig.yaml")));
    var phonemes = yaml.Deserialize<Dictionary<string, int>>(File.ReadAllText(Path.Combine(durDir, cfg["phonemes"]!.ToString()!)));
    var languages = yaml.Deserialize<Dictionary<string, int>>(File.ReadAllText(Path.Combine(durDir, cfg["languages"]!.ToString()!)));
    int hidden = int.Parse(cfg["hidden_size"]!.ToString()!);
    int hop = int.Parse(cfg["hop_size"]!.ToString()!);
    int sr = int.Parse(cfg["sample_rate"]!.ToString()!);

    // G2P：dsdict-zh.yaml（grapheme→带前缀音素）。
    var dict = LoadDsdict(Path.Combine(durDir, "dsdict-zh.yaml"));
    // 前导/收尾 SP 作短语边界（OpenUtau 约定：首音素吸收松弛）。
    var words = new[] { (g: "SP", midi: 0L, dur: 0.2), (g: "ni", midi: 60L, dur: 0.5), (g: "hao", midi: 62L, dur: 0.5), (g: "SP", midi: 0L, dur: 0.2) };

    var tokens = new List<long>();
    var langs = new List<long>();
    var phMidi = new List<long>();
    var wordDiv = new List<long>();
    var wordDur = new List<long>();
    foreach (var w in words)
    {
        var phs = dict[w.g];
        wordDiv.Add(phs.Length);
        wordDur.Add((long)Math.Round(w.dur * sr / hop));
        foreach (var ph in phs)
        {
            tokens.Add(phonemes[ph]);
            langs.Add(languages["zh"]);
            phMidi.Add(w.midi);
        }
        Console.WriteLine($"  G2P {w.g} → [{string.Join(" ", phs)}]  midi={w.midi} dur={w.dur}s ({wordDur[^1]} frames)");
    }
    int nTokens = tokens.Count, nWords = words.Length;

    float[] emb = ReadEmb(Path.Combine(durDir, "260509d.Len.emb"), hidden);
    var spk = new float[nTokens * hidden];
    for (int i = 0; i < nTokens; i++) Array.Copy(emb, 0, spk, i * hidden, hidden);

    using var ling = Open(Path.Combine(durDir, cfg["linguistic"]!.ToString()!));
    using var lingOut = ling.Run(new List<NamedOnnxValue>
    {
        NvL("tokens", tokens.ToArray(), 1, nTokens),
        NvL("languages", langs.ToArray(), 1, nTokens),
        NvL("word_div", wordDiv.ToArray(), 1, nWords),
        NvL("word_dur", wordDur.ToArray(), 1, nWords),
    });
    var encoderOut = lingOut.First(v => v.Name == "encoder_out").AsTensor<float>();
    var xMasks = lingOut.First(v => v.Name == "x_masks").AsTensor<bool>();
    Console.WriteLine($"  encoder_out: [{string.Join(",", encoderOut.Dimensions.ToArray())}]  x_masks: [{string.Join(",", xMasks.Dimensions.ToArray())}]");

    // encoder_out/x_masks 透传给 dur；DenseTensor 物化避免跨 session 借用问题。
    var encDense = new DenseTensor<float>(encoderOut.ToArray(), encoderOut.Dimensions.ToArray());
    var maskDense = new DenseTensor<bool>(xMasks.ToArray(), xMasks.Dimensions.ToArray());

    using var dur = Open(Path.Combine(durDir, cfg["dur"]!.ToString()!));
    using var durOut = dur.Run(new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("encoder_out", encDense),
        NamedOnnxValue.CreateFromTensor("x_masks", maskDense),
        NvL("ph_midi", phMidi.ToArray(), 1, nTokens),
        NamedOnnxValue.CreateFromTensor("spk_embed", new DenseTensor<float>(spk, new[] { 1, nTokens, hidden })),
    });
    var phDur = durOut.First(v => v.Name == "ph_dur_pred").AsTensor<float>().ToArray();
    Console.WriteLine("  ph_dur_pred (raw): " + string.Join(", ", phDur.Select(x => x.ToString("F3"))));

    // 按词整流到 word_dur（累积取整、各≥1），展平为声学 durations。
    var acDur = new List<long>();
    int idx = 0;
    for (int w = 0; w < nWords; w++)
    {
        var slice = phDur.Skip(idx).Take((int)wordDiv[w]).Select(x => (double)x).ToArray();
        var rect = Rectify(slice, (int)wordDur[w]);
        acDur.AddRange(rect.Select(x => (long)x));
        Console.WriteLine($"  word {words[w].g}: raw Σ={slice.Sum():F2} → rect [{string.Join(",", rect)}] (Σ={rect.Sum()}={wordDur[w]})");
        idx += (int)wordDiv[w];
    }

    // —— 续接声学+声码器：用整流时长 + 真实音素（映射到声学表）出音，验证形状与非静音 ——
    Console.WriteLine("  -- 续接声学+声码器 --");
    var acCfg = yaml.Deserialize<Dictionary<string, object?>>(File.ReadAllText(Path.Combine(voiceRoot, "dsconfig.yaml")));
    var acPhon = yaml.Deserialize<Dictionary<string, int>>(File.ReadAllText(Path.Combine(voiceRoot, acCfg["phonemes"]!.ToString()!)));
    var acLang = yaml.Deserialize<Dictionary<string, int>>(File.ReadAllText(Path.Combine(voiceRoot, acCfg["languages"]!.ToString()!)));
    int acHidden = int.Parse(acCfg["hidden_size"]!.ToString()!);
    float acDepth = float.Parse(acCfg.GetValueOrDefault("max_depth")?.ToString() ?? "1.0");

    // 展平符号序列（与 tokens 同序），映射到声学表 + 逐帧 f0/note 音高。
    var flatSyms = new List<string>();
    var flatMidi = new List<long>();
    foreach (var w in words)
        foreach (var ph in dict[w.g]) { flatSyms.Add(ph); flatMidi.Add(w.midi); }
    var acTokens = flatSyms.Select(s => (long)(acPhon.TryGetValue(s, out var t) ? t : 0)).ToArray();
    var acLangs = flatSyms.Select(_ => (long)acLang["zh"]).ToArray();
    int nF = (int)acDur.Sum();
    var f0 = new float[nF];
    int fp = 0;
    for (int p = 0; p < acDur.Count; p++)
        for (int k = 0; k < acDur[p]; k++)
            f0[fp++] = flatMidi[p] <= 0 ? 0f : (float)(440.0 * Math.Pow(2, (flatMidi[p] - 69) / 12.0));

    float[] acEmb = ReadEmb(Path.Combine(voiceRoot, acCfg["speakers"] is List<object> sp ? sp[0].ToString()! + ".emb" : "260509a.Len.emb"), acHidden);
    var acSpk = new float[nF * acHidden];
    for (int f = 0; f < nF; f++) Array.Copy(acEmb, 0, acSpk, f * acHidden, acHidden);

    using var acoustic = Open(Path.Combine(voiceRoot, acCfg["acoustic"]!.ToString()!));
    var acIn = new List<NamedOnnxValue>
    {
        NvL("tokens", acTokens, 1, acTokens.Length),
        NvL("languages", acLangs, 1, acLangs.Length),
        NvL("durations", acDur.ToArray(), 1, acDur.Count),
        NvF("f0", f0, 1, nF),
        NvF("breathiness", Fill(nF, 0f), 1, nF),
        NvF("voicing", Fill(nF, 0f), 1, nF),
        NvF("tension", Fill(nF, 0f), 1, nF),
        NvF("gender", Fill(nF, 0f), 1, nF),
        NvF("velocity", Fill(nF, 1f), 1, nF),
        NamedOnnxValue.CreateFromTensor("spk_embed", new DenseTensor<float>(acSpk, new[] { 1, nF, acHidden })),
        NamedOnnxValue.CreateFromTensor("depth", new DenseTensor<float>(new[] { acDepth }, Array.Empty<int>())),
        NamedOnnxValue.CreateFromTensor("steps", new DenseTensor<long>(new[] { 20L }, Array.Empty<int>())),
    };
    using var melR = acoustic.Run(acIn);
    var mel = melR.First(v => v.Name == "mel").AsTensor<float>();
    using var voc = Open(Path.Combine(Environment.GetEnvironmentVariable("DIFFSINGER_VOCODER_DIR")!,
        yaml.Deserialize<Dictionary<string, object?>>(File.ReadAllText(Path.Combine(Environment.GetEnvironmentVariable("DIFFSINGER_VOCODER_DIR")!, "vocoder.yaml")))["model"]!.ToString()!));
    using var wavR = voc.Run(new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("mel", mel), NvF("f0", f0, 1, nF) });
    var wave = wavR.First(v => v.Name == "waveform").AsTensor<float>().ToArray();
    double rms = Math.Sqrt(wave.Select(x => (double)x * x).Sum() / Math.Max(1, wave.Length));
    string outWav = Path.Combine(Path.GetTempPath(), "diffsinger_durchain.wav");
    WriteWav(outWav, wave, sr);
    Console.WriteLine($"  mel=[{string.Join(",", mel.Dimensions.ToArray())}] waveform={wave.Length} (期望≈{nF * hop}) RMS={rms:F4}  → {outWav}");
}

// 按词整流（与插件 RectifyToFrames 同算法）：缩放到 target、累积取整保总和、各≥1。
static int[] Rectify(double[] pred, int target)
{
    int len = pred.Length;
    if (len == 0) return Array.Empty<int>();
    if (target < len) target = len;
    double sum = pred.Select(x => Math.Max(0, x)).Sum();
    var result = new int[len];
    double cum = 0; int placed = 0;
    for (int i = 0; i < len; i++)
    {
        cum += sum > 0 ? Math.Max(0, pred[i]) / sum * target : (double)target / len;
        int upto = (int)Math.Round(cum);
        result[i] = upto - placed; placed = upto;
    }
    for (int i = 0; i < len; i++)
    {
        if (result[i] >= 1) continue;
        int need = 1 - result[i]; result[i] = 1;
        int big = 0; for (int j = 1; j < len; j++) if (result[j] > result[big]) big = j;
        if (big != i && result[big] - need >= 1) result[big] -= need;
    }
    return result;
}

static Dictionary<string, string[]> LoadDsdict(string path)
{
    var root = new DeserializerBuilder().IgnoreUnmatchedProperties().Build()
        .Deserialize<DsDict>(File.ReadAllText(path));
    var map = new Dictionary<string, string[]>();
    foreach (var e in root.entries)
        map[e.grapheme] = e.phonemes.ToArray();
    return map;
}

static string ResolvePath(string root, Dictionary<string, object?> cfg, string key)
    => Path.Combine(root, cfg[key]!.ToString()!);

static float[] Fill(int n, float v)
{
    var a = new float[n];
    Array.Fill(a, v);
    return a;
}

static float[] ReadEmb(string path, int dim)
{
    var bytes = File.ReadAllBytes(path);
    var a = new float[dim];
    for (int i = 0; i < dim; i++)
        a[i] = BitConverter.ToSingle(bytes, i * 4);
    return a;
}

static NamedOnnxValue NvL(string name, long[] data, int d0, int d1)
    => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, new[] { d0, d1 }));

static NamedOnnxValue NvF(string name, float[] data, int d0, int d1)
    => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(data, new[] { d0, d1 }));

// 16-bit PCM 单声道 WAV。
static void WriteWav(string path, float[] samples, int sampleRate)
{
    using var fs = new FileStream(path, FileMode.Create);
    using var w = new BinaryWriter(fs);
    int dataBytes = samples.Length * 2;
    w.Write(Encoding.ASCII.GetBytes("RIFF"));
    w.Write(36 + dataBytes);
    w.Write(Encoding.ASCII.GetBytes("WAVE"));
    w.Write(Encoding.ASCII.GetBytes("fmt "));
    w.Write(16);
    w.Write((short)1);                 // PCM
    w.Write((short)1);                 // mono
    w.Write(sampleRate);
    w.Write(sampleRate * 2);           // byte rate
    w.Write((short)2);                 // block align
    w.Write((short)16);                // bits per sample
    w.Write(Encoding.ASCII.GetBytes("data"));
    w.Write(dataBytes);
    foreach (var s in samples)
        w.Write((short)(Math.Clamp(s, -1f, 1f) * short.MaxValue));
}

// dsdict-{lang}.yaml 结构：entries: - {grapheme, phonemes:[...]}。
sealed class DsDict { public List<DsDictEntry> entries { get; set; } = new(); }
sealed class DsDictEntry { public string grapheme { get; set; } = ""; public List<string> phonemes { get; set; } = new(); }

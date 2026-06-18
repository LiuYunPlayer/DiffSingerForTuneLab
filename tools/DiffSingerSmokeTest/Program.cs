using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using YamlDotNet.Serialization;

// dev-only 冒烟测试：对真实声学/声码器模型敲定 ONNX 张量 I/O、验证 DirectML、跑出第一声。
// 用法（路径外部化，绝不硬编码敏感声库/角色名）：
//   dotnet run --project tools/DiffSingerSmokeTest -- <voiceRoot> <vocoderDir> [outWav]
//   或设环境变量 DIFFSINGER_VOICE_ROOT / DIFFSINGER_VOCODER_DIR 后直接 dotnet run。
// 一切声库相关标识（speaker、音素、语言）均从配置/文件运行时解析，源码不写入任何敏感名。

string? voiceRoot = ArgOrEnv(args, 0, "DIFFSINGER_VOICE_ROOT");
string? vocoderDir = ArgOrEnv(args, 1, "DIFFSINGER_VOCODER_DIR");
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

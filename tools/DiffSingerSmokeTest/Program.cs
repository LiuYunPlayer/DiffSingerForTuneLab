using System.Diagnostics;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using YamlDotNet.Serialization;
using DiffSingerForTuneLab;   // 链接进来的运行时核心（IModelSession / InProcessModelSession / RemoteModelSession / …）

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

if (args.Contains("--dur-bench"))
{
    PredictDurBench(voiceRoot);
    return 0;
}

if (args.Contains("--dur"))
{
    PredictDurChain(voiceRoot);
    return 0;
}

if (args.Contains("--dump-only"))
    return 0;

if (args.Contains("--retake"))
{
    RetakeTest(voiceRoot, acousticCfg, acousticPath);
    return 0;
}

if (args.Contains("--bench-ipc"))
{
    BenchIpc(voiceRoot, acousticCfg, acousticPath);
    return 0;
}

Synthesize(voiceRoot, acousticCfg, acousticPath, vocoderPath, hopSize, sampleRate, outWav);
return 0;

// note 级 retake 的 DML 忠实复现：用插件同款 NuGet OnnxRuntime.DirectML 跑声学，喂 retake/gt_mel
// （+ depth/steps 按插件 [1] 形状），定位插件里 /fs2/Sub_1 之类的 DML 运行时错。
static void RetakeTest(string voiceRoot, Dictionary<string, object?> cfg, string acousticPath)
{
    Console.WriteLine("\n======== RETAKE DML TEST ========");
    var yaml = new DeserializerBuilder().Build();
    var phonemes = yaml.Deserialize<Dictionary<string, int>>(File.ReadAllText(ResolvePath(voiceRoot, cfg, "phonemes")));
    var languages = yaml.Deserialize<Dictionary<string, int>>(File.ReadAllText(ResolvePath(voiceRoot, cfg, "languages")));
    using var ac = Open(acousticPath);
    Console.WriteLine("  -- Acoustic inputs --");
    foreach (var (name, meta) in ac.InputMetadata)
        Console.WriteLine($"    {name}: {Describe(meta)}");

    int melBins = ac.InputMetadata["gt_mel"].Dimensions[2];
    var nd = ac.InputMetadata["noise"].Dimensions;   // [1, feats, outDims, -1]
    int feats = nd[1], outDims = nd[2];
    int F = 120;
    long[] tokens = { phonemes["SP"], phonemes.GetValueOrDefault("zh/a", 1), phonemes["SP"] };
    long[] langs = { languages["zh"], languages["zh"], languages["zh"] };
    long[] durations = { 8, F - 16, 8 };

    foreach (var (label, mask) in new (string, bool[])[]
    {
        ("retake 全 true（非混合）", Enumerable.Repeat(true, F).ToArray()),
        ("retake 混合（后半 true）", Enumerable.Range(0, F).Select(i => i >= F / 2).ToArray()),
    })
    {
        var feeds = new List<NamedOnnxValue>
        {
            NvL("tokens", tokens, 1, 3), NvL("languages", langs, 1, 3), NvL("durations", durations, 1, 3),
            NvF("f0", Fill(F, 300f), 1, F),
            NamedOnnxValue.CreateFromTensor("retake", new DenseTensor<bool>(mask, new[] { 1, F })),
            NamedOnnxValue.CreateFromTensor("gt_mel", new DenseTensor<float>(new float[F * melBins], new[] { 1, F, melBins })),
            NamedOnnxValue.CreateFromTensor("depth", new DenseTensor<float>(new[] { 0.6f }, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("steps", new DenseTensor<long>(new[] { 10L }, new[] { 1 })),
            NamedOnnxValue.CreateFromTensor("noise", new DenseTensor<float>(new float[feats * outDims * F], new[] { 1, feats, outDims, F })),
        };
        try
        {
            using var r = ac.Run(feeds);
            var mel = r.First(v => v.Name == "mel").AsTensor<float>();
            Console.WriteLine($"  {label}: OK mel [{string.Join(",", mel.Dimensions.ToArray())}]");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  {label}: FAIL -> {ex.Message}");
        }
    }

    // —— 无缓存逐 bit 确定性：同一会话、相同输入跑两遍，比 max|run1-run2| ——
    Console.WriteLine("  -- 确定性（同输入跑两遍，无缓存）--");
    var detFeeds = new List<NamedOnnxValue>
    {
        NvL("tokens", tokens, 1, 3), NvL("languages", langs, 1, 3), NvL("durations", durations, 1, 3),
        NvF("f0", Fill(F, 300f), 1, F),
        NamedOnnxValue.CreateFromTensor("retake", new DenseTensor<bool>(Enumerable.Repeat(true, F).ToArray(), new[] { 1, F })),
        NamedOnnxValue.CreateFromTensor("gt_mel", new DenseTensor<float>(new float[F * melBins], new[] { 1, F, melBins })),
        NamedOnnxValue.CreateFromTensor("depth", new DenseTensor<float>(new[] { 0.6f }, new[] { 1 })),
        NamedOnnxValue.CreateFromTensor("steps", new DenseTensor<long>(new[] { 10L }, new[] { 1 })),
        NamedOnnxValue.CreateFromTensor("noise", new DenseTensor<float>(new float[feats * outDims * F], new[] { 1, feats, outDims, F })),
    };
    float[] Run1() { using var r = ac.Run(detFeeds); return r.First(v => v.Name == "mel").AsTensor<float>().ToArray(); }
    var a = Run1(); var b = Run1();
    double maxDiff = 0; for (int i = 0; i < a.Length; i++) maxDiff = Math.Max(maxDiff, Math.Abs(a[i] - b[i]));
    Console.WriteLine($"  DML 两遍 max|diff| = {maxDiff:G17}  ({(maxDiff == 0 ? "逐 bit 一致" : "非逐 bit")})");
    // 注：同进程内再建第二个 CPU session 会触发 onnxruntime-DirectML 原生 AccessViolation（已知冲突），
    //   故 CPU 对照/跨 EP 比对需另起进程，这里不做。
}

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

// —— IPC 基准：同一份声学输入，分别经 InProcessModelSession（进程内直跑）与 RemoteModelSession（管道→MLRuntime.exe）
//    各跑 N 遍，量 per-Run 耗时，隔离出「子进程比进程内多花多少」。子进程先跑、跑完即杀 MLRuntime，避免与随后的
//    进程内 DML 会话并存于同一 GPU；两模式喂完全相同的一份 feeds。 ——
static void BenchIpc(string voiceRoot, Dictionary<string, object?> cfg, string acousticPath)
{
    const int warmup = 2, iters = 8, F = 400;
    const long steps = 20;
    string mlExe = Environment.GetEnvironmentVariable("DIFFSINGER_MLRUNTIME_EXE")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "TuneLab", "Extensions", "diffsingerfortunelab", "mlruntime", "MLRuntime.exe");

    Console.WriteLine($"\n======== IPC BENCH: acoustic  F={F} steps={steps} warmup={warmup} iters={iters} ========");
    Console.WriteLine($"  模型: {Path.GetFileName(acousticPath)}");
    Console.WriteLine($"  MLRuntime.exe: {mlExe}  存在={File.Exists(mlExe)}");

    List<NamedOnnxValue> feeds;
    using (var metaSession = Open(acousticPath))   // 只读元数据构造 feeds，构造完即释放
        feeds = BuildAcousticFeeds(voiceRoot, cfg, metaSession, F, steps);

    double[] sub;
    {
        using var client = new RuntimeClient("directml",
            p => new PipeTransport(mlExe, p, l => Console.WriteLine($"    [MLRuntime] {l}")), canRespawn: true);
        var s = RemoteModelSession.Load(client, acousticPath);
        sub = TimeRuns(s, feeds, warmup, iters);
    }

    double[] inp;
    {
        using var session = Open(acousticPath);
        var s = new InProcessModelSession(session);
        inp = TimeRuns(s, feeds, warmup, iters);
    }

    Report("进程内 in-process", inp);
    Report("子进程 subprocess", sub);
    double mi = Median(inp), ms = Median(sub);
    Console.WriteLine("\n  === 结论 ===");
    Console.WriteLine($"  进程内 中位 {mi:F1} ms  |  子进程 中位 {ms:F1} ms");
    Console.WriteLine($"  子进程额外开销 = {ms - mi:F1} ms/遍  ({(ms - mi) / mi * 100:F1}% 相对进程内)");
}

static double[] TimeRuns(IModelSession s, IReadOnlyCollection<NamedOnnxValue> feeds, int warmup, int iters)
{
    for (int i = 0; i < warmup; i++) { _ = s.Run(feeds); }
    var t = new double[iters];
    for (int i = 0; i < iters; i++)
    {
        var sw = Stopwatch.StartNew();
        _ = s.Run(feeds);
        sw.Stop();
        t[i] = sw.Elapsed.TotalMilliseconds;
    }
    return t;
}

static void Report(string label, double[] t)
{
    Console.WriteLine($"\n  [{label}] 每遍(ms): {string.Join(", ", t.Select(x => x.ToString("F1")))}");
    Console.WriteLine($"    中位={Median(t):F1}  均值={t.Average():F1}  min={t.Min():F1}  max={t.Max():F1}");
}

static double Median(double[] a)
{
    var b = a.OrderBy(x => x).ToArray();
    int n = b.Length;
    return n % 2 == 1 ? b[n / 2] : (b[n / 2 - 1] + b[n / 2]) / 2;
}

static List<NamedOnnxValue> BuildAcousticFeeds(string voiceRoot, Dictionary<string, object?> cfg, InferenceSession m, int F, long steps)
{
    var yaml = new DeserializerBuilder().Build();
    var phonemes = yaml.Deserialize<Dictionary<string, int>>(File.ReadAllText(ResolvePath(voiceRoot, cfg, "phonemes")));
    var languages = yaml.Deserialize<Dictionary<string, int>>(File.ReadAllText(ResolvePath(voiceRoot, cfg, "languages")));
    int hidden = int.Parse(cfg["hidden_size"]!.ToString()!);

    long[] tokens = { phonemes["SP"], phonemes.GetValueOrDefault("zh/a", 1), phonemes["SP"] };
    long[] langs = { languages.GetValueOrDefault("zh", 0), languages.GetValueOrDefault("zh", 0), languages.GetValueOrDefault("zh", 0) };
    long[] durations = { 8, F - 16, 8 };
    int nTokens = 3, nFrames = F;

    var feeds = new List<NamedOnnxValue>
    {
        NvL("tokens", tokens, 1, nTokens),
        NvL("durations", durations, 1, nTokens),
        NvF("f0", Fill(nFrames, 220f), 1, nFrames),
    };
    bool Has(string n) => m.InputMetadata.ContainsKey(n);
    void OptF(string n, float v) { if (Has(n)) feeds.Add(NvF(n, Fill(nFrames, v), 1, nFrames)); }

    if (Has("languages")) feeds.Add(NvL("languages", langs, 1, nTokens));
    OptF("breathiness", 0); OptF("voicing", 0); OptF("tension", 0); OptF("energy", 0);
    OptF("gender", 0); OptF("velocity", 1);

    if (Has("spk_embed"))
    {
        string spk0 = ((List<object>)cfg["speakers"]!)[0].ToString()!;
        var emb = ReadEmb(Path.Combine(voiceRoot, spk0 + ".emb"), hidden);
        var spk = new float[nFrames * hidden];
        for (int f = 0; f < nFrames; f++) Array.Copy(emb, 0, spk, f * hidden, hidden);
        feeds.Add(NamedOnnxValue.CreateFromTensor("spk_embed", new DenseTensor<float>(spk, new[] { 1, nFrames, hidden })));
    }
    if (Has("depth"))
    {
        float depth = float.Parse(cfg.GetValueOrDefault("max_depth")?.ToString() ?? "1.0");
        feeds.Add(NamedOnnxValue.CreateFromTensor("depth", new DenseTensor<float>(new[] { depth }, Array.Empty<int>())));
    }
    if (Has("steps"))
        feeds.Add(NamedOnnxValue.CreateFromTensor("steps", new DenseTensor<long>(new[] { steps }, Array.Empty<int>())));
    else if (Has("speedup"))
        feeds.Add(NamedOnnxValue.CreateFromTensor("speedup", new DenseTensor<long>(new[] { Math.Max(1L, 1000 / steps) }, Array.Empty<int>())));

    // 外置噪声 fork 模型的附加输入（stock 模型无这些口，跳过）。
    if (Has("retake"))
        feeds.Add(NamedOnnxValue.CreateFromTensor("retake", new DenseTensor<bool>(Enumerable.Repeat(true, nFrames).ToArray(), new[] { 1, nFrames })));
    if (Has("gt_mel"))
    {
        int melBins = m.InputMetadata["gt_mel"].Dimensions[2];
        feeds.Add(NamedOnnxValue.CreateFromTensor("gt_mel", new DenseTensor<float>(new float[nFrames * melBins], new[] { 1, nFrames, melBins })));
    }
    if (Has("noise"))
    {
        var nd = m.InputMetadata["noise"].Dimensions;
        int feats = nd[1], outDims = nd[2];
        feeds.Add(NamedOnnxValue.CreateFromTensor("noise", new DenseTensor<float>(new float[feats * outDims * nFrames], new[] { 1, feats, outDims, nFrames })));
    }
    return feeds;
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

// dur 全 part 计时基准：按真实 part 量级合成超长音素序列，量 linguistic+dur 一次前向耗时。
//   内容对耗时无关紧要（计算量随序列长），故用合法 token 凑长度；标识全运行时解析、不硬编码敏感名。
static void PredictDurBench(string voiceRoot)
{
    Console.WriteLine("\n======== DUR BENCH (全 part 一次 dsdur 前向耗时) ========");
    var yaml = new DeserializerBuilder().Build();
    string durDir = Path.Combine(voiceRoot, "dsdur");
    var cfg = yaml.Deserialize<Dictionary<string, object?>>(File.ReadAllText(Path.Combine(durDir, "dsconfig.yaml")));
    var phonemes = yaml.Deserialize<Dictionary<string, int>>(File.ReadAllText(Path.Combine(durDir, cfg["phonemes"]!.ToString()!)));
    int hidden = int.Parse(cfg["hidden_size"]!.ToString()!);
    int hop = int.Parse(cfg["hop_size"]!.ToString()!);
    int sr = int.Parse(cfg["sample_rate"]!.ToString()!);

    // 首个 .emb 作 spk（运行时解析、不写死角色名）。
    string embPath = Directory.GetFiles(durDir, "*.emb").OrderBy(p => p).First();
    float[] emb = ReadEmb(embPath, hidden);

    using var ling = Open(Path.Combine(durDir, cfg["linguistic"]!.ToString()!));
    using var dur = Open(Path.Combine(durDir, cfg["dur"]!.ToString()!));
    bool useLang = ling.InputMetadata.ContainsKey("languages");

    // 取两个合法 token 循环填充（凑长度即可）；每词 3 音素、每词 ~0.3s。
    int tokA = phonemes.GetValueOrDefault("SP", phonemes.Values.First());
    int tokB = phonemes.Values.Where(v => v != tokA).DefaultIfEmpty(tokA).First();
    int perWord = 3, wordFrames = (int)Math.Round(0.3 * sr / hop);

    foreach (int nPhonemes in new[] { 100, 250, 500, 1000, 2000, 4000 })
    {
        int nWords = nPhonemes / perWord;
        int nTokens = nWords * perWord;
        var tokens = new long[nTokens];
        var langs = new long[nTokens];
        var phMidi = new long[nTokens];
        for (int i = 0; i < nTokens; i++) { tokens[i] = i % 2 == 0 ? tokA : tokB; langs[i] = 0; phMidi[i] = 60; }
        var wordDiv = new long[nWords];
        var wordDur = new long[nWords];
        for (int w = 0; w < nWords; w++) { wordDiv[w] = perWord; wordDur[w] = wordFrames; }
        var spk = new float[nTokens * hidden];
        for (int i = 0; i < nTokens; i++) Array.Copy(emb, 0, spk, i * hidden, hidden);

        Func<double> runOnce = () =>
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var lingIn = new List<NamedOnnxValue>
            {
                NvL("tokens", tokens, 1, nTokens),
                NvL("word_div", wordDiv, 1, nWords),
                NvL("word_dur", wordDur, 1, nWords),
            };
            if (useLang) lingIn.Add(NvL("languages", langs, 1, nTokens));
            using var lingOut = ling.Run(lingIn);
            var enc = lingOut.First(v => v.Name == "encoder_out").AsTensor<float>();
            var msk = lingOut.First(v => v.Name == "x_masks").AsTensor<bool>();
            var encDense = new DenseTensor<float>(enc.ToArray(), enc.Dimensions.ToArray());
            var mskDense = new DenseTensor<bool>(msk.ToArray(), msk.Dimensions.ToArray());
            using var durOut = dur.Run(new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("encoder_out", encDense),
                NamedOnnxValue.CreateFromTensor("x_masks", mskDense),
                NvL("ph_midi", phMidi, 1, nTokens),
                NamedOnnxValue.CreateFromTensor("spk_embed", new DenseTensor<float>(spk, new[] { 1, nTokens, hidden })),
            });
            _ = durOut.First(v => v.Name == "ph_dur_pred").AsTensor<float>().ToArray();
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds;
        };

        double cold = runOnce();          // 冷跑（含图优化/分配）
        var warm = new List<double>();
        for (int r = 0; r < 5; r++) warm.Add(runOnce());
        warm.Sort();
        Console.WriteLine($"  {nPhonemes,5} 音素 ({nWords,4} 词): 冷 {cold,7:F1}ms | 暖 min {warm[0],6:F1}ms / 中位 {warm[2],6:F1}ms");
    }
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

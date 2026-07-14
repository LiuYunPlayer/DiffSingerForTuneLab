using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using TuneLab.Foundation;
using TuneLab.SDK;
using YamlDotNet.Serialization;

namespace DiffSingerForTuneLab;

// 引擎级模型缓存：按 voiceId 缓存声学模型束、按声码器名共享声码器会话。
//   · 懒加载（首次合成时按需载入）、加锁串行化首载；载入后多会话共享同一会话，但 DirectML EP 的 Run
//     是设备级不可并发（跨会话亦然），故所有推理 Run 进程级全局串行（见 InProcessModelSession）；
//   · 执行设备是进程级同质承诺、不做进程内 DML→CPU 回退：onnxruntime-DirectML 单包内，同进程一旦碰过 DML EP，
//     再建 CPU 会话（或反之）会触发原生 AccessViolation 直接崩进程。故 directml 失败即抛异常上冒、绝不就地建 CPU 会话；
//     用户需改设置为 CPU 并重启（新进程从未碰 DML、纯 CPU 安全）。彻底隔离/自愈留给后续 MLRuntime.exe 子进程方案；
//   · provider 变更 → 引擎弃旧缓存建新缓存（旧缓存 Dispose 释放原生会话）。
// 原生 onnxruntime/DirectML 库随包进本插件 ALC（见 csproj 注释）；首次构造 InferenceSession 即触发原生加载，
// 失败抛异常由宿主在调用边界 catch 标插件失败（§5.10）。
public sealed class DiffSingerModelCache : IDisposable
{
    readonly string mProvider;
    readonly ILogger mLogger;
    readonly object mLock = new();
    readonly Dictionary<string, VoiceModels> mVoices = new(StringComparer.Ordinal);
    readonly Dictionary<string, (IModelSession Session, ulong Hash)> mVocoders = new(StringComparer.Ordinal);

    // MLRuntime 路由：推理模式由设置决定（默认 subprocess——隔离子进程跑 ONNX，原生崩溃不拖垮宿主）：
    //   · "subprocess"（默认）→ PipeTransport：spawn mlruntime/MLRuntime.exe；启动探测失败（杀软/权限）则退回进程内。
    //   · "inprocess"          → mRuntimeClient=null，走 InProcessModelSession（进程内直跑，逃生模式）。
    //   · env DIFFSINGER_RUNTIME_LOOPBACK=1（dev）→ LoopbackTransport：进程内 host、走完整 IPC 编解码链（测试用）。
    //   非 null 的 mRuntimeClient 令 LoadSession 返回 RemoteModelSession。
    readonly RuntimeClient? mRuntimeClient;

    public DiffSingerModelCache(string provider, string runtimeMode, ILogger logger)
    {
        mProvider = provider;
        mLogger = logger;
        if (Environment.GetEnvironmentVariable("DIFFSINGER_RUNTIME_LOOPBACK") == "1")
        {
            mRuntimeClient = new RuntimeClient(provider, p => new LoopbackTransport(new RuntimeHost(p)), canRespawn: false);
            mLogger.Info("DiffSinger：MLRuntime loopback 模式（dev）");
        }
        else if (runtimeMode != "inprocess")   // 默认 subprocess
        {
            mRuntimeClient = TryCreateSubprocessClient(provider);
        }
    }

    // 建子进程客户端：先立即 spawn+连一个传输作启动探测——成功则以它为初始传输建弹性 client；
    //   失败（exe 缺失/杀软拦截/权限/管道建不起）则记警告、返回 null 退回进程内（这是「起不来」非「崩」，退回救场）。
    RuntimeClient? TryCreateSubprocessClient(string provider)
    {
        var dir = Path.GetDirectoryName(typeof(DiffSingerModelCache).Assembly.Location)!;
        var exe = Path.Combine(dir, "mlruntime", "MLRuntime.exe");
        Action<string> exeLog = line => mLogger.Info($"[MLRuntime] {line}");   // 子进程 stdout/stderr 转发
        Func<string, IRuntimeTransport> factory = p => new PipeTransport(exe, p, exeLog);
        try
        {
            var probe = factory(provider);   // 立即 spawn+连，探测子进程能否起来
            mLogger.Info($"DiffSinger：MLRuntime 子进程模式（{exe}）");
            return new RuntimeClient(provider, factory, canRespawn: true, line => mLogger.Info(line), probe);
        }
        catch (Exception ex)
        {
            mLogger.Warning($"DiffSinger：MLRuntime 子进程启动失败（{ex.Message}），本次退回进程内。可在插件设置的「推理模式」调整，或检查杀软/权限。");
            return null;
        }
    }

    // 默认声码器根：插件用户数据根下的 Vocoders 目录（与声库目录彼此独立，仅默认值恰好同在用户数据根下）；声学 dsconfig 的 vocoder 字段即子目录名。
    // 作为可配置列表的默认播种项——用户可在设置里追加其他目录（如已有的声码器收藏位置），免去复制。
    public static string VocodersDirectory => Path.Combine(DiffSingerDeclarations.UserDataRoot, "Vocoders");

    // 声码器搜索根（默认 + 用户追加，由引擎每次取会话前刷入）；按序取首个含 <vocoderName>/vocoder.yaml 的目录。
    // 引用整体赋值原子、读取取一次快照即可；默认仅含固定目录，引擎未设置时行为与旧版一致。
    public IReadOnlyList<string> VocoderRoots { get; set; } = new[] { VocodersDirectory };

    // 按物理模型包目录（config.RootPath）缓存——同一 voice 选不同 model/version = 不同物理包 = 各自缓存、不重复加载。
    public VoiceModels GetOrLoad(VoicebankConfig config)
    {
        lock (mLock)
        {
            if (mVoices.TryGetValue(config.RootPath, out var cached))
                return cached;

            var acousticPath = Path.Combine(config.RootPath, config.AcousticFileName);
            var acoustic = LoadSession(acousticPath);
            var acousticHash = DiffSingerTensorCache.HashFile(acousticPath);
            var (vocoder, vocoderHash) = GetOrLoadVocoder(config);
            // 预测器（dsdur/dspitch/dsvariance）懒加载共用本缓存的会话加载器（含 DML→CPU 回退）。
            var models = new VoiceModels(config, acoustic, acousticHash, vocoder, vocoderHash, LoadSession, mLogger);
            mVoices[config.RootPath] = models;
            return models;
        }
    }

    (IModelSession Session, ulong Hash) GetOrLoadVocoder(VoicebankConfig config)
    {
        var vocoderName = config.VocoderName;
        var dir = ResolveVocoderDir(config);
        if (dir is null)
            throw new InvalidOperationException(
                $"未找到声码器 {vocoderName}：声库根下无 dsvocoder/vocoder.yaml（bundled），" +
                $"且声码器目录下无 {vocoderName}/vocoder.yaml。" +
                $"（已搜索声码器目录：{string.Join(" ; ", VocoderRoots)}）请把声码器随声库放入 dsvocoder/，" +
                $"或放入声码器目录之一（子目录名须等于 dsconfig 的 vocoder 字段），或在插件设置的「声码器目录」里追加其所在目录。");

        // 按解析出的物理目录缓存，而非按 dsconfig.vocoder 名：bundled dsvocoder 各声库自带、内容不同，
        // 不可按名共享（不同声库 dsconfig.vocoder 可能同名却指向各自的 dsvocoder）；全局共享声码器则同目录天然复用。
        var cacheKey = Path.GetFullPath(dir);
        if (mVocoders.TryGetValue(cacheKey, out var cached))
            return cached;

        var yaml = new DeserializerBuilder().Build()
            .Deserialize<Dictionary<string, object?>>(File.ReadAllText(Path.Combine(dir, "vocoder.yaml")));
        var modelFile = yaml.TryGetValue("model", out var m) ? m?.ToString() : null;
        if (string.IsNullOrEmpty(modelFile))
            throw new InvalidOperationException($"声码器目录 {dir} 的 vocoder.yaml 缺少 model 字段");

        var modelPath = Path.Combine(dir, modelFile);
        var entry = (LoadSession(modelPath), DiffSingerTensorCache.HashFile(modelPath));
        mVocoders[cacheKey] = entry;
        return entry;
    }

    // 声码器目录解析，对齐 OpenUtau 的 getVocoder()（两级）：
    //   0) bundled：声库根下 dsvocoder/vocoder.yaml 存在即无条件用之，忽略 dsconfig 的 vocoder 字段
    //      （OpenUtau 语义：声库自带声码器优先。常见于 dsconfig 从模板复制、vocoder 行没改，但库里塞了自己的 dsvocoder）；
    //   1) 全局：<root>/<dsconfig.vocoder>/vocoder.yaml——按目录名命中（对应 OpenUtau 的「依赖目录/dsconfig.vocoder」）。
    string? ResolveVocoderDir(VoicebankConfig config)
    {
        // 0) bundled：声库自带 dsvocoder/ 优先（OpenUtau 语义，无条件压过 dsconfig.vocoder）
        var bundled = Path.Combine(config.RootPath, "dsvocoder");
        if (File.Exists(Path.Combine(bundled, "vocoder.yaml")))
        {
            mLogger.Info($"DiffSinger：使用声库自带声码器 dsvocoder/（忽略 dsconfig 的 vocoder 字段「{config.VocoderName}」）。");
            return bundled;
        }
        // 1) 全局：按目录名命中 dsconfig.vocoder
        foreach (var root in VocoderRoots)
        {
            if (string.IsNullOrWhiteSpace(root))
                continue;
            var dir = Path.Combine(root, config.VocoderName);
            if (File.Exists(Path.Combine(dir, "vocoder.yaml")))
                return dir;
        }
        return null;
    }

    // 执行设备同质承诺，不做进程内回退（见类注释的 AccessViolation 约束）：
    //   · cpu 设置：全程只建 CPU 会话，绝不碰 DML EP；
    //   · directml 设置：建 DML 会话，失败即抛可读异常上冒（宿主按 §5.10 标合成失败），绝不就地回退 CPU。
    IModelSession LoadSession(string modelPath)
    {
        var fileName = Path.GetFileName(modelPath);
        if (mRuntimeClient != null)   // loopback 验证模式：会话在 host 侧、经 IPC 派发
        {
            var remote = RemoteModelSession.Load(mRuntimeClient, modelPath);
            mLogger.Info($"DiffSinger：加载 {fileName} · MLRuntime({mProvider})");
            return remote;
        }
        if (mProvider == "cpu")
        {
            // 见 RuntimeHost.LoadSession：onnxruntime 1.20.1 CPU EP 扩展层图优化在 DiffSinger 声学图上原生崩溃，封顶 BASIC。
            var cpuOptions = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC };
            var cpu = new InferenceSession(modelPath, cpuOptions);
            mLogger.Info($"DiffSinger：加载 {fileName} · CPU");
            return new InProcessModelSession(cpu);
        }

        try
        {
            var options = new SessionOptions();
            options.AppendExecutionProvider_DML(0);
            var session = new InferenceSession(modelPath, options);
            mLogger.Info($"DiffSinger：加载 {fileName} · DirectML");
            return new InProcessModelSession(session);
        }
        catch (Exception ex)
        {
            // 全量异常入日志（含类型/HResult/inner/栈），供定位 DML 失败根因——是算子不支持还是设备层起不来。
            mLogger.Error($"DiffSinger：DirectML 加载 {fileName} 失败：{ex}");
            throw new InvalidOperationException(
                $"DirectML 加载 {fileName} 失败，无法在同进程内安全回退 CPU。" +
                $"请在插件设置里把「执行设备」改为 CPU 并重启 TuneLab。原始错误：{ex.Message}", ex);
        }
    }

    public void Dispose()
    {
        lock (mLock)
        {
            foreach (var v in mVoices.Values)
                v.Dispose();   // 释放声学 + 懒加载的预测器（声码器为缓存共享、单独释放）
            // 声码器会话（会话自身在设备级锁内退役并释放，杜绝与在飞 Run 并发释放）。
            foreach (var entry in mVocoders.Values)
                entry.Session.Dispose();
            mVoices.Clear();
            mVocoders.Clear();
            // loopback 模式：真实会话在 host 侧，释放 client 连带释放 host 及其所有会话（RemoteModelSession.Dispose 为空操作）。
            mRuntimeClient?.Dispose();
        }
    }
}

// 一个声库的已加载模型束（声学私有、声码器为缓存共享引用）+ 推理所需的音素/语言表与说话人嵌入。
public sealed class VoiceModels : IDisposable
{
    readonly VoicebankConfig mConfig;
    readonly ILogger mLogger;
    readonly Func<string, IModelSession> mLoad;
    readonly Dictionary<string, int> mPhonemes;
    readonly Dictionary<string, int> mLanguages;
    readonly Dictionary<string, float[]> mEmbCache = new(StringComparer.Ordinal);
    readonly object mEmbLock = new();
    // 预测器懒加载缓存（键 = 子目录名）；缺失或加载失败记 null（不重试）。
    readonly Dictionary<string, DiffSingerPredictor?> mPredictors = new(StringComparer.Ordinal);
    readonly object mPredictorLock = new();

    public IModelSession Acoustic { get; }
    public IModelSession Vocoder { get; }

    // 张量缓存 identifier（模型 .onnx 文件内容哈希，加载时算一次）。
    public ulong AcousticHash { get; }
    public ulong VocoderHash { get; }

    public int HiddenSize => mConfig.HiddenSize;
    public int HopSize => mConfig.HopSize;
    public int SampleRate => mConfig.SampleRate;
    public int NumMelBins => mConfig.NumMelBins;
    public double MaxDepth => mConfig.MaxDepth;
    public IReadOnlyList<string> Speakers => mConfig.Speakers;

    public VoiceModels(VoicebankConfig config, IModelSession acoustic, ulong acousticHash,
        IModelSession vocoder, ulong vocoderHash, Func<string, IModelSession> load, ILogger logger)
    {
        mConfig = config;
        mLogger = logger;
        mLoad = load;
        Acoustic = acoustic;
        Vocoder = vocoder;
        AcousticHash = acousticHash;
        VocoderHash = vocoderHash;

        var yaml = new DeserializerBuilder().Build();   // JSON ⊂ YAML：音素/语言表（值为 int id）
        mPhonemes = yaml.Deserialize<Dictionary<string, int>>(
            File.ReadAllText(Path.Combine(config.RootPath, config.PhonemesFileName)));
        mLanguages = string.IsNullOrEmpty(config.LanguagesFileName)
            ? new Dictionary<string, int>()
            : yaml.Deserialize<Dictionary<string, int>>(
                File.ReadAllText(Path.Combine(config.RootPath, config.LanguagesFileName)));
    }

    public bool TryGetPhoneme(string symbol, out int id) => mPhonemes.TryGetValue(symbol, out id);
    public bool TryGetLanguage(string lang, out int id) => mLanguages.TryGetValue(lang, out id);

    // 子目录 → 该预测器唯一职责 role（对齐 OpenUtau：dsdur 只用 dur、dspitch 只用 pitch、dsvariance 只用 variance）。
    //   仅加载本职 role + linguistic；dsconfig 里其余 role 字段（打包遗留/串味）一律忽略——即便指向不存在的文件也不影响加载。
    static readonly Dictionary<string, string> PredictorRole = new(StringComparer.Ordinal)
    {
        ["dsdur"] = "dur", ["dspitch"] = "pitch", ["dsvariance"] = "variance",
    };

    // 预测器懒加载（子目录如 "dsvariance" / "dspitch"）：首用时按声库会话加载器构造；
    // 子目录无 dsconfig 或加载失败返回 null（记忆化，不重试），调用方据此降级。
    public DiffSingerPredictor? GetPredictor(string subdir)
    {
        lock (mPredictorLock)
        {
            if (mPredictors.TryGetValue(subdir, out var cached))
                return cached;

            DiffSingerPredictor? predictor = null;
            var dir = Path.Combine(mConfig.RootPath, subdir);
            if (File.Exists(Path.Combine(dir, "dsconfig.yaml")))
            {
                // 未知子目录（无映射）退化为加载全部 role，保守不改老行为。
                var role = PredictorRole.GetValueOrDefault(subdir);
                try { predictor = new DiffSingerPredictor(dir, mLoad, role); }
                catch (Exception ex) { mLogger.Warning($"DiffSinger：加载预测器 {subdir} 失败：{ex.Message}"); }
            }
            mPredictors[subdir] = predictor;
            return predictor;
        }
    }

    // 说话人嵌入向量（.emb = HiddenSize 个 float32 LE）：按 suffix 关联同后缀声学条目、无则回退首个；按 suffix 缓存。
    //   忠实对齐 OpenUtau getSpeakerIndexBySuffix 的「按 suffix 解析、回退首个」；供 DiffSingerSpeakerMix 逐帧混合解析声学域 emb。
    public float[] GetSpeakerEmbeddingBySuffix(string suffix)
    {
        lock (mEmbLock)
        {
            if (mEmbCache.TryGetValue(suffix, out var cached))
                return cached;

            string? match = mConfig.Speakers.FirstOrDefault(s => DiffSingerDeclarations.Suffix(s) == suffix)
                ?? (mConfig.Speakers.Count > 0 ? mConfig.Speakers[0] : null);
            var emb = new float[mConfig.HiddenSize];
            if (match != null)
            {
                var bytes = File.ReadAllBytes(Path.Combine(mConfig.RootPath, match + ".emb"));
                for (int i = 0; i < emb.Length; i++)
                    emb[i] = BitConverter.ToSingle(bytes, i * 4);
            }
            mEmbCache[suffix] = emb;
            return emb;
        }
    }

    // 释放声学 + 懒加载的预测器；声码器为缓存共享引用，由缓存单独释放（不在此 Dispose）。
    //   会话经退役机制在推理锁内释放（杜绝与在飞 Run 并发释放）；预测器自退役其会话。
    public void Dispose()
    {
        Acoustic.Dispose();   // 会话自身在设备级锁内退役并释放
        lock (mPredictorLock)
        {
            foreach (var p in mPredictors.Values)
                p?.Dispose();
            mPredictors.Clear();
        }
    }
}

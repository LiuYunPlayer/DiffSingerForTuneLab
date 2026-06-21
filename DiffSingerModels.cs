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
//   · 懒加载（首次合成时按需载入）、加锁串行化首载，ORT 的 Run 并发安全故载入后多会话共享同一会话；
//   · 执行设备由引擎设置决定：directml 失败（无 DX12/驱动）回退 CPU，回退在每个模型载入时判定并落地；
//   · provider 变更 → 引擎弃旧缓存建新缓存（旧缓存 Dispose 释放原生会话）。
// 原生 onnxruntime/DirectML 库随包进本插件 ALC（见 csproj 注释）；首次构造 InferenceSession 即触发原生加载，
// 失败抛异常由宿主在调用边界 catch 标插件失败（§5.10）。
public sealed class DiffSingerModelCache : IDisposable
{
    readonly string mProvider;
    readonly ILogger mLogger;
    readonly object mLock = new();
    readonly Dictionary<string, VoiceModels> mVoices = new(StringComparer.Ordinal);
    readonly Dictionary<string, (InferenceSession Session, ulong Hash)> mVocoders = new(StringComparer.Ordinal);

    public DiffSingerModelCache(string provider, ILogger logger)
    {
        mProvider = provider;
        mLogger = logger;
    }

    // 声码器根目录约定：插件用户数据根下、与声库 Voices 目录并列的 Vocoders 目录；声学 dsconfig 的 vocoder 字段即子目录名。
    public static string VocodersDirectory => Path.Combine(DiffSingerDeclarations.UserDataRoot, "Vocoders");

    public VoiceModels GetOrLoad(string voiceId, VoicebankConfig config)
    {
        lock (mLock)
        {
            if (mVoices.TryGetValue(voiceId, out var cached))
                return cached;

            var acousticPath = Path.Combine(config.RootPath, config.AcousticFileName);
            var acoustic = LoadSession(acousticPath);
            var acousticHash = DiffSingerTensorCache.HashFile(acousticPath);
            var (vocoder, vocoderHash) = GetOrLoadVocoder(config.VocoderName);
            // 预测器（dsdur/dspitch/dsvariance）懒加载共用本缓存的会话加载器（含 DML→CPU 回退）。
            var models = new VoiceModels(config, acoustic, acousticHash, vocoder, vocoderHash, LoadSession, mLogger);
            mVoices[voiceId] = models;
            return models;
        }
    }

    (InferenceSession Session, ulong Hash) GetOrLoadVocoder(string vocoderName)
    {
        if (mVocoders.TryGetValue(vocoderName, out var cached))
            return cached;

        var dir = Path.Combine(VocodersDirectory, vocoderName);
        var yaml = new DeserializerBuilder().Build()
            .Deserialize<Dictionary<string, object?>>(File.ReadAllText(Path.Combine(dir, "vocoder.yaml")));
        var modelFile = yaml.TryGetValue("model", out var m) ? m?.ToString() : null;
        if (string.IsNullOrEmpty(modelFile))
            throw new InvalidOperationException($"声码器 {vocoderName} 的 vocoder.yaml 缺少 model 字段");

        var modelPath = Path.Combine(dir, modelFile);
        var entry = (LoadSession(modelPath), DiffSingerTensorCache.HashFile(modelPath));
        mVocoders[vocoderName] = entry;
        return entry;
    }

    // directml 优先、建会话失败回退 CPU（cpu 设置直接走 CPU）。
    InferenceSession LoadSession(string modelPath)
    {
        var fileName = Path.GetFileName(modelPath);
        if (mProvider != "cpu")
        {
            try
            {
                var options = new SessionOptions();
                options.AppendExecutionProvider_DML(0);
                var session = new InferenceSession(modelPath, options);
                mLogger.Info($"DiffSinger：加载 {fileName} · DirectML");
                return session;
            }
            catch (Exception ex)
            {
                mLogger.Warning($"DiffSinger：DirectML 加载 {fileName} 失败，回退 CPU：{ex.Message}");
            }
        }

        var cpu = new InferenceSession(modelPath);
        mLogger.Info($"DiffSinger：加载 {fileName} · CPU");
        return cpu;
    }

    public void Dispose()
    {
        lock (mLock)
        {
            foreach (var v in mVoices.Values)
                v.Dispose();   // 释放声学 + 懒加载的预测器（声码器为缓存共享、单独释放）
            foreach (var entry in mVocoders.Values)
                entry.Session.Dispose();
            mVoices.Clear();
            mVocoders.Clear();
        }
    }
}

// 一个声库的已加载模型束（声学私有、声码器为缓存共享引用）+ 推理所需的音素/语言表与说话人嵌入。
public sealed class VoiceModels : IDisposable
{
    readonly VoicebankConfig mConfig;
    readonly ILogger mLogger;
    readonly Func<string, InferenceSession> mLoad;
    readonly Dictionary<string, int> mPhonemes;
    readonly Dictionary<string, int> mLanguages;
    readonly Dictionary<string, float[]> mEmbCache = new(StringComparer.Ordinal);
    readonly object mEmbLock = new();
    // 预测器懒加载缓存（键 = 子目录名）；缺失或加载失败记 null（不重试）。
    readonly Dictionary<string, DiffSingerPredictor?> mPredictors = new(StringComparer.Ordinal);
    readonly object mPredictorLock = new();

    readonly object mAcousticLock = new();
    readonly object mVocoderLock = new();

    public InferenceSession Acoustic { get; }
    public InferenceSession Vocoder { get; }

    // 张量缓存 identifier（模型 .onnx 文件内容哈希，加载时算一次）。
    public ulong AcousticHash { get; }
    public ulong VocoderHash { get; }

    public int HiddenSize => mConfig.HiddenSize;
    public int HopSize => mConfig.HopSize;
    public int SampleRate => mConfig.SampleRate;
    public int NumMelBins => mConfig.NumMelBins;
    public double MaxDepth => mConfig.MaxDepth;
    public IReadOnlyList<string> Speakers => mConfig.Speakers;

    public VoiceModels(VoicebankConfig config, InferenceSession acoustic, ulong acousticHash,
        InferenceSession vocoder, ulong vocoderHash, Func<string, InferenceSession> load, ILogger logger)
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
                try { predictor = new DiffSingerPredictor(dir, mLoad); }
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
    public void Dispose()
    {
        Acoustic.Dispose();
        lock (mPredictorLock)
        {
            foreach (var p in mPredictors.Values)
                p?.Dispose();
            mPredictors.Clear();
        }
    }
}

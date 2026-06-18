using System;
using System.Collections.Generic;
using System.IO;
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
    readonly Dictionary<string, InferenceSession> mVocoders = new(StringComparer.Ordinal);

    public DiffSingerModelCache(string provider, ILogger logger)
    {
        mProvider = provider;
        mLogger = logger;
    }

    // 声码器根目录约定（本阶段定稿）：与声库 Voices 目录并列的 Vocoders 目录；声学 dsconfig 的 vocoder 字段即子目录名。
    public static string VocodersDirectory => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "TuneLab", "DiffSinger", "Vocoders");

    public VoiceModels GetOrLoad(string voiceId, VoicebankConfig config)
    {
        lock (mLock)
        {
            if (mVoices.TryGetValue(voiceId, out var cached))
                return cached;

            var acoustic = LoadSession(Path.Combine(config.RootPath, config.AcousticFileName));
            var vocoder = GetOrLoadVocoder(config.VocoderName);
            var models = new VoiceModels(config, acoustic, vocoder, mLogger);
            mVoices[voiceId] = models;
            return models;
        }
    }

    InferenceSession GetOrLoadVocoder(string vocoderName)
    {
        if (mVocoders.TryGetValue(vocoderName, out var cached))
            return cached;

        var dir = Path.Combine(VocodersDirectory, vocoderName);
        var yaml = new DeserializerBuilder().Build()
            .Deserialize<Dictionary<string, object?>>(File.ReadAllText(Path.Combine(dir, "vocoder.yaml")));
        var modelFile = yaml.TryGetValue("model", out var m) ? m?.ToString() : null;
        if (string.IsNullOrEmpty(modelFile))
            throw new InvalidOperationException($"声码器 {vocoderName} 的 vocoder.yaml 缺少 model 字段");

        var session = LoadSession(Path.Combine(dir, modelFile));
        mVocoders[vocoderName] = session;
        return session;
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
                v.Acoustic.Dispose();
            foreach (var s in mVocoders.Values)
                s.Dispose();
            mVoices.Clear();
            mVocoders.Clear();
        }
    }
}

// 一个声库的已加载模型束（声学私有、声码器为缓存共享引用）+ 推理所需的音素/语言表与说话人嵌入。
public sealed class VoiceModels
{
    readonly VoicebankConfig mConfig;
    readonly ILogger mLogger;
    readonly Dictionary<string, int> mPhonemes;
    readonly Dictionary<string, int> mLanguages;
    readonly Dictionary<string, float[]> mEmbCache = new(StringComparer.Ordinal);
    readonly object mEmbLock = new();

    public InferenceSession Acoustic { get; }
    public InferenceSession Vocoder { get; }

    public int HiddenSize => mConfig.HiddenSize;
    public int HopSize => mConfig.HopSize;
    public int SampleRate => mConfig.SampleRate;
    public int NumMelBins => mConfig.NumMelBins;
    public double MaxDepth => mConfig.MaxDepth;
    public IReadOnlyList<string> Speakers => mConfig.Speakers;

    public VoiceModels(VoicebankConfig config, InferenceSession acoustic, InferenceSession vocoder, ILogger logger)
    {
        mConfig = config;
        mLogger = logger;
        Acoustic = acoustic;
        Vocoder = vocoder;

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

    // 说话人逐帧嵌入向量（.emb = HiddenSize 个 float32 LE）；按说话人缓存。
    public float[] GetSpeakerEmbedding(string speaker)
    {
        lock (mEmbLock)
        {
            if (mEmbCache.TryGetValue(speaker, out var cached))
                return cached;

            var bytes = File.ReadAllBytes(Path.Combine(mConfig.RootPath, speaker + ".emb"));
            var emb = new float[mConfig.HiddenSize];
            for (int i = 0; i < emb.Length; i++)
                emb[i] = BitConverter.ToSingle(bytes, i * 4);
            mEmbCache[speaker] = emb;
            return emb;
        }
    }
}

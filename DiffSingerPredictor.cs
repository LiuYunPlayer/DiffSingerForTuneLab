using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using YamlDotNet.Serialization;

namespace DiffSingerForTuneLab;

// 一个 DiffSinger 预测器子目录（dsdur / dspitch / dsvariance）的已加载束：
//   linguistic 编码器 + 角色模型（dur / variance / pitch）+ 自有 phonemes/languages 表 + 说话人 .emb
//   + G2P 词典（dsdict：entries 词条 + symbols 类型）。
// 各预测器有独立 phonemes/languages 表与 .emb（与声学的 260509a.* 不同），故符号→id、嵌入、G2P 都走本类。
// 由 VoiceModels 懒加载、跨会话共享；张量装配在编排层（DiffSingerPhonemizer / Render），本类只提供资源与查询。
// 忠实对齐 OpenUtau：DiffSingerBasePhonemizer / DiffSingerUtils（见记忆 diffsinger-predictor-io）。
public sealed class DiffSingerPredictor : IDisposable
{
    static readonly HashSet<string> AlwaysVowel = new(StringComparer.Ordinal) { "SP", "AP", "ExAP" };

    readonly string mDir;
    readonly Dictionary<string, int> mPhonemes;
    readonly Dictionary<string, int> mLanguages;
    readonly IReadOnlyList<string> mSpeakers;
    readonly int mHidden;
    readonly Dictionary<string, InferenceSession> mModels = new(StringComparer.Ordinal);
    readonly Dictionary<string, float[]> mEmbCache = new(StringComparer.Ordinal);
    readonly Dictionary<string, Dictionary<string, string[]>> mEntryCache = new(StringComparer.Ordinal);
    readonly Dictionary<string, string> mSymbolTypes = new(StringComparer.Ordinal);  // symbol → type（合并 dsdict）
    readonly object mLock = new();

    public InferenceSession Linguistic { get; }
    public int HiddenSize => mHidden;
    // linguistic 是否吃 word_div/word_dur（dsdur/dsvariance 词边界；dspitch 用已知 ph_dur）。
    public bool LinguisticUsesWordBoundary { get; }

    // 张量缓存 identifier（模型 .onnx 文件内容哈希，加载时算一次）：linguistic 与各 role 模型。
    public ulong LinguisticHash { get; }
    readonly Dictionary<string, ulong> mModelHashes = new(StringComparer.Ordinal);
    public ulong ModelHash(string role) => mModelHashes.TryGetValue(role, out var h) ? h : 0;

    public DiffSingerPredictor(string dir, Func<string, InferenceSession> load)
    {
        mDir = dir;
        var yaml = new DeserializerBuilder().Build();
        var cfg = yaml.Deserialize<Dictionary<string, object?>>(File.ReadAllText(Path.Combine(dir, "dsconfig.yaml")))
            ?? new Dictionary<string, object?>();
        string Get(string k) => cfg.TryGetValue(k, out var v) && v is string s ? s : string.Empty;

        mHidden = cfg.TryGetValue("hidden_size", out var h) && int.TryParse(h?.ToString(), out var hi) ? hi : 256;
        mPhonemes = LoadIntMap(Path.Combine(dir, Get("phonemes")));
        mLanguages = string.IsNullOrEmpty(Get("languages"))
            ? new Dictionary<string, int>()
            : LoadIntMap(Path.Combine(dir, Get("languages")));
        mSpeakers = cfg.TryGetValue("speakers", out var sp) && sp is System.Collections.IEnumerable seq && sp is not string
            ? seq.Cast<object?>().Select(x => x?.ToString()).Where(x => !string.IsNullOrEmpty(x)).Select(x => x!).ToList()
            : new List<string>();

        // 类型表（IsVowel/IsGlide）：从合并 dsdict.yaml 的 symbols 段读全语言符号类型。
        LoadSymbolTypes(Path.Combine(dir, "dsdict.yaml"));

        var lingPath = Path.Combine(dir, Get("linguistic"));
        Linguistic = load(lingPath);
        LinguisticHash = DiffSingerTensorCache.HashFile(lingPath);
        LinguisticUsesWordBoundary = Linguistic.InputMetadata.ContainsKey("word_div");
        foreach (var role in new[] { "dur", "variance", "pitch" })
            if (!string.IsNullOrEmpty(Get(role)))
            {
                var rolePath = Path.Combine(dir, Get(role));
                mModels[role] = load(rolePath);
                mModelHashes[role] = DiffSingerTensorCache.HashFile(rolePath);
            }
    }

    public bool HasModel(string role) => mModels.ContainsKey(role);
    public InferenceSession Model(string role) => mModels[role];

    // —— 符号查询 ——
    public bool IsVowel(string symbol)
        => AlwaysVowel.Contains(symbol) || (mSymbolTypes.TryGetValue(symbol, out var t) && t == "vowel");
    public bool IsGlide(string symbol) => mSymbolTypes.TryGetValue(symbol, out var t) && t == "glide";
    public bool IsKnownSymbol(string symbol) => AlwaysVowel.Contains(symbol) || mSymbolTypes.ContainsKey(symbol);

    public bool TryPhoneme(string symbol, out int id) => mPhonemes.TryGetValue(symbol, out id);
    public int PhonemeToken(string symbol)
        => mPhonemes.TryGetValue(symbol, out var id) ? id
            : throw new InvalidOperationException($"音素 \"{symbol}\" 不在 {Path.GetFileName(mDir)} 的音素表中");
    public long LangId(string lang) => mLanguages.TryGetValue(lang, out var id) ? id : 0;

    // —— G2P：按语言查 dsdict-{lang}.yaml 词条（grapheme→带前缀音素），exact 后小写回退 ——
    public string[] G2P(string lyric, string lang)
    {
        var entries = GetEntries(lang);
        var key = lyric.Trim();
        if (entries.TryGetValue(key, out var phs)) return phs;
        if (entries.TryGetValue(key.ToLowerInvariant(), out phs)) return phs;
        return Array.Empty<string>();
    }

    // 说话人逐元素嵌入（.emb = HiddenSize 个 float32 LE）：按声学说话人后缀（如 "Miku"）关联本预测器同后缀条目，
    // 无同后缀回退首个。按后缀缓存。
    public float[] GetEmbedding(string acousticSpeaker)
    {
        string suffix = DiffSingerDeclarations.Suffix(acousticSpeaker);
        lock (mLock)
        {
            if (mEmbCache.TryGetValue(suffix, out var cached))
                return cached;

            string? match = mSpeakers.FirstOrDefault(s => DiffSingerDeclarations.Suffix(s) == suffix) ?? mSpeakers.FirstOrDefault();
            var emb = new float[mHidden];
            if (match != null)
            {
                var bytes = File.ReadAllBytes(Path.Combine(mDir, match + ".emb"));
                for (int i = 0; i < mHidden; i++)
                    emb[i] = BitConverter.ToSingle(bytes, i * 4);
            }
            mEmbCache[suffix] = emb;
            return emb;
        }
    }

    // —— 词典加载 ——
    Dictionary<string, string[]> GetEntries(string lang)
    {
        lock (mLock)
        {
            if (mEntryCache.TryGetValue(lang, out var cached))
                return cached;

            var map = new Dictionary<string, string[]>(StringComparer.Ordinal);
            foreach (var file in new[] { $"dsdict-{lang}.yaml", $"dsdict-zh-{lang}.yaml", "dsdict.yaml" })
            {
                var path = Path.Combine(mDir, file);
                if (!File.Exists(path)) continue;
                var root = DeserializeDsDict(path);
                foreach (var e in root.entries)
                    if (!string.IsNullOrEmpty(e.grapheme))
                        map[e.grapheme] = e.phonemes.ToArray();
                break;
            }
            mEntryCache[lang] = map;
            return map;
        }
    }

    void LoadSymbolTypes(string dsdictPath)
    {
        if (!File.Exists(dsdictPath)) return;
        var root = DeserializeDsDict(dsdictPath);
        foreach (var s in root.symbols)
            if (!string.IsNullOrEmpty(s.symbol))
                mSymbolTypes[s.symbol] = s.type ?? string.Empty;
    }

    static DsDictFile DeserializeDsDict(string path)
        => new DeserializerBuilder().IgnoreUnmatchedProperties().Build()
            .Deserialize<DsDictFile>(File.ReadAllText(path)) ?? new DsDictFile();

    // 音素/语言表（JSON ⊂ YAML，值为 int id）。
    static Dictionary<string, int> LoadIntMap(string path)
        => new DeserializerBuilder().Build().Deserialize<Dictionary<string, int>>(File.ReadAllText(path))
           ?? new Dictionary<string, int>();

    // 会话经退役机制在推理锁内释放（杜绝与在飞 Run 并发释放，根治关闭 / 换设备时的 AccessViolation）。
    public void Dispose()
    {
        DiffSingerTensorCache.RetireAndDispose(new[] { Linguistic }.Concat(mModels.Values));
        mModels.Clear();
    }
}

// dsdict-{lang}.yaml / dsdict.yaml 结构：entries: -{grapheme, phonemes:[...]}；symbols: -{symbol, type}。
sealed class DsDictFile
{
    public List<DsDictEntry> entries { get; set; } = new();
    public List<DsDictSymbol> symbols { get; set; } = new();
}
sealed class DsDictEntry { public string grapheme { get; set; } = ""; public List<string> phonemes { get; set; } = new(); }
sealed class DsDictSymbol { public string symbol { get; set; } = ""; public string type { get; set; } = ""; }

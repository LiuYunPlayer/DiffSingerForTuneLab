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
    // 推理锁：DirectML EP 的 InferenceSession.Run() 非线程安全，串行化所有 Run 调用。
    readonly object mRunLock = new();

    public InferenceSession Linguistic { get; }
    public int HiddenSize => mHidden;
    // linguistic 是否吃 word_div/word_dur（dsdur/dsvariance 词边界；dspitch 用已知 ph_dur）。
    public bool LinguisticUsesWordBoundary { get; }

    // 线程安全的推理包装（DirectML EP 需要串行化 Run 调用）
    public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunLinguistic(List<NamedOnnxValue> inputs)
    {
        lock (mRunLock) return Linguistic.Run(inputs);
    }

    public IDisposableReadOnlyCollection<DisposableNamedOnnxValue> RunModel(string role, List<NamedOnnxValue> inputs)
    {
        lock (mRunLock) return mModels[role].Run(inputs);
    }

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

        Linguistic = load(Path.Combine(dir, Get("linguistic")));
        LinguisticUsesWordBoundary = Linguistic.InputMetadata.ContainsKey("word_div");
        foreach (var role in new[] { "dur", "variance", "pitch" })
            if (!string.IsNullOrEmpty(Get(role)))
                mModels[role] = load(Path.Combine(dir, Get(role)));
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

    // —— G2P：优先查语言特定词典（dsdict-{lang}.yaml），避免默认底库（dsdict.yaml 以 zh 为主）污染；再试 replacements；最后才兜底查合并词典。 ——
    public string[] G2P(string lyric, string lang)
    {
        var key = lyric.Trim();
        // 1. 语言特定词典（不含默认底库）
        var langEntries = GetLanguageSpecificEntries(lang);
        if (langEntries.TryGetValue(key, out var phs)) return phs;
        if (langEntries.TryGetValue(key.ToLowerInvariant(), out phs)) return phs;
        // 2. 替换规则（en/ko 等无 entries 的语种）
        var replaced = ApplyReplacements(lyric, lang);
        if (replaced.Length > 0) return replaced;
        // 3. 最后才查合并词典（含默认底库 dsdict.yaml，作为未知字素的最终兜底）
        var allEntries = GetEntries(lang);
        if (allEntries.TryGetValue(key, out phs)) return phs;
        if (allEntries.TryGetValue(key.ToLowerInvariant(), out phs)) return phs;
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

    // —— 替换规则（用于 EN/KO 等无 entries 仅 replacements 的语种）——
    readonly Dictionary<string, List<(string from, string to)>> mReplacements = new(StringComparer.Ordinal);

    void LoadReplacements(string lang)
    {
        if (mReplacements.ContainsKey(lang)) return;
        var list = new List<(string from, string to)>();
        foreach (var file in new[] { $"dsdict-{lang}.yaml", $"dsdict-zh-{lang}.yaml", "dsdict.yaml" })
        {
            var path = Path.Combine(mDir, file);
            if (!File.Exists(path)) continue;
            try
            {
                var yaml = new DeserializerBuilder().Build();
                var doc = yaml.Deserialize<Dictionary<string, object?>>(File.ReadAllText(path));
                if (doc != null && doc.TryGetValue("replacements", out var reps) && reps is List<object?> repList)
                {
                    foreach (var r in repList)
                    {
                        if (r is Dictionary<object, object?> repDict)
                        {
                            string? from = repDict.TryGetValue("from", out var fv) ? fv?.ToString() : null;
                            string? to = repDict.TryGetValue("to", out var tv) ? tv?.ToString() : null;
                            if (!string.IsNullOrEmpty(from) && !string.IsNullOrEmpty(to))
                                list.Add((from, to));
                        }
                    }
                }
            }
            catch { }
        }
        mReplacements[lang] = list;
    }

    // 用替换规则将歌词转为音素（按最长匹配优先）
    public string[] ApplyReplacements(string lyric, string lang)
    {
        LoadReplacements(lang);
        if (!mReplacements.TryGetValue(lang, out var reps) || reps.Count == 0)
            return Array.Empty<string>();

        var repsSorted = reps.OrderByDescending(r => r.from.Length).ToList();
        var result = new List<string>();
        string text = lyric.ToLowerInvariant();
        int pos = 0;
        while (pos < text.Length)
        {
            bool matched = false;
            foreach (var (from, to) in repsSorted)
            {
                if (pos + from.Length <= text.Length && text.Substring(pos, from.Length) == from)
                {
                    result.Add(to);
                    pos += from.Length;
                    matched = true;
                    break;
                }
            }
            if (!matched)
            {
                // 单个字符作为独立音素
                string ch = text[pos].ToString();
                result.Add(ch);
                pos++;
            }
        }
        return result.ToArray();
    }

    // —— 词典加载 ——
    // 策略：先加载 dsdict.yaml 作为默认底库，再叠加载入语种特定文件（后面覆盖前面）。
    // 若 entries 为空且 replacements 存在，留空返回（上层调用 ApplyReplacements）。
    Dictionary<string, string[]> GetEntries(string lang)
    {
        lock (mLock)
        {
            if (mEntryCache.TryGetValue(lang, out var cached))
                return cached;

            var map = new Dictionary<string, string[]>(StringComparer.Ordinal);

            // 1. 加载默认底库 dsdict.yaml（总是存在）
            var defaultPath = Path.Combine(mDir, "dsdict.yaml");
            if (File.Exists(defaultPath))
            {
                var root = DeserializeDsDict(defaultPath);
                foreach (var e in root.entries)
                    if (!string.IsNullOrEmpty(e.grapheme))
                        map[e.grapheme] = e.phonemes.ToArray();
            }

            // 2. 叠加载入语种特定文件（若存在则覆盖/补充）
            foreach (var file in new[] { $"dsdict-{lang}.yaml", $"dsdict-zh-{lang}.yaml" })
            {
                var path = Path.Combine(mDir, file);
                if (!File.Exists(path)) continue;
                var root = DeserializeDsDict(path);
                foreach (var e in root.entries)
                    if (!string.IsNullOrEmpty(e.grapheme))
                        map[e.grapheme] = e.phonemes.ToArray();
            }

            mEntryCache[lang] = map;
            return map;
        }
    }

    // 仅加载语言特定词典（不含默认底库 dsdict.yaml），用于 G2P 的优先查表——避免 zh 底库污染其他语言的译音。
    Dictionary<string, string[]> GetLanguageSpecificEntries(string lang)
    {
        var map = new Dictionary<string, string[]>(StringComparer.Ordinal);
        foreach (var file in new[] { $"dsdict-{lang}.yaml", $"dsdict-zh-{lang}.yaml" })
        {
            var path = Path.Combine(mDir, file);
            if (!File.Exists(path)) continue;
            var root = DeserializeDsDict(path);
            foreach (var e in root.entries)
                if (!string.IsNullOrEmpty(e.grapheme))
                    map[e.grapheme] = e.phonemes.ToArray();
        }
        return map;
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

    public void Dispose()
    {
        Linguistic.Dispose();
        foreach (var m in mModels.Values)
            m.Dispose();
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

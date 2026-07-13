using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using YamlDotNet.Serialization;

using DiffSingerForTuneLab.G2p;

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
    readonly Dictionary<string, IModelSession> mModels = new(StringComparer.Ordinal);
    readonly Dictionary<string, float[]> mEmbCache = new(StringComparer.Ordinal);
    readonly Dictionary<string, IG2p?> mG2pChains = new(StringComparer.Ordinal);  // lang → G2P 兜底链（词典 ⊕ 算法 remap），懒构建缓存
    readonly Dictionary<string, string> mSymbolTypes = new(StringComparer.Ordinal);  // symbol → type（合并 dsdict）
    readonly object mLock = new();

    public IModelSession Linguistic { get; }
    public int HiddenSize => mHidden;
    // linguistic 是否吃 word_div/word_dur（dsdur/dsvariance 词边界；dspitch 用已知 ph_dur）。
    public bool LinguisticUsesWordBoundary { get; }

    // 张量缓存 identifier（模型 .onnx 文件内容哈希，加载时算一次）：linguistic 与各 role 模型。
    public ulong LinguisticHash { get; }
    readonly Dictionary<string, ulong> mModelHashes = new(StringComparer.Ordinal);
    public ulong ModelHash(string role) => mModelHashes.TryGetValue(role, out var h) ? h : 0;

    // onlyRole 非空 ⇒ 只加载该职责 role（对齐 OpenUtau：各预测器只用自己那一个 role，忽略 dsconfig 里其余 role 字段，
    //   哪怕它们指向不存在的文件）；为空（未知子目录）⇒ 保守加载全部声明的 role，不改老行为。
    public DiffSingerPredictor(string dir, Func<string, IModelSession> load, string? onlyRole = null)
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
        LinguisticUsesWordBoundary = Linguistic.HasInput("word_div");
        var roles = string.IsNullOrEmpty(onlyRole) ? new[] { "dur", "variance", "pitch" } : new[] { onlyRole };
        foreach (var role in roles)
            if (!string.IsNullOrEmpty(Get(role)))
            {
                var rolePath = Path.Combine(dir, Get(role));
                mModels[role] = load(rolePath);
                mModelHashes[role] = DiffSingerTensorCache.HashFile(rolePath);
            }
    }

    public bool HasModel(string role) => mModels.ContainsKey(role);
    public IModelSession Model(string role) => mModels[role];

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

    // P1-a 音素混合（linguistic 编码器侧）：从主 tokens 克隆出次要 tokens_b + 逐 token blend，
    //   只覆盖设了混合、且本预测器词表可解析目标符号的音素槽；其余槽 tokens_b=primary、blend=0（等价无混合）。
    //   primaryTokens 布局 = [SP, ...phones..., SP]（含 head/tail padding），phones[i] 落 primaryTokens[i+1]。
    //   目标符号按本预测器自己的词表解析（acoustic 与预测器 id 空间不共享，须按符号字符串各查各表；查不到即优雅降级不混）。
    public static (long[] TokensB, float[] Blend) BuildPhonemeMix(
        DiffSingerPredictor v, IReadOnlyList<PhonemeSpan> phones, long[] primaryTokens)
    {
        var tokensB = (long[])primaryTokens.Clone();
        var blend = new float[primaryTokens.Length];
        for (int i = 0; i < phones.Count; i++)
        {
            var p = phones[i];
            if (p.MixRatio > 0 && !string.IsNullOrEmpty(p.MixSymbol) && v.TryPhoneme(p.MixSymbol, out var mid))
            {
                tokensB[i + 1] = mid;
                blend[i + 1] = (float)Math.Clamp(p.MixRatio, 0, 1);
            }
        }
        return (tokensB, blend);
    }

    // —— G2P：按语言懒构建兜底链（词典 → 算法引擎 remap），exact 后小写回退（忠实 OpenUtau GetSymbols）——
    public string[] G2P(string lyric, string lang)
    {
        var chain = ResolveChain(lang);
        if (chain == null) return Array.Empty<string>();
        var key = lyric.Trim();
        return chain.Query(key) ?? chain.Query(key.ToLowerInvariant()) ?? Array.Empty<string>();
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

    // —— G2P 兜底链构建（懒，按 lang 缓存）——
    //   链 = G2pFallbacks([ 声库词典层, 算法引擎 remap 层 ])：词典命中作者精修词优先，OOV 落算法引擎。
    //   tunelab.yaml 的 g2p 覆盖经 langEngineOverride 传入（缺省 null = 走内置默认绑定）。
    IG2p? ResolveChain(string lang)
    {
        lock (mLock)
        {
            if (mG2pChains.TryGetValue(lang, out var cached))
                return cached;
            var chain = BuildChain(lang);
            mG2pChains[lang] = chain;
            return chain;
        }
    }

    IG2p? BuildChain(string lang)
    {
        var dictData = LoadDsDictFile(lang);
        var layers = new List<IG2p>();

        // 1) 词典层：dsdict 的 symbols(类型) + entries(grapheme→声库音素) + SP/AP。先声明符号再加词条（否则词条里未声明符号被丢）。
        var builder = G2pDictionary.NewBuilder();
        foreach (var s in dictData.symbols)
            if (!string.IsNullOrEmpty(s.symbol))
                builder.AddSymbol(s.symbol.Trim(), s.type ?? string.Empty);
        builder.AddSymbol("SP", true);
        builder.AddSymbol("AP", true);
        foreach (var e in dictData.entries)
            if (!string.IsNullOrEmpty(e.grapheme))
                builder.AddEntry(e.grapheme, e.phonemes);
        layers.Add(builder.Build());

        // 类型表 augment：把 dsdict 声明的符号类型（归一化为 vowel/glide/consonant）补进预测器类型表，
        //   供 phonemizer 的 IsVowel/IsGlide 分组使用（现有 LoadSymbolTypes 只读合并 dsdict.yaml，多语言库的 per-lang 符号靠这里补）。
        foreach (var s in dictData.symbols)
            if (!string.IsNullOrEmpty(s.symbol))
                AddSymbolTypeIfAbsent(s.symbol.Trim(), NormalizeType(s.type));

        // 2) 算法引擎 + remap 层（按语言绑定；无绑定 = 纯词典）。
        var desc = G2pEngines.ForLanguage(lang);
        if (desc != null)
        {
            var engine = desc.Create();
            var replacements = dictData.replacementsDict();           // dsdict 显式覆盖（最高优先）
            var phonemeSymbols = new Dictionary<string, bool>(StringComparer.Ordinal); // 声库符号 → isVowel
            var glide = new HashSet<string>(StringComparer.Ordinal);
            foreach (var ph in desc.Vowels.Concat(desc.Consonants))
            {
                // 规范符号 ph → 声库符号 resolved：dsdict 覆盖 > 前缀版存在 > 裸版存在 > 默认前缀。
                //   这条 bare/prefix 回退是「让单语(裸符号)与多语言(前缀)库都能用」的关键，比 OpenUtau 无条件前缀更稳。
                string resolved;
                if (replacements.TryGetValue(ph, out var explicitTo))
                    resolved = explicitTo;
                else
                {
                    resolved = ResolveRemapTarget(ph, lang);
                    replacements[ph] = resolved;                      // 写回供 G2pRemapper.Query 改写
                }
                bool isVowel = engine.IsVowel(ph);
                bool isGlide = engine.IsGlide(ph);
                phonemeSymbols[resolved] = isVowel;
                if (isGlide) glide.Add(resolved);
                AddSymbolTypeIfAbsent(resolved, isVowel ? "vowel" : isGlide ? "glide" : "consonant");
            }
            layers.Add(new G2pRemapper(engine, phonemeSymbols, replacements, glide));
        }

        return layers.Count > 0 ? new G2pFallbacks(layers.ToArray()) : null;
    }

    // 规范符号 → 声库音素表里实际存在的符号：优先 <lang>/<ph>（多语言库），其次裸 <ph>（单语库），都无则默认带前缀。
    string ResolveRemapTarget(string ph, string lang)
    {
        if (!string.IsNullOrEmpty(lang) && mPhonemes.ContainsKey(lang + "/" + ph))
            return lang + "/" + ph;
        if (mPhonemes.ContainsKey(ph))
            return ph;
        return string.IsNullOrEmpty(lang) ? ph : lang + "/" + ph;
    }

    void AddSymbolTypeIfAbsent(string symbol, string type)
    {
        if (!mSymbolTypes.ContainsKey(symbol))
            mSymbolTypes[symbol] = type;
    }

    // 归一化到预测器类型表词表（IsVowel 认 "vowel"、IsGlide 认 "glide"）：OpenUtau/包用 semivowel/liquid 表滑音。
    static string NormalizeType(string? type) => type switch
    {
        "vowel" => "vowel",
        "semivowel" or "liquid" or "glide" => "glide",
        _ => "consonant",
    };

    // 按 dsdict-{lang}.yaml → dsdict-zh-{lang}.yaml → dsdict.yaml 顺序取首个存在的词典文件。
    DsDictFile LoadDsDictFile(string lang)
    {
        foreach (var file in new[] { $"dsdict-{lang}.yaml", $"dsdict-zh-{lang}.yaml", "dsdict.yaml" })
        {
            var path = Path.Combine(mDir, file);
            if (File.Exists(path)) return DeserializeDsDict(path);
        }
        return new DsDictFile();
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

    // 会话自身在设备级锁内退役并释放（杜绝与在飞 Run 并发释放，根治关闭 / 换设备时的 AccessViolation）。
    public void Dispose()
    {
        Linguistic.Dispose();
        foreach (var model in mModels.Values)
            model.Dispose();
        mModels.Clear();
    }
}

// dsdict-{lang}.yaml / dsdict.yaml 结构：entries: -{grapheme, phonemes:[...]}；symbols: -{symbol, type}；
//   replacements: -{from, to}（规范符号→声库符号的逐项覆盖，OpenUtau DiffSingerG2pDictionaryData 同款；读、不改）。
sealed class DsDictFile
{
    public List<DsDictEntry> entries { get; set; } = new();
    public List<DsDictSymbol> symbols { get; set; } = new();
    public List<DsDictReplacement> replacements { get; set; } = new();

    public Dictionary<string, string> replacementsDict()
    {
        var dict = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var r in replacements)
            if (!string.IsNullOrEmpty(r.from))
                dict[r.from] = r.to;
        return dict;
    }
}
sealed class DsDictEntry { public string grapheme { get; set; } = ""; public List<string> phonemes { get; set; } = new(); }
sealed class DsDictSymbol { public string symbol { get; set; } = ""; public string type { get; set; } = ""; }
sealed class DsDictReplacement { public string from { get; set; } = ""; public string to { get; set; } = ""; }

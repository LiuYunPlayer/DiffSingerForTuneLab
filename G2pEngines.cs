using System;
using System.Collections.Generic;
using System.Linq;
using DiffSingerForTuneLab.G2p;
using OpenUtau.Api;

namespace DiffSingerForTuneLab;

// 语言 id → 算法 G2P 引擎的注册与默认绑定。这是「不硬编码语言」的落点，也替代了 OpenUtau 的「用户手选 phonemizer 类」：
//   在 OpenUtau 里选哪个 phonemizer 类就等于选了引擎；本插件语言是 note/part 数据，故用一张表把语言映到引擎描述符。
//   · 内置默认绑定覆盖上游全部 DiffSinger G2P 音素器语言（en/ko/fr/de/es/it/pt/ru/fil），multi-dict 声库零声明可用；
//   · 一语言多方案（de 经典 vs Marzipan）按「声库带哪个词典变体」消歧——这正是 OpenUtau 用户选类时参考的信号；
//   · tunelab.yaml 的 languages.expose[].g2p 可覆盖（含显式 "dictionary-only" 抑制默认）——见后续接入。
// 引擎本体（资源包）已捆绑进插件；新增一门内置语言 = 此表加一行 + 带一个包。
// 上游未搬的两个 DiffSinger 侧引擎（有意为之）：ArpabetG2p（旧版英语，已被 arpabet-plus 取代、词典名同为
//   dsdict-en.yaml 无从消歧）、RuleBasedFilipinoG2p（与神经包版同词典名、上游靠手选区分，等 g2p 覆盖接入后再补）。
static class G2pEngines
{
    // 引擎描述符：构造工厂 + 该引擎的规范元音/辅音清单 + 专属词典名（可空）。
    //   清单用于给「remap 后的声库符号」定 vowel/consonant（glide 直接问引擎实例 IsGlide，源自包 phones.txt）。
    //   取自 OpenUtau 对应 phonemizer 的 GetBaseG2pVowels()/GetBaseG2pConsonants()/GetDictionaryName()——
    //   专属词典名如 dsdict-fr-millefeuille.yaml（该方案的 replacements 表住在这个变体里），缺省走 dsdict-{lang}.yaml 链。
    public sealed record EngineDescriptor(string Id, Func<IG2p> Create, string[] Vowels, string[] Consonants,
        string? DictionaryName = null);

    // —— 各引擎规范元音/辅音清单（照抄上游对应 phonemizer，含 uy/vf/cl 归辅音这类原版口径）——
    static readonly string[] ArpabetVowels =
        { "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw" };
    static readonly string[] ArpabetConsonants =
        { "b", "ch", "d", "dh", "dr", "dx", "f", "g", "hh", "jh", "k", "l", "m", "n", "ng", "p", "q", "r", "s", "sh", "t", "th", "tr", "v", "w", "y", "z", "zh" };
    static readonly string[] KoreanVowels =
        { "a", "e", "eo", "eu", "i", "o", "u", "w", "y" };
    static readonly string[] KoreanConsonants =
        { "K", "L", "M", "N", "NG", "P", "T", "b", "ch", "d", "g", "h", "j", "jj", "k", "kk", "m", "n", "p", "pp", "r", "s", "ss", "t", "tt" };
    static readonly string[] FrenchMillefeuilleVowels =
        { "ah", "eh", "ae", "ee", "oe", "ih", "oh", "oo", "ou", "uh", "en", "in", "on" };
    static readonly string[] FrenchMillefeuilleConsonants =
        { "y", "w", "f", "k", "p", "s", "sh", "t", "h", "b", "d", "g", "l", "m", "n", "r", "v", "z", "j", "ng", "q", "uy", "vf", "cl" };
    static readonly string[] GermanVowels =
        { "aa", "ae", "ah", "ao", "aw", "ax", "ay", "ee", "eh", "er", "ex", "ih", "iy", "oe", "ohh", "ooh", "oy", "ue", "uh", "uw", "yy" };
    static readonly string[] GermanConsonants =
        { "b", "cc", "ch", "d", "dh", "f", "g", "hh", "jh", "k", "l", "m", "n", "ng", "p", "pf", "q", "r", "rr", "s", "sh", "t", "th", "ts", "v", "w", "x", "y", "z", "zh" };
    static readonly string[] GermanMarzipanVowels =
        { "a", "er", "eh", "e", "ih", "i", "uh", "u", "oh", "o", "ueh", "ue", "oeh", "oe", "ex", "ei", "au", "eu" };
    static readonly string[] GermanMarzipanConsonants =
        { "j", "p", "t", "k", "f", "s", "sh", "ch", "xh", "h", "pf", "ts", "tsh", "th", "m", "n", "ng", "b", "d", "g", "v", "z", "l", "r", "dsh", "zh", "rh", "rr", "rx", "dh", "q", "vf", "cl" };
    static readonly string[] SpanishVowels =
        { "a", "e", "i", "o", "u" };
    static readonly string[] SpanishConsonants =
        { "b", "B", "ch", "d", "D", "f", "g", "G", "gn", "I", "k", "l", "ll", "m", "n", "p", "r", "rr", "s", "t", "U", "w", "x", "y", "Y", "z" };
    static readonly string[] ItalianVowels =
        { "a", "e", "EE", "i", "o", "OO", "u" };
    static readonly string[] ItalianConsonants =
        { "b", "d", "dz", "dZZ", "f", "g", "j", "JJ", "k", "l", "LL", "m", "n", "nf", "ng", "p", "r", "s", "SS", "t", "ts", "tSS", "v", "w", "z" };
    static readonly string[] PortugueseVowels =
        { "E", "O", "a", "a~", "e", "e~", "i", "i~", "o", "o~", "u", "u~" };
    static readonly string[] PortugueseConsonants =
        { "J", "L", "R", "S", "X", "Z", "b", "d", "dZ", "f", "g", "j", "j~", "k", "l", "m", "n", "p", "r", "s", "t", "tS", "v", "w", "w~", "z" };
    static readonly string[] RussianVowels =
        { "a", "aa", "ay", "ee", "i", "ii", "ja", "je", "jo", "ju", "oo", "u", "uj", "uu", "y", "yy" };
    static readonly string[] RussianConsonants =
        { "b", "bb", "c", "ch", "d", "dd", "f", "ff", "g", "gg", "h", "hh", "j", "k", "kk", "l", "ll", "m", "mm", "n", "nn", "p", "pp", "r", "rr", "s", "sch", "sh", "ss", "t", "tt", "v", "vv", "z", "zh", "zz" };
    static readonly string[] FilipinoVowels =
        { "a", "e", "i", "o", "u" };
    static readonly string[] FilipinoConsonants =
        { "q", "b", "d", "dy", "f", "g", "H", "hh", "j", "k", "l", "m", "n", "ng", "ny", "p", "dx", "s", "sy", "t", "th", "ch", "v", "w", "z" };

    static readonly Dictionary<string, EngineDescriptor> Engines = new(StringComparer.OrdinalIgnoreCase)
    {
        ["arpabet-plus"] = new("arpabet-plus", () => new ArpabetPlusG2p(), ArpabetVowels, ArpabetConsonants,
            "dsdict-en.yaml"),
        ["korean"] = new("korean", () => new KoreanG2p(), KoreanVowels, KoreanConsonants,
            "dsdict-ko.yaml"),
        ["french-millefeuille"] = new("french-millefeuille", () => new FrenchMillefeuilleG2p(),
            FrenchMillefeuilleVowels, FrenchMillefeuilleConsonants, "dsdict-fr-millefeuille.yaml"),
        ["german"] = new("german", () => new GermanG2p(), GermanVowels, GermanConsonants,
            "dsdict-de.yaml"),
        ["german-marzipan"] = new("german-marzipan", () => new GermanMarzipanG2p(),
            GermanMarzipanVowels, GermanMarzipanConsonants, "dsdict-de-marzipan.yaml"),
        ["spanish"] = new("spanish", () => new SpanishG2p(), SpanishVowels, SpanishConsonants,
            "dsdict-es.yaml"),
        ["italian"] = new("italian", () => new ItalianG2p(), ItalianVowels, ItalianConsonants,
            "dsdict-it.yaml"),
        ["portuguese"] = new("portuguese", () => new PortugueseG2p(), PortugueseVowels, PortugueseConsonants,
            "dsdict-pt.yaml"),
        ["russian"] = new("russian", () => new RussianG2p(), RussianVowels, RussianConsonants,
            "dsdict-ru.yaml"),
        ["filipino"] = new("filipino", () => new FilipinoG2p(), FilipinoVowels, FilipinoConsonants,
            "dsdict-fil.yaml"),
        // "dictionary-only"：无算法层（仅词典），以 null 描述符表示。
    };

    // 语言 id → engineId 候选（有序）。多候选时按「声库带哪个词典变体」消歧（首个词典存在者胜），
    //   都不带则取首位；单候选无条件生效（词典缺失走 dsdict.yaml 兜底，与旧行为一致）。
    static readonly Dictionary<string, string[]> DefaultBindings = new(StringComparer.OrdinalIgnoreCase)
    {
        ["en"] = new[] { "arpabet-plus" },
        ["ko"] = new[] { "korean" },
        ["fr"] = new[] { "french-millefeuille" },
        ["de"] = new[] { "german-marzipan", "german" },   // Marzipan（新方案）优先——近年多语言库按其调校
        ["es"] = new[] { "spanish" },
        ["it"] = new[] { "italian" },
        ["pt"] = new[] { "portuguese" },
        ["ru"] = new[] { "russian" },
        ["fil"] = new[] { "filipino" },
    };

    public static bool IsKnownEngine(string? engineId)
        => !string.IsNullOrEmpty(engineId) && Engines.ContainsKey(engineId);

    public static EngineDescriptor? ResolveEngine(string? engineId)
        => !string.IsNullOrEmpty(engineId) && Engines.TryGetValue(engineId, out var d) ? d : null;

    // 给定语言 id + 可选覆盖（来自 tunelab.yaml），定到引擎描述符。
    //   overrideEngineId == null：无覆盖，走默认绑定（hasDictionary 供多候选按词典变体消歧，null = 恒取首位）；
    //   overrideEngineId == "dictionary-only" 或未知名：返回 null（纯词典）。
    public static EngineDescriptor? ForLanguage(string lang, string? overrideEngineId = null,
        Func<string, bool>? hasDictionary = null)
    {
        if (overrideEngineId != null)
            return ResolveEngine(overrideEngineId);
        if (string.IsNullOrEmpty(lang) || !DefaultBindings.TryGetValue(lang, out var candidates))
            return null;
        if (candidates.Length > 1 && hasDictionary != null)
            foreach (var id in candidates)
                if (Engines[id].DictionaryName is { } dict && hasDictionary(dict))
                    return Engines[id];
        return Engines[candidates[0]];
    }
}

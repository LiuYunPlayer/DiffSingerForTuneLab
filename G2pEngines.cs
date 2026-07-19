using System;
using System.Collections.Generic;
using DiffSingerForTuneLab.G2p;
using OpenUtau.Api;

namespace DiffSingerForTuneLab;

// 语言 id → 算法 G2P 引擎的注册与默认绑定。这是「不硬编码 en/ko」的落点，也替代了 OpenUtau 的「用户手选 phonemizer 类」：
//   在 OpenUtau 里选哪个 phonemizer 类就等于选了引擎；本插件语言是 note/part 数据，故用一张表把语言映到引擎描述符。
//   · 内置默认绑定（en→arpabet-plus、ko→korean）让现状 multi-dict 英韩声库零声明可用；
//   · tunelab.yaml 的 languages.expose[].g2p 可覆盖（含显式 "dictionary-only" 抑制默认）——见后续接入。
// 引擎本体（资源包）已捆绑进插件；新增一门内置语言 = 此表加一行 + 带一个包。
static class G2pEngines
{
    // 引擎描述符：构造工厂 + 该引擎的规范元音/辅音清单 + 专属词典名（可空）。
    //   清单用于给「remap 后的声库符号」定 vowel/consonant（glide 直接问引擎实例 IsGlide，源自包 phones.txt）。
    //   取自 OpenUtau 对应 phonemizer 的 GetBaseG2pVowels()/GetBaseG2pConsonants()/GetDictionaryName()——
    //   专属词典名如 dsdict-fr-millefeuille.yaml（该方案的 replacements 表住在这个变体里），缺省走 dsdict-{lang}.yaml 链。
    public sealed record EngineDescriptor(string Id, Func<IG2p> Create, string[] Vowels, string[] Consonants,
        string? DictionaryName = null);

    static readonly string[] ArpabetVowels =
        { "aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw" };
    static readonly string[] ArpabetConsonants =
        { "b", "ch", "d", "dh", "dr", "dx", "f", "g", "hh", "jh", "k", "l", "m", "n", "ng", "p", "q", "r", "s", "sh", "t", "th", "tr", "v", "w", "y", "z", "zh" };
    static readonly string[] KoreanVowels =
        { "a", "e", "eo", "eu", "i", "o", "u", "w", "y" };
    static readonly string[] KoreanConsonants =
        { "K", "L", "M", "N", "NG", "P", "T", "b", "ch", "d", "g", "h", "j", "jj", "k", "kk", "m", "n", "p", "pp", "r", "s", "ss", "t", "tt" };
    // 法语 Millefeuille（UFR 方案，已并入 OpenUtau 内置）：清单取自 DiffSingerFrenchMillfeuillePhonemizer
    //   （uy/vf/cl 归辅音是原版口径，照抄）。
    static readonly string[] FrenchMillefeuilleVowels =
        { "ah", "eh", "ae", "ee", "oe", "ih", "oh", "oo", "ou", "uh", "en", "in", "on" };
    static readonly string[] FrenchMillefeuilleConsonants =
        { "y", "w", "f", "k", "p", "s", "sh", "t", "h", "b", "d", "g", "l", "m", "n", "r", "v", "z", "j", "ng", "q", "uy", "vf", "cl" };

    static readonly Dictionary<string, EngineDescriptor> Engines = new(StringComparer.OrdinalIgnoreCase)
    {
        ["arpabet-plus"] = new("arpabet-plus", () => new ArpabetPlusG2p(), ArpabetVowels, ArpabetConsonants),
        ["korean"] = new("korean", () => new KoreanG2p(), KoreanVowels, KoreanConsonants),
        ["french-millefeuille"] = new("french-millefeuille", () => new FrenchMillefeuilleG2p(),
            FrenchMillefeuilleVowels, FrenchMillefeuilleConsonants, "dsdict-fr-millefeuille.yaml"),
        // "dictionary-only"：无算法层（仅词典），以 null 描述符表示。
    };

    // 语言 id → engineId 默认绑定。让 multi-dict 英韩法库零声明可用（tunelab.yaml 缺失也成立）。
    static readonly Dictionary<string, string> DefaultBindings = new(StringComparer.OrdinalIgnoreCase)
    {
        ["en"] = "arpabet-plus",
        ["ko"] = "korean",
        ["fr"] = "french-millefeuille",
    };

    public static bool IsKnownEngine(string? engineId)
        => !string.IsNullOrEmpty(engineId) && Engines.ContainsKey(engineId);

    public static EngineDescriptor? ResolveEngine(string? engineId)
        => !string.IsNullOrEmpty(engineId) && Engines.TryGetValue(engineId, out var d) ? d : null;

    // 给定语言 id + 可选覆盖（来自 tunelab.yaml），定到引擎描述符。
    //   overrideEngineId == null：无覆盖，走默认绑定；
    //   overrideEngineId == "dictionary-only" 或未知名：返回 null（纯词典）。
    public static EngineDescriptor? ForLanguage(string lang, string? overrideEngineId = null)
    {
        if (overrideEngineId != null)
            return ResolveEngine(overrideEngineId);
        if (!string.IsNullOrEmpty(lang) && DefaultBindings.TryGetValue(lang, out var id))
            return Engines[id];
        return null;
    }
}

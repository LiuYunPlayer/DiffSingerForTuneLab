using System.Collections.Generic;
using System.Linq;
using OpenUtau.Api;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/Api/G2pRemapper.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 重映射层：把被包裹引擎(mapped)产出的「规范符号」按 replacements 改写成「本声库符号」。
//   算法引擎(arpabet/korean)永远吐规范符号(aa/ae… 或 a/e/K…)，声库各有自己的音素方案——
//   replacements = 规范→声库（默认 <langcode>/<符号> 前缀，dsdict 的 replacements 块可逐符号覆盖；
//   单语裸符号库则映射为恒等/裸符号）。符号类型表(phonemeSymbols/glideSymbols)按「声库符号」给定。
public sealed class G2pRemapper : IG2p
{
    readonly IG2p mapped;
    readonly Dictionary<string, bool> phonemeSymbols; // (声库符号, isVowel)
    readonly HashSet<string> glideSymbols;
    readonly Dictionary<string, string> replacements;

    public G2pRemapper(IG2p mapped,
        Dictionary<string, bool> phonemeSymbols,
        Dictionary<string, string> replacements,
        HashSet<string>? glideSymbols = null)
    {
        this.mapped = mapped;
        this.phonemeSymbols = phonemeSymbols;
        this.replacements = replacements;
        this.glideSymbols = glideSymbols ?? new HashSet<string>();
    }

    public bool IsValidSymbol(string symbol) => phonemeSymbols.ContainsKey(symbol);
    public bool IsVowel(string symbol) => phonemeSymbols.TryGetValue(symbol, out var isVowel) && isVowel;
    public bool IsGlide(string symbol) => glideSymbols.Contains(symbol);

    public string[]? Query(string grapheme)
    {
        var phonemes = mapped.Query(grapheme);
        if (phonemes == null)
            return null;
        phonemes = (string[])phonemes.Clone();
        for (int i = 0; i < phonemes.Length; ++i)
            if (replacements.TryGetValue(phonemes[i], out var replacement))
                phonemes[i] = replacement;
        return phonemes;
    }

    public string[]? UnpackHint(string hint, char separator = ' ')
        => hint.Split(separator).Where(s => phonemeSymbols.ContainsKey(s)).ToArray();
}

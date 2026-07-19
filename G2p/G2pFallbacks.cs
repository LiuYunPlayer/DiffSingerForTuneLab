using OpenUtau.Api;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/Api/G2pFallbacks.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 链式兜底：按顺序问每一层，第一个认领该符号 / 查出该词的层胜出。典型链 = [声库词典, 算法引擎(remap 后)]：
//   词典命中作者精修词优先，OOV 落算法引擎。符号类型(IsVowel/IsGlide)归属于「第一个声称该符号有效」的层。
public sealed class G2pFallbacks : IG2p
{
    readonly IG2p[] dictionaries;

    public G2pFallbacks(IG2p[] dictionaries)
    {
        this.dictionaries = dictionaries;
    }

    public bool IsValidSymbol(string symbol)
    {
        foreach (var dict in dictionaries)
            if (dict.IsValidSymbol(symbol))
                return true;
        return false;
    }

    public bool IsVowel(string symbol)
    {
        foreach (var dict in dictionaries)
            if (dict.IsValidSymbol(symbol))
                return dict.IsVowel(symbol);
        return false;
    }

    public bool IsGlide(string symbol)
    {
        foreach (var dict in dictionaries)
            if (dict.IsValidSymbol(symbol))
                return dict.IsGlide(symbol);
        return false;
    }

    public string[]? Query(string grapheme)
    {
        foreach (var dict in dictionaries)
        {
            var result = dict.Query(grapheme);
            if (result != null)
                return result;
        }
        return null;
    }

    public string[]? UnpackHint(string hint, char separator = ' ')
    {
        foreach (var dict in dictionaries)
        {
            var result = dict.UnpackHint(hint, separator);
            if (result != null)
                return result;
        }
        return null;
    }
}

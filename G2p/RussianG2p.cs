using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Api;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/G2p/RussianG2p.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 俄语算法 G2P：西里尔字母词表 + 神经 OOV（软硬辅音成对，如 b/bb）。资源包改为从插件嵌入资源加载。
public sealed class RussianG2p : G2pPack
{
    static readonly string[] graphemes =
    {
        "", "", "", "", "-", "а", "б", "в", "г", "д", "е", "ж", "з",
        "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у",
        "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я", "ё",
    };

    static readonly string[] phonemes =
    {
        "", "", "", "", "a", "aa", "ay", "b", "bb", "c", "ch",
        "d", "dd", "ee", "f", "ff", "g", "gg", "h", "hh", "i", "ii",
        "j", "ja", "je", "jo", "ju", "k", "kk", "l", "ll", "m", "mm",
        "n", "nn", "oo", "p", "pp", "r", "rr", "s", "sch", "sh", "ss",
        "t", "tt", "u", "uj", "uu", "v", "vv", "y", "yy", "z", "zh", "zz",
    };

    static readonly object lockObj = new();
    static Dictionary<string, int>? graphemeIndexes;
    static IG2p? dict;
    static InferenceSession? session;
    static readonly Dictionary<string, string[]> predCache = new();

    public RussianG2p()
    {
        lock (lockObj)
        {
            if (graphemeIndexes == null)
            {
                graphemeIndexes = graphemes
                    .Skip(4)
                    .Select((g, i) => Tuple.Create(g, i))
                    .ToDictionary(t => t.Item1, t => t.Item2 + 4);
                var tuple = LoadPack(EmbeddedG2pPacks.Load("g2p-ru.zip"));
                dict = tuple.Item1;
                session = tuple.Item2;
            }
        }
        GraphemeIndexes = graphemeIndexes;
        Phonemes = phonemes;
        Dict = dict!;
        Session = session;
        PredCache = predCache;
    }
}

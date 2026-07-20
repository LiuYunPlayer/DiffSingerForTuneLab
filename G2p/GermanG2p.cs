using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Api;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/G2p/GermanG2p.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 德语（经典方案）算法 G2P：词表 + 神经 OOV。与 Marzipan 方案并存，绑定时按声库词典变体消歧（见 G2pEngines）。
//   资源包改为从插件嵌入资源加载。静态包共享、线程安全初始化。
public sealed class GermanG2p : G2pPack
{
    static readonly string[] graphemes =
    {
        "", "", "", "", "2", "3", "/", "a", "b", "c", "d", "e",
        "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "ä", "ë", "ö", "ü", "ß", "ê", "î", "ô", "á", "é",
        "í", "ú", "à", "è", "ù", "ę",
    };

    static readonly string[] phonemes =
    {
        "", "", "", "", "aa", "ae", "ah", "ao", "aw", "ax", "ay",
        "b", "cc", "ch", "d", "dh", "ee", "eh", "er", "ex", "f",
        "g", "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng",
        "oe", "ohh", "ooh", "oy", "p", "pf", "q", "r", "rr", "s",
        "sh", "t", "th", "ts", "ue", "uh", "uw", "v", "w", "x",
        "y", "yy", "z", "zh",
    };

    static readonly object lockObj = new();
    static Dictionary<string, int>? graphemeIndexes;
    static IG2p? dict;
    static InferenceSession? session;
    static readonly Dictionary<string, string[]> predCache = new();

    public GermanG2p()
    {
        lock (lockObj)
        {
            if (graphemeIndexes == null)
            {
                graphemeIndexes = graphemes
                    .Skip(4)
                    .Select((g, i) => Tuple.Create(g, i))
                    .ToDictionary(t => t.Item1, t => t.Item2 + 4);
                var tuple = LoadPack(EmbeddedG2pPacks.Load("g2p-de.zip"));
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

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Api;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/G2p/GermanMarzipanG2p.cs（UFR Marzipan 德语方案，已并入上游内置）。
// 见仓库根 THIRD-PARTY-NOTICES.md。词典名 dsdict-de-marzipan.yaml；与经典德语方案并存，
//   绑定时按声库词典变体消歧（见 G2pEngines）。资源包改为从插件嵌入资源加载。静态包共享、线程安全初始化。
public sealed class GermanMarzipanG2p : G2pPack
{
    static readonly string[] graphemes =
    {
        "", "", "", "", "a", "b", "c", "d", "e", "f", "g", "h",
        "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
        "t", "u", "v", "w", "x", "y", "z", "ä", "ë", "ö", "ü", "ß",
    };

    static readonly string[] phonemes =
    {
        "", "", "", "", "a", "er", "eh", "e", "ih", "i", "uh",
        "u", "oh", "o", "ueh", "ue", "oeh", "oe", "ex", "ei", "au",
        "eu", "w", "j", "p", "t", "k", "f", "s", "sh", "ch",
        "xh", "h", "pf", "ts", "tsh", "th", "m", "n", "ng", "b",
        "d", "g", "v", "z", "l", "r", "dsh", "zh", "rh", "rr",
        "rx", "dh", "q", "vf", "cl",
    };

    static readonly object lockObj = new();
    static Dictionary<string, int>? graphemeIndexes;
    static IG2p? dict;
    static InferenceSession? session;
    static readonly Dictionary<string, string[]> predCache = new();

    public GermanMarzipanG2p()
    {
        lock (lockObj)
        {
            if (graphemeIndexes == null)
            {
                graphemeIndexes = graphemes
                    .Skip(4)
                    .Select((g, i) => Tuple.Create(g, i))
                    .ToDictionary(t => t.Item1, t => t.Item2 + 4);
                var tuple = LoadPack(EmbeddedG2pPacks.Load("g2p-de-marzipan.zip"));
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

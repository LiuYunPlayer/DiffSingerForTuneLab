using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Api;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/G2p/ItalianG2p.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 意大利语算法 G2P：词表（音素预处理去尾数字重音）+ 神经 OOV。资源包改为从插件嵌入资源加载。
public sealed class ItalianG2p : G2pPack
{
    static readonly string[] graphemes =
    {
        "", "", "", "", "'", "a", "b", "c", "d", "e",
        "f", "g", "h", "i", "j", "k", "l", "m", "n",
        "o", "p", "q", "r", "s", "t", "u", "v", "w",
        "x", "y", "z", "à", "è", "é", "ì", "í", "ò",
        "ù", "ú",
    };

    static readonly string[] phonemes =
    {
        "", "", "", "", "a", "b", "d", "dz", "dZZ", "e",
        "EE", "f", "g", "i", "j", "JJ", "k", "l", "LL",
        "m", "n", "nf", "ng", "o", "OO", "p", "r", "s",
        "SS", "t", "ts", "tSS", "u", "v", "w", "z",
    };

    static readonly object lockObj = new();
    static Dictionary<string, int>? graphemeIndexes;
    static IG2p? dict;
    static InferenceSession? session;
    static readonly Dictionary<string, string[]> predCache = new();

    public ItalianG2p()
    {
        lock (lockObj)
        {
            if (graphemeIndexes == null)
            {
                graphemeIndexes = graphemes
                    .Skip(4)
                    .Select((g, i) => Tuple.Create(g, i))
                    .ToDictionary(t => t.Item1, t => t.Item2 + 4);
                var tuple = LoadPack(
                    EmbeddedG2pPacks.Load("g2p-it.zip"),
                    s => s,
                    s => RemoveTailDigits(s));
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

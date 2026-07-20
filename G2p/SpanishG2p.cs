using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Api;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/G2p/SpanishG2p.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 西班牙语算法 G2P：词表（grapheme 预处理小写化）+ 神经 OOV。资源包改为从插件嵌入资源加载。
public sealed class SpanishG2p : G2pPack
{
    static readonly string[] graphemes =
    {
        "", "", "", "", "\'", "-", "a", "b", "c", "d", "e",
        "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "Á", "É", "Í", "Ó", "Ú", "á", "ã", "é", "ë", "ê", "í",
        "ñ", "ó", "ú", "ü",
    };

    static readonly string[] phonemes =
    {
        "", "", "", "", "a", "b", "B", "ch", "d", "D", "e", "f",
        "g", "G", "gn", "i", "I", "k", "l", "ll", "m", "n", "o",
        "p", "r", "rr", "s", "t", "u", "U", "w", "x", "y", "Y", "z",
    };

    static readonly object lockObj = new();
    static Dictionary<string, int>? graphemeIndexes;
    static IG2p? dict;
    static InferenceSession? session;
    static readonly Dictionary<string, string[]> predCache = new();

    public SpanishG2p()
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
                    EmbeddedG2pPacks.Load("g2p-es.zip"),
                    s => s.ToLowerInvariant());
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

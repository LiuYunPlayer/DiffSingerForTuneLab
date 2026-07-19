using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Api;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/G2p/PortugueseG2p.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 葡萄牙语算法 G2P：词表 + 神经 OOV（X-SAMPA 风格符号，含鼻化 ~ 后缀）。资源包改为从插件嵌入资源加载。
public sealed class PortugueseG2p : G2pPack
{
    static readonly string[] graphemes =
    {
        "", "", "", "", "-", "a", "b", "c", "d", "e", "f", "g", "h",
        "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z", "à", "á", "â", "ã", "ç",
        "è", "é", "ê", "í", "î", "ó", "ô", "õ", "ú", "û", "ü",
    };

    static readonly string[] phonemes =
    {
        "", "", "", "", "E", "J", "L", "O", "R", "S", "X", "Z",
        "a", "a~", "b", "d", "dZ", "e", "e~", "f", "g",
        "i", "i~", "j", "j~", "k", "l", "m", "n", "o", "o~",
        "p", "r", "s", "t", "tS", "u", "u~", "v", "w", "w~", "z",
    };

    static readonly object lockObj = new();
    static Dictionary<string, int>? graphemeIndexes;
    static IG2p? dict;
    static InferenceSession? session;
    static readonly Dictionary<string, string[]> predCache = new();

    public PortugueseG2p()
    {
        lock (lockObj)
        {
            if (graphemeIndexes == null)
            {
                graphemeIndexes = graphemes
                    .Skip(4)
                    .Select((g, i) => Tuple.Create(g, i))
                    .ToDictionary(t => t.Item1, t => t.Item2 + 4);
                var tuple = LoadPack(EmbeddedG2pPacks.Load("g2p-pt.zip"));
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

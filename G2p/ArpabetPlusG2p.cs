using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/G2p/ArpabetPlusG2p.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 英语 ARPA+ 算法 G2P：词表(CMUdict 系) + 神经 OOV。规范符号为小写 arpabet（aa ae … 无尾数字重音）。
//   资源包改为从插件嵌入资源加载（原 OpenUtau 走 Data.Resources）。静态包共享、线程安全初始化。
public sealed class ArpabetPlusG2p : G2pPack
{
    static readonly string[] graphemes =
    {
        "", "", "", "", "\'", "-", "a", "b", "c", "d", "e",
        "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "A", "B", "C", "D", "E",
        "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
        "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    };

    static readonly string[] phonemes =
    {
        "", "", "", "", "aa", "ae", "ah", "ao", "aw", "ax", "ay", "b", "ch",
        "d", "dh", "dr", "dx", "eh", "er", "ey", "f", "g", "hh", "ih", "iy", "jh",
        "k", "l", "m", "n", "ng", "ow", "oy", "p", "q", "r", "s", "sh", "t",
        "th", "tr", "uh", "uw", "v", "w", "y", "z", "zh",
    };

    static readonly object lockObj = new();
    static IG2p? dict;
    static Dictionary<string, int>? graphemeIndexes;
    static InferenceSession? session;
    static readonly Dictionary<string, string[]> predCache = new();

    public ArpabetPlusG2p()
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
                    LoadEmbeddedPack("g2p-arpabet-plus.zip"),
                    s => s.ToLowerInvariant(),
                    s => RemoveTailDigits(s.ToLowerInvariant()));
                dict = tuple.Item1;
                session = tuple.Item2;
            }
        }
        Dict = dict!;
        PredCache = predCache;
        GraphemeIndexes = graphemeIndexes;
        Phonemes = phonemes;
        Session = session;
    }
}

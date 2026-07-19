using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Api;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/G2p/FilipinoG2p.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 菲律宾语（他加禄）算法 G2P：词表（小写化 + 去尾数字）+ 神经 OOV。上游另有纯规则版
//   RuleBasedFilipinoG2p（同词典名、需手选），本插件暂只绑神经包版。资源包改为从插件嵌入资源加载。
public sealed class FilipinoG2p : G2pPack
{
    static readonly string[] graphemes =
    {
        "", "", "", "", "\'", "-", "a", "b", "c", "d", "e", "f", "g",
        "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z", "ñ",
    };

    static readonly string[] phonemes =
    {
        "", "", "", "", "a", "b", "d", "dx", "dy", "e", "f", "g", "h", "hh", "i",
        "j", "k", "l", "m", "n", "ng", "ny", "o", "p", "q", "s",
        "sy", "t", "th", "ts", "u", "v", "w", "z",
    };

    static readonly object lockObj = new();
    static Dictionary<string, int>? graphemeIndexes;
    static IG2p? dict;
    static InferenceSession? session;
    static readonly Dictionary<string, string[]> predCache = new();

    public FilipinoG2p()
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
                    EmbeddedG2pPacks.Load("g2p-fil.zip"),
                    s => s.ToLowerInvariant(),
                    s => RemoveTailDigits(s.ToLowerInvariant()));
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

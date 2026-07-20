using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Api;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/G2p/FrenchMillefeuilleG2p.cs（UFR Millefeuille 法语方案，已并入上游内置）。
// 见仓库根 THIRD-PARTY-NOTICES.md。法语算法 G2P：词表 + 神经 OOV。规范符号为 millefeuille 法语音素
//   （ah eh ae … uy）；多语言前缀声库经 dsdict-fr-millefeuille.yaml 的 replacements 落到声库符号。
//   资源包改为从插件嵌入资源加载。静态包共享、线程安全初始化。
public sealed class FrenchMillefeuilleG2p : G2pPack
{
    static readonly string[] graphemes =
    {
        "", "", "", "", "'", "-", "a", "b", "c", "d", "e", "f",
        "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
        "r", "s", "t", "u", "v", "w", "x", "y", "z", "é",
        "è", "ê", "à", "â", "î", "ô", "ù", "û", "ç", "œ",
        "ï", "(", ")", "0", "1", "2", "3", "4", "5", "6",
        "7", "8", "9",
    };

    static readonly string[] phonemes =
    {
        "", "", "", "", "ah", "eh", "ae", "ee", "oe", "ih", "oh", "oo", "ou",
        "uh", "en", "in", "on", "uy", "y", "w", "f", "k", "p", "s", "sh",
        "t", "h", "b", "d", "g", "l", "m", "n", "r", "v", "z", "j", "ng", "q",
    };

    static readonly object lockObj = new();
    static Dictionary<string, int>? graphemeIndexes;
    static IG2p? dict;
    static InferenceSession? session;
    static readonly Dictionary<string, string[]> predCache = new();

    public FrenchMillefeuilleG2p()
    {
        lock (lockObj)
        {
            if (graphemeIndexes == null)
            {
                graphemeIndexes = graphemes
                    .Skip(4)
                    .Select((g, i) => Tuple.Create(g, i))
                    .ToDictionary(t => t.Item1, t => t.Item2 + 4);
                var tuple = LoadPack(EmbeddedG2pPacks.Load("g2p-fr-millefeuille.zip"));
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

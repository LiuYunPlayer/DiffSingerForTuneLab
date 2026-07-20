using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Api;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/G2p/KoreanG2p.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 韩语算法 G2P：Predict 前先把每个谚文音节按 Unicode 规则拆成 초성/중성/종성 jamo（纯算术、无模型），
//   再走词表 + 神经 OOV。规范符号见 phonemes 表。资源包改为从插件嵌入资源加载。
//   基类 G2pPack 住 OpenUtau.Core 门面程序集（与声库自带音素器 DLL 共用同一实现）。
public sealed class KoreanG2p : G2pPack
{
    static readonly string[] graphemes =
    {
        "", "", "", "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ",
        "ㄸ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄾ", "ㅀ", "ㅁ", "ㅂ", "ㅃ",
        "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ",
        "ㅎ", "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ",
        "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ",
        "ㅢ", "ㅣ",
    };

    static readonly string[] phonemes =
    {
        "", "", "", "", "K", "L", "M", "N", "NG", "P", "T",
        "a", "b", "ch", "d", "e", "eo", "eu", "g", "h",
        "i", "j", "jj", "k", "kk", "m", "n", "o", "p", "pp",
        "r", "s", "ss", "t", "tt", "u", "w", "y",
    };

    static readonly object lockObj = new();
    static Dictionary<string, int>? graphemeIndexes;
    static IG2p? dict;
    static InferenceSession? session;
    static readonly Dictionary<string, string[]> predCache = new();

    public KoreanG2p()
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
                    EmbeddedG2pPacks.Load("g2p-ko.zip"),
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

    protected override string[] Predict(string grapheme)
    {
        var sb = new StringBuilder();
        foreach (var item in grapheme)
        {
            if (TryDivideHangeul(item, out var jamo))
                sb.Append(jamo);
            else
                sb.Append(item);
        }
        return base.Predict(sb.ToString());
    }

    static readonly string onset = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ";
    static readonly string nucleus = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ";
    static readonly string coda = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ";

    const ushort UnicodeHangeulBase = 0xAC00;
    const ushort UnicodeHangeulLast = 0xD79F;

    public bool TryDivideHangeul(char c, out string result)
    {
        ushort check = Convert.ToUInt16(c);
        if (check > UnicodeHangeulLast || check < UnicodeHangeulBase)
        {
            result = "";
            return false;
        }

        int code = check - UnicodeHangeulBase;
        int codaCode = code % 28;
        code = (code - codaCode) / 28;
        int nucleusCode = code % 21;
        code = (code - nucleusCode) / 21;
        int onsetCode = code;

        result = $"{onset[onsetCode]}{nucleus[nucleusCode]}{coda[codaCode]}";
        return true;
    }
}

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/Api/G2pPack.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 算法 G2P 引擎基类：从一个 zip 包（dict.txt 词表 + phones.txt 符号类型 + g2p.onnx 神经模型）装配。
//   Query：先查内置词表，未命中再走神经 seq2seq Predict——按 grapheme 缓存（PredCache，子类静态、跨实例/跨重跑存活，
//   故同一词在频繁的整块 G2P 重跑里只推一次）。产物为「规范符号」，由上层 G2pRemapper 落到声库音素。
//   OpenUtau 的 Zip 工具换成 System.IO.Compression；其余忠实保留。
public abstract class G2pPack : IG2p
{
    protected static readonly Regex kAllPunct = new(@"^[\p{P}]$");

    protected Dictionary<string, int> GraphemeIndexes { get; set; } = new();
    protected string[] Phonemes { get; set; } = Array.Empty<string>();
    protected IG2p Dict { get; set; } = null!;
    protected InferenceSession? Session { get; set; }
    protected Dictionary<string, string[]> PredCache { get; set; } = new();

    protected Tuple<IG2p, InferenceSession> LoadPack(
        byte[] data,
        Func<string, string>? prepGrapheme = null,
        Func<string, string>? prepPhoneme = null)
    {
        prepGrapheme ??= s => s;
        prepPhoneme ??= s => s;
        string[] dictTxt = ExtractText(data, "dict.txt");
        string[] phonesTxt = ExtractText(data, "phones.txt");
        byte[] g2pData = ExtractBytes(data, "g2p.onnx");
        var builder = G2pDictionary.NewBuilder();
        phonesTxt.Select(line => line.Trim())
            .Select(line => line.Split())
            .Where(parts => parts.Length == 2)
            .ToList()
            .ForEach(parts => builder.AddSymbol(prepPhoneme(parts[0]), parts[1]));
        dictTxt.Where(line => !line.StartsWith(";;;"))
            .Select(line => line.Trim())
            .Select(line => line.Split(new string[] { "  " }, StringSplitOptions.None))
            .Where(parts => parts.Length == 2)
            .ToList()
            .ForEach(parts => builder.AddEntry(
                prepGrapheme(parts[0]),
                parts[1].Split().Select(symbol => prepPhoneme(symbol))));
        var dict = builder.Build();
        var session = new InferenceSession(g2pData);
        return Tuple.Create((IG2p)dict, session);
    }

    public static string RemoveTailDigits(string s)
    {
        while (s.Length > 0 && char.IsDigit(s[^1]))
            s = s.Substring(0, s.Length - 1);
        return s;
    }

    public bool IsValidSymbol(string symbol) => Dict.IsValidSymbol(symbol);
    public bool IsVowel(string symbol) => Dict.IsVowel(symbol);
    public bool IsGlide(string symbol) => Dict.IsGlide(symbol);

    public string[]? Query(string grapheme)
    {
        if (grapheme.Length == 0 || kAllPunct.IsMatch(grapheme))
            return null;
        var phonemes = Dict.Query(grapheme);
        if (phonemes == null)
        {
            if (!PredCache.TryGetValue(grapheme, out var cached))
            {
                cached = Predict(grapheme);
                if (cached.Length == 0)
                    return null;
                PredCache.Add(grapheme, cached);
            }
            phonemes = cached;
        }
        return (string[])phonemes.Clone();
    }

    public string[]? UnpackHint(string hint, char separator = ' ') => Dict.UnpackHint(hint, separator);

    // 神经 seq2seq 逐 token 解码：以 2(BOS) 起，t 指向已消费的源位置，预测到 2 即推进源位置，否则追加目标符号；
    //   终止于源耗尽或目标超长(48)。忠实 OpenUtau。
    protected virtual string[] Predict(string grapheme)
    {
        Tensor<int> src = EncodeWord(grapheme);
        if (src.Length == 0 || Session == null)
            return Array.Empty<string>();
        var tgtInit = new DenseTensor<int>(new[] { 1, 1 });
        tgtInit[0, 0] = 2;
        Tensor<int> tgt = tgtInit;
        Tensor<int> t = new DenseTensor<int>(new[] { 1 });
        var srcLength = src.Dimensions[1];
        var inputs = new List<NamedOnnxValue>();
        while (t[0] < srcLength && tgt.Length < 48)
        {
            inputs.Clear();
            inputs.Add(NamedOnnxValue.CreateFromTensor("src", src));
            inputs.Add(NamedOnnxValue.CreateFromTensor("tgt", tgt));
            inputs.Add(NamedOnnxValue.CreateFromTensor("t", t));
            using var outputs = Session.Run(inputs);
            var pred = outputs.First().AsTensor<int>()[0];
            if (pred != 2)
            {
                var newTgt = new DenseTensor<int>(new[] { 1, tgt.Dimensions[1] + 1 });
                for (int i = 0; i < tgt.Dimensions[1]; ++i)
                    newTgt[0, i] = tgt[0, i];
                newTgt[0, tgt.Dimensions[1]] = pred;
                tgt = newTgt;
            }
            else
            {
                t[0] += 1;
            }
        }
        return DecodePhonemes(tgt.Skip(1).ToArray());
    }

    protected Tensor<int> EncodeWord(string grapheme)
    {
        var encoded = new List<int>();
        foreach (char c in grapheme.ToLowerInvariant())
            if (GraphemeIndexes.TryGetValue(c.ToString(), out var index))
                encoded.Add(index);
        var tensor = new DenseTensor<int>(new[] { 1, encoded.Count });
        for (int i = 0; i < encoded.Count; ++i)
            tensor[0, i] = encoded[i];
        return tensor;
    }

    protected string[] DecodePhonemes(int[] indexes) => indexes.Select(idx => Phonemes[idx]).ToArray();

    // —— zip 包内文件读取（取代 OpenUtau.Core.Util.Zip）——
    static string[] ExtractText(byte[] data, string entryName)
    {
        using var ms = new MemoryStream(data);
        using var zip = new ZipArchive(ms, ZipArchiveMode.Read);
        var entry = zip.GetEntry(entryName)
            ?? throw new FileNotFoundException($"g2p pack 内缺少 {entryName}");
        using var reader = new StreamReader(entry.Open(), Encoding.UTF8);
        return reader.ReadToEnd().Split(new[] { "\r\n", "\n" }, StringSplitOptions.None);
    }

    static byte[] ExtractBytes(byte[] data, string entryName)
    {
        using var ms = new MemoryStream(data);
        using var zip = new ZipArchive(ms, ZipArchiveMode.Read);
        var entry = zip.GetEntry(entryName)
            ?? throw new FileNotFoundException($"g2p pack 内缺少 {entryName}");
        using var es = entry.Open();
        using var outMs = new MemoryStream();
        es.CopyTo(outMs);
        return outMs.ToArray();
    }

    // 加载捆绑在插件程序集里的 g2p 资源包（嵌入资源名 = DiffSingerForTuneLab.G2p.Data.<fileName>）。
    protected static byte[] LoadEmbeddedPack(string fileName)
    {
        var asm = typeof(G2pPack).Assembly;
        var resName = $"DiffSingerForTuneLab.G2p.Data.{fileName}";
        using var s = asm.GetManifestResourceStream(resName)
            ?? throw new FileNotFoundException($"嵌入 g2p 资源未找到：{resName}");
        using var ms = new MemoryStream();
        s.CopyTo(ms);
        return ms.ToArray();
    }
}

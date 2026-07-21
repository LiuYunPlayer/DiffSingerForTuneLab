using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OpenUtau.Api {
    // 忠实移植自 OpenUtau（MIT）——OpenUtau.Core/Api/G2pPack.cs。见仓库根 THIRD-PARTY-NOTICES.md。
    // 算法 G2P 引擎基类：从一个 zip 包（dict.txt 词表 + phones.txt 符号类型 + g2p.onnx 神经模型）装配。
    //   Query：先查内置词表，未命中再走神经 seq2seq Predict（按 grapheme 缓存于 PredCache——子类置静态实例共享）。
    //   声库自带音素器 DLL 的 G2pPack 子类按本类的成员签名绑定（ctor / LoadPack / 五个属性 setter /
    //   RemoveTailDigits / Predict 虚槽）——签名不可动。OpenUtau 的 Zip 工具换成 System.IO.Compression。
    public abstract class G2pPack : IG2p {
        protected readonly static Regex kAllPunct = new Regex(@"^[\p{P}]$");

        protected Dictionary<string, int> GraphemeIndexes { get; set; }
        protected string[] Phonemes { get; set; }
        protected IG2p Dict { get; set; }
        protected InferenceSession Session { get; set; }
        protected Dictionary<string, string[]> PredCache { get; set; }

        protected Tuple<IG2p, InferenceSession> LoadPack(
            byte[] data,
            Func<string, string> prepGrapheme = null,
            Func<string, string> prepPhoneme = null) {
            prepGrapheme = prepGrapheme ?? ((string s) => s);
            prepPhoneme = prepPhoneme ?? ((string s) => s);
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

        public static string RemoveTailDigits(string s) {
            while (s.Length > 0 && char.IsDigit(s.Last())) {
                s = s.Substring(0, s.Length - 1);
            }
            return s;
        }

        public bool IsValidSymbol(string symbol) {
            return Dict.IsValidSymbol(symbol);
        }

        public bool IsVowel(string symbol) {
            return Dict.IsVowel(symbol);
        }

        public bool IsGlide(string symbol) {
            return Dict.IsGlide(symbol);
        }

        public string[] Query(string grapheme) {
            if (grapheme.Length == 0 || kAllPunct.IsMatch(grapheme)) {
                return null;
            }
            var phonemes = Dict.Query(grapheme);
            if (phonemes == null && !PredCache.TryGetValue(grapheme, out phonemes)) {
                phonemes = Predict(grapheme);
                if (phonemes.Length == 0) {
                    return null;
                }
                PredCache.Add(grapheme, phonemes);
            }
            return phonemes.Clone() as string[];
        }

        public string[] UnpackHint(string hint, char separator = ' ') {
            return Dict.UnpackHint(hint, separator);
        }

        // 神经 seq2seq 逐 token 解码：以 2(BOS) 起，t 指向已消费的源位置，预测到 2 即推进源位置，否则追加目标符号；
        //   终止于源耗尽或目标超长(48)。
        protected virtual string[] Predict(string grapheme) {
            Tensor<int> src = EncodeWord(grapheme);
            if (src.Length == 0 || Session == null) {
                return new string[0];
            }
            Tensor<int> tgt = new int[,] { { 2 } }.ToTensor();
            Tensor<int> t = new DenseTensor<int>(1);
            var srcLength = src.Dimensions[1];
            var inputs = new List<NamedOnnxValue>();
            while (t[0] < srcLength && tgt.Length < 48) {
                inputs.Clear();
                inputs.Add(NamedOnnxValue.CreateFromTensor("src", src));
                inputs.Add(NamedOnnxValue.CreateFromTensor("tgt", tgt));
                inputs.Add(NamedOnnxValue.CreateFromTensor("t", t));
                var outputs = Session.Run(inputs);
                var pred = outputs.First().AsTensor<int>()[0];
                if (pred != 2) {
                    var newTgt = new DenseTensor<int>(new int[] { 1, tgt.Dimensions[1] + 1 });
                    for (int i = 0; i < tgt.Dimensions[1]; ++i) {
                        newTgt[0, i] = tgt[0, i];
                    }
                    newTgt[0, tgt.Dimensions[1]] = pred;
                    tgt = newTgt;
                } else {
                    t[0] += 1;
                }
                outputs.Dispose();
            }
            var phonemes = DecodePhonemes(tgt.Skip(1).ToArray());
            return phonemes;
        }

        protected Tensor<int> EncodeWord(string grapheme) {
            var encoded = new List<int>();
            foreach (char c in grapheme.ToLowerInvariant()) {
                if (GraphemeIndexes.TryGetValue(c.ToString(), out var index)) {
                    encoded.Add(index);
                }
            }
            var tensor = new DenseTensor<int>(new int[] { 1, encoded.Count });
            for (int i = 0; i < encoded.Count; ++i) {
                tensor[0, i] = encoded[i];
            }
            return tensor;
        }

        protected string[] DecodePhonemes(int[] indexes) {
            return indexes.Select(idx => Phonemes[idx]).ToArray();
        }

        // —— zip 包内文件读取（取代 OpenUtau.Core.Util.Zip）——
        static string[] ExtractText(byte[] data, string entryName) {
            using (var ms = new MemoryStream(data))
            using (var zip = new ZipArchive(ms, ZipArchiveMode.Read)) {
                var entry = zip.GetEntry(entryName);
                if (entry == null) {
                    throw new FileNotFoundException($"g2p pack 内缺少 {entryName}");
                }
                using (var reader = new StreamReader(entry.Open(), Encoding.UTF8)) {
                    return reader.ReadToEnd().Split(new[] { "\r\n", "\n" }, StringSplitOptions.None);
                }
            }
        }

        static byte[] ExtractBytes(byte[] data, string entryName) {
            using (var ms = new MemoryStream(data))
            using (var zip = new ZipArchive(ms, ZipArchiveMode.Read)) {
                var entry = zip.GetEntry(entryName);
                if (entry == null) {
                    throw new FileNotFoundException($"g2p pack 内缺少 {entryName}");
                }
                using (var es = entry.Open())
                using (var outMs = new MemoryStream()) {
                    es.CopyTo(outMs);
                    return outMs.ToArray();
                }
            }
        }

        // —— cfg.yaml 支持（可选，zip 内含此文件时训练配置精确覆盖硬编码）——
        //    由子类在 LoadPack 后调用，成功时用 cfg 的 mapping 覆盖 GraphemeIndexes / Phonemes。
        protected static string[] s_LoadedGraphemes;
        protected static string[] s_LoadedPhonemes;

        // 从 zip byte[] 中解析 cfg.yaml，提取 encoder.graphemes / decoder.phonemes 列表。
        protected static bool TryLoadCfg(byte[] zipData, out string[] graphemes, out string[] phonemes) {
            graphemes = null;
            phonemes = null;
            try {
                using var ms = new MemoryStream(zipData);
                using var zip = new ZipArchive(ms, ZipArchiveMode.Read);
                var entry = zip.GetEntry("cfg.yaml");
                if (entry == null) return false;
                using var reader = new StreamReader(entry.Open(), Encoding.UTF8);
                var lines = reader.ReadToEnd().Split(new[] { "\r\n", "\n" }, StringSplitOptions.None);
                var gList = new List<string>();
                var pList = new List<string>();
                List<string> current = null;
                foreach (var raw in lines) {
                    var line = raw.TrimEnd();
                    if (line.Contains("graphemes:")) { current = gList; continue; }
                    if (line.Contains("phonemes:")) { current = pList; continue; }
                    if (line.StartsWith("- ") && current != null)
                        current.Add(line.Substring(2).Trim());
                }
                if (gList.Count > 0 && pList.Count > 0) {
                    graphemes = gList.ToArray();
                    phonemes = pList.ToArray();
                    return true;
                }
                return false;
            } catch { return false; }
        }
    }
}

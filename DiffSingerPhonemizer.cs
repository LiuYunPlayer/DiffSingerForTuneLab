using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 一个音素的解析结果（绝对秒、归属 note、是否韵核）。时间可越界 note（前置辅音落在 note 起点之前）。
public readonly record struct PhonemeSpan(string Symbol, double StartTime, double EndTime, int NoteIndex, bool IsVowel);

// DiffSinger phonemizer：歌词 → 音素时间线。忠实移植 OpenUtau DiffSingerBasePhonemizer.ProcessPart：
//   · 短语首加前导 SP 组 + 500ms padding（给首辅音留空间），尾加哨兵；
//   · 音素按元音对齐到 note 起点（consonant-glide-vowel 特例：滑音起拍）；onset 辅音归前一组、侵入前一 note 尾；
//   · word_div=各组音素数、word_dur=相邻组（note 起点）间帧数 → linguistic(词模式) + dur → 每音素帧；
//   · 对齐：首辅音保自然时长（ratio=frameSec），其余每组按比例缩放、终点对齐到下一 note 起点。
// 在 TuneLab 绝对秒域工作（OpenUtau 用 tick+tempo，这里 note 边界即秒）。钉死 note（note.Phonemes 非空）
// 用其相对偏移覆盖对齐结果（§5.7：用户钉死全音素，引擎遵守时序）。
public static class DiffSingerPhonemizer
{
    const string Pause = "SP";
    const double PaddingSec = 0.5;   // 短语首辅音前导空间（OpenUtau padding=500ms）

    sealed class Group
    {
        public double Pos;            // 组起点（秒）；元音组=note 起点，首组=首 note 起点-padding，哨兵=末 note 终点
        public int Tone;
        public readonly List<string> Phonemes = new();
        public Group(double pos, int tone) { Pos = pos; Tone = tone; }
    }

    public static List<PhonemeSpan> Phonemize(
        DiffSingerPredictor dur, IReadOnlyList<SynthesisNoteSnapshot> notes,
        IReadOnlyList<string> noteLang, string speaker, int hop, int sampleRate)
    {
        if (notes.Count == 0)
            return new List<PhonemeSpan>();
        double frameSec = (double)hop / sampleRate;

        // —— 构建短语组：首组(前导 SP) + 各 note 的元音对齐分组 + 尾哨兵 ——
        var groups = new List<Group> { new(-1, notes[0].Pitch) { } };
        groups[0].Phonemes.Add(Pause);
        var notePhIndex = new List<int> { 1 };          // 各 note 在展平序列中的起始下标（含前导 SP 占 0）
        var noteSymbolCount = new int[notes.Count];     // 每 note 的音素数（onset+韵核）
        var pinned = new bool[notes.Count];

        for (int i = 0; i < notes.Count; i++)
        {
            var note = notes[i];
            string lyric = note.Lyric ?? string.Empty;
            string[] symbols = GetSymbols(dur, note, noteLang[i], out pinned[i]);
            noteSymbolCount[i] = symbols.Length;

            // 连音符：不产生新音素，只延展前音的时长（通过 group 自然吸收）
            // 连音符：- 不产生边界（前组自然吸收），+ 拆前组末音素到独立组强制 dur 边界
            if (symbols.Length == 0 && (lyric == "-" || lyric == "+"))
            {
                if (lyric == "+")
                {
                    // 把前一个非空组的最后一个音素拆分到 + 组（仅当 >1 音素，否则退化为空组边界）
                    string moved = "AP";
                    bool splitted = false;
                    for (int gi = groups.Count - 1; gi >= 0; gi--)
                    {
                        if (groups[gi].Phonemes.Count > 1)
                        {
                            int lastIdx = groups[gi].Phonemes.Count - 1;
                            moved = groups[gi].Phonemes[lastIdx];
                            groups[gi].Phonemes.RemoveAt(lastIdx);
                            splitted = true;
                            break;
                        }
                    }
                    var g = new Group(note.StartTime, note.Pitch);
                    if (splitted) g.Phonemes.Add(moved);
                    groups.Add(g);
                }
                // - 则完全不做任何事，前组自然吸收时长，不影响 dur
                notePhIndex.Add(notePhIndex[^1]);
                continue;
            }

            var wordGroups = ProcessWord(dur, note, symbols);
            groups[^1].Phonemes.AddRange(wordGroups[0].Phonemes);   // 前置辅音并入前一组（侵入前一 note 尾）
            groups.AddRange(wordGroups.Skip(1));                    // 韵核组（起点=note 起点）
            notePhIndex.Add(notePhIndex[^1] + symbols.Length);
        }

        var last = notes[^1];
        groups.Add(new Group(last.EndTime, last.Pitch));            // 哨兵：标短语终点，无音素
        groups[0].Pos = notes[0].StartTime - PaddingSec;            // 前导 SP 起点

        // —— linguistic(词模式) + dur ——
        var flatSymbols = groups.SelectMany(g => g.Phonemes).ToArray();
        var flatTone = groups.SelectMany(g => Enumerable.Repeat(g.Tone, g.Phonemes.Count)).ToArray();
        int nTokens = flatSymbols.Length;

        var tokens = flatSymbols.Select(s => (long)dur.PhonemeToken(s)).ToArray();
        var langs = flatSymbols.Select(s => dur.LangId(PhonemeLanguage(s))).ToArray();
        var wordDiv = groups.Take(groups.Count - 1).Select(g => (long)g.Phonemes.Count).ToArray();
        var wordDur = groups.Zip(groups.Skip(1), (a, b) => (long)FramesBetween(a.Pos, b.Pos, frameSec)).ToArray();

        var durationFrames = RunDur(dur, tokens, langs, wordDiv, wordDur, flatTone, speaker);

        // —— 对齐：首辅音自然，其余每组按比例缩放，终点对齐到下一组(note)起点 ——
        // phAlignPoints[k] = (展平下标=第 k+1 组首音素, 第 k+1 组起点秒)
        var cum = new int[groups.Count];
        for (int i = 1; i < groups.Count; i++) cum[i] = cum[i - 1] + groups[i - 1].Phonemes.Count;
        var alignIdx = new List<int>();
        var alignPos = new List<double>();
        for (int k = 1; k < groups.Count; k++) { alignIdx.Add(cum[k]); alignPos.Add(groups[k].Pos); }

        var positions = new List<double>(nTokens - 1);   // flat[1..] 的起点秒
        // 首组：跳过 index 0 的前导 SP，取首 note 的前置辅音，自然时长(ratio=frameSec)、终点对齐首 note 起点
        positions.AddRange(Stretch(durationFrames, 1, alignIdx[0] - 1, frameSec, alignPos[0]));
        for (int k = 0; k < alignIdx.Count - 1; k++)
        {
            int from = alignIdx[k], to = alignIdx[k + 1];
            double sum = 0;
            for (int j = from; j < to; j++) sum += durationFrames[j];
            double ratio = sum > 0 ? (alignPos[k + 1] - alignPos[k]) / sum : frameSec;
            positions.AddRange(Stretch(durationFrames, from, to - from, ratio, alignPos[k + 1]));
        }

        // —— 展平 → body 音素时间线（flat[1..]，跳过前导 SP；末音素终点=哨兵=末 note 终点）——
        var result = new List<PhonemeSpan>(nTokens - 1);
        int bodyCount = nTokens - 1;
        for (int b = 0; b < bodyCount; b++)
        {
            int flatIndex = b + 1;
            double start = positions[b];
            double end = b + 1 < bodyCount ? positions[b + 1] : last.EndTime;
            int noteIndex = NoteOf(notePhIndex, flatIndex);
            string sym = flatSymbols[flatIndex];
            result.Add(new PhonemeSpan(sym, start, end, noteIndex, dur.IsVowel(sym)));
        }

        // —— 钉死 note：用相对偏移覆盖（§5.7）——
        for (int i = 0; i < notes.Count; i++)
        {
            if (!pinned[i]) continue;
            var note = notes[i];
            int baseIdx = notePhIndex[i] - 1;   // body 下标（flat-1）
            for (int j = 0; j < note.Phonemes.Count && baseIdx + j < result.Count; j++)
            {
                var p = note.Phonemes[j];
                var span = result[baseIdx + j];
                result[baseIdx + j] = span with
                {
                    StartTime = note.StartTime + p.StartTime,
                    EndTime = note.StartTime + p.EndTime,
                };
            }
        }
        return result;
    }

    // 单 note 的音素→组分配：元音起拍（consonant-glide-vowel：滑音起拍）；前置辅音落在 wordGroups[0]（归前组）。
    static List<Group> ProcessWord(DiffSingerPredictor dur, SynthesisNoteSnapshot note, string[] symbols)
    {
        var wordGroups = new List<Group> { new(-1, note.Pitch) };
        var isVowel = symbols.Select(dur.IsVowel).ToArray();
        var isGlide = symbols.Select(dur.IsGlide).ToArray();
        var isStart = new bool[symbols.Length];
        if (isVowel.All(b => !b) && symbols.Length > 0)
            isStart[0] = true;
        for (int i = 0; i < symbols.Length; i++)
        {
            if (!isVowel[i]) continue;
            if (i >= 2 && isGlide[i - 1] && !isVowel[i - 2])
                isStart[i - 1] = true;
            else
                isStart[i] = true;
        }
        bool started = false;
        for (int i = 0; i < symbols.Length; i++)
        {
            if (isStart[i] && !started) { wordGroups.Add(new Group(note.StartTime, note.Pitch)); started = true; }
            wordGroups[^1].Phonemes.Add(symbols[i]);
        }
        return wordGroups;
    }

    // 取音素符号串：钉死/编辑器→用已有 phonemes 或 _phonemes 属性；连音符→空（slur 延展前音素）；否则 G2P。
    // 空结果且非连音符→ [SP] 兜底。
    static string[] GetSymbols(DiffSingerPredictor dur, SynthesisNoteSnapshot note, string lang, out bool pinned)
    {
        string lyric = note.Lyric ?? string.Empty;
        if (lyric == "-" || lyric == "+")
        {
            pinned = false;
            return Array.Empty<string>();
        }

        // 优先使用 _phonemes 属性（音素编辑器写入）
        var phonemesProp = note.Properties.GetString("_phonemes", "");
        if (!string.IsNullOrEmpty(phonemesProp) && phonemesProp != "[]")
        {
            pinned = true;
            return ParsePhonemesProperty(phonemesProp).Where(s => !string.IsNullOrEmpty(s) && dur.TryPhoneme(s, out _)).ToArray();
        }

        // 其次使用钉死音素
        pinned = note.Phonemes.Count > 0;
        IEnumerable<string> raw = pinned
            ? note.Phonemes.Select(p => p.Symbol)
            : dur.G2P(lyric, lang);
        var symbols = raw.Where(s => dur.IsKnownSymbol(s) && dur.TryPhoneme(s, out _)).ToArray();
        return symbols.Length > 0 ? symbols : new[] { Pause };
    }

    // 从 JSON 字符串解析音素符号列表：[{"s":"ja/b","v":false},...]
    static string[] ParsePhonemesProperty(string json)
    {
        var result = new List<string>();
        if (string.IsNullOrEmpty(json) || json.Length < 2) return result.ToArray();
        try
        {
            int i = 0;
            while (true)
            {
                int sIdx = json.IndexOf("\"s\":\"", i);
                if (sIdx < 0) break;
                sIdx += 5;
                int eIdx = json.IndexOf('"', sIdx);
                if (eIdx > sIdx) result.Add(json.Substring(sIdx, eIdx - sIdx));
                i = eIdx + 1;
            }
        }
        catch { }
        return result.ToArray();
    }

    // OpenUtau stretch：source[from..from+count) 的帧时长按 ratio 缩放、终点对齐 endPos，返回各音素起点秒。
    static IEnumerable<double> Stretch(IReadOnlyList<double> source, int from, int count, double ratio, double endPos)
    {
        double sum = 0;
        for (int j = 0; j < count; j++) sum += source[from + j];
        double pos = endPos - sum * ratio;
        for (int j = 0; j < count; j++)
        {
            yield return pos;
            pos += source[from + j] * ratio;
        }
    }

    static int FramesBetween(double t1, double t2, double frameSec)
        => (int)(t2 / frameSec) - (int)(t1 / frameSec);

    static int NoteOf(List<int> notePhIndex, int flatIndex)
    {
        // notePhIndex[i] = note i 起始 flat 下标；返回包含 flatIndex 的 note。
        for (int i = 0; i < notePhIndex.Count - 1; i++)
            if (flatIndex >= notePhIndex[i] && flatIndex < notePhIndex[i + 1])
                return i;
        return Math.Max(0, notePhIndex.Count - 2);
    }

    static string PhonemeLanguage(string phoneme)
    {
        int slash = phoneme.IndexOf('/');
        return slash > 0 ? phoneme[..slash] : string.Empty;
    }

    // linguistic(词模式) + dur → 每音素预测帧（float）。
    static double[] RunDur(DiffSingerPredictor dur, long[] tokens, long[] langs,
        long[] wordDiv, long[] wordDur, int[] phMidi, string speaker)
    {
        int nTokens = tokens.Length, nWords = wordDiv.Length, hidden = dur.HiddenSize;
        var lingInputs = new List<NamedOnnxValue>
        {
            Nv("tokens", tokens, nTokens),
            Nv("word_div", wordDiv, nWords),
            Nv("word_dur", wordDur, nWords),
        };
        if (dur.Linguistic.InputMetadata.ContainsKey("languages"))
            lingInputs.Add(Nv("languages", langs, nTokens));

        using var lingOut = dur.RunLinguistic(lingInputs);
        var enc = lingOut.First(v => v.Name == "encoder_out").AsTensor<float>();
        var mask = lingOut.First(v => v.Name == "x_masks").AsTensor<bool>();
        var encDense = new DenseTensor<float>(enc.ToArray(), enc.Dimensions.ToArray());
        var maskDense = new DenseTensor<bool>(mask.ToArray(), mask.Dimensions.ToArray());

        float[] emb = dur.GetEmbedding(speaker);
        var spk = new float[nTokens * hidden];
        for (int i = 0; i < nTokens; i++) Array.Copy(emb, 0, spk, i * hidden, hidden);

        using var durOut = dur.RunModel("dur", new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_out", encDense),
            NamedOnnxValue.CreateFromTensor("x_masks", maskDense),
            Nv("ph_midi", phMidi.Select(x => (long)x).ToArray(), nTokens),
            NamedOnnxValue.CreateFromTensor("spk_embed", new DenseTensor<float>(spk, new[] { 1, nTokens, hidden })),
        });
        return durOut.First(v => v.Name == "ph_dur_pred").AsTensor<float>().Select(x => (double)x).ToArray();
    }

    static NamedOnnxValue Nv(string name, long[] data, int n)
        => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, new[] { 1, n }));
}

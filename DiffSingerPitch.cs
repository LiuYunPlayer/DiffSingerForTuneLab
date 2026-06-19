using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// pitch 预测器（dspitch）：忠实移植 OpenUtau DsPitch.Process。
//   linguistic（词模式 word_div/word_dur 或音素模式 ph_dur，按编码器实际输入判）+ pitch 模型；
//   从音符构造 note_midi / note_dur(帧) / note_rest（head/tail padding、间隙插 rest、slur 继承、rest 组 tone 填充）；
//   pitch 基值全 60、retake 全 true ⇒ 不吃用户音高，纯从音符全量预测自然音高轮廓（逐帧 MIDI）。
//   PEXP（expr）本阶段喂中性 1.0（满表现力）；steps 暂与声学共用（OpenUtau 另有 DiffSingerStepsPitch，后续统一）。
// 调用方（Render）拿到预测轮廓后，用它替代自由区（用户音高 NaN 处）的矩形 note-step 兜底；用户已画处用户值覆盖，
// PITD/vibrato 永远叠加在上（见共识：自由区填 f0 + 回显、retake 全 true 事后合并）。无 dspitch ⇒ 返回 null 降级。
public static class DiffSingerPitch
{
    // phones = body 音素（不含 head/tail）；phDur = padded 帧（len=phones+2，和=totalFrames）；
    // renderStart = 渲染起点秒（phones[0].StartTime - head*frameSec）。返回逐帧 MIDI 音高（len=totalFrames），无预测器返回 null。
    public static float[]? Predict(
        DiffSingerPredictor? v, IReadOnlyList<PhonemeSpan> phones,
        IReadOnlyList<SynthesisNoteSnapshot> notes, int[] phDur,
        double renderStart, double frameSec, DiffSingerSpeakerMix mix, VoicebankConfig cfg, int steps)
    {
        if (v is null || !v.HasModel("pitch") || phones.Count == 0 || notes.Count == 0)
            return null;

        int hidden = v.HiddenSize;
        int nTokens = phones.Count + 2;
        int totalFrames = phDur.Sum();
        int head = DiffSingerFrames.HeadFrames, tail = DiffSingerFrames.TailFrames;

        var tokens = phones.Select(p => (long)v.PhonemeToken(p.Symbol))
            .Prepend((long)v.PhonemeToken("SP")).Append((long)v.PhonemeToken("SP")).ToArray();

        // —— linguistic 编码器（词模式吃 word_div/word_dur，否则音素模式吃 ph_dur）——
        var lingInputs = new List<NamedOnnxValue> { NvL("tokens", tokens, nTokens) };
        if (v.LinguisticUsesWordBoundary)
        {
            var isVowel = phones.Select(p => v.IsVowel(p.Symbol)).ToArray();
            var (wordDiv, wordDur) = DiffSingerFrames.PaddedWordDivAndDur(isVowel, phDur);
            lingInputs.Add(NvL("word_div", wordDiv, wordDiv.Length));
            lingInputs.Add(NvL("word_dur", wordDur, wordDur.Length));
        }
        else
        {
            lingInputs.Add(NvL("ph_dur", phDur.Select(x => (long)x).ToArray(), nTokens));
        }
        if (v.Linguistic.InputMetadata.ContainsKey("languages"))
        {
            var langs = phones.Select(p => v.LangId(PhonemeLanguage(p.Symbol))).Prepend(0L).Append(0L).ToArray();
            lingInputs.Add(NvL("languages", langs, nTokens));
        }
        using var lingOut = v.Linguistic.Run(lingInputs);
        var enc = lingOut.First(o => o.Name == "encoder_out").AsTensor<float>();
        var encDense = new DenseTensor<float>(enc.ToArray(), enc.Dimensions.ToArray());

        // —— note 序列：head padding + 各 note（间隙插 rest）+ tail padding；rest 组 tone 由最近非 rest 填充 ——
        var (noteMidi, noteDur, noteRest) = BuildNotes(v, phones, notes, renderStart, frameSec, totalFrames, head, tail);

        // —— pitch 模型：pitch 全 60、retake 全 true（全量预测）——
        var model = v.Model("pitch");
        var pitch = new float[totalFrames];
        Array.Fill(pitch, 60f);
        var retake = new bool[totalFrames];
        Array.Fill(retake, true);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_out", encDense),
            NvF("note_midi", noteMidi, noteMidi.Length),
            NvL("note_dur", noteDur.Select(x => (long)x).ToArray(), noteDur.Length),
            NvL("ph_dur", phDur.Select(x => (long)x).ToArray(), nTokens),
            NvF("pitch", pitch, totalFrames),
            NamedOnnxValue.CreateFromTensor("retake", new DenseTensor<bool>(retake, new[] { 1, totalFrames })),
        };

        AddAccel(inputs, model, cfg, steps);

        // 表现力（PEXP）：本阶段喂中性 1.0（满表现力）；可编辑 PEXP 轨后续再加。
        if (model.InputMetadata.ContainsKey("expr"))
        {
            var expr = new float[totalFrames];
            Array.Fill(expr, 1f);
            inputs.Add(NvF("expr", expr, totalFrames));
        }
        if (model.InputMetadata.ContainsKey("spk_embed"))
        {
            var spk = mix.ToEmbedding(v.GetEmbedding, hidden);
            inputs.Add(NamedOnnxValue.CreateFromTensor("spk_embed",
                new DenseTensor<float>(spk, new[] { 1, totalFrames, hidden })));
        }
        if (model.InputMetadata.ContainsKey("note_rest"))
            inputs.Add(NamedOnnxValue.CreateFromTensor("note_rest",
                new DenseTensor<bool>(noteRest, new[] { 1, noteRest.Length })));

        using var outputs = model.Run(inputs);
        return outputs.First().AsTensor<float>().ToArray();
    }

    // note 序列构造（忠实移植 OpenUtau）：head padding(rest) + 各 note（间隙插 rest note）+ tail padding(rest)。
    //   note_rest：slur（歌词 +）继承前一个；否则该 note 的音素全为辅音/AP/SP（无真元音）⇒ rest。
    //   note_midi：rest 组的 tone 由最近的非 rest note 填充（全 rest ⇒ 全填 60）。
    static (float[] midi, int[] durFrames, bool[] rest) BuildNotes(
        DiffSingerPredictor v, IReadOnlyList<PhonemeSpan> phones,
        IReadOnlyList<SynthesisNoteSnapshot> notes,
        double renderStart, double frameSec, int totalFrames, int head, int tail)
    {
        var durSec = new List<double>();
        var midiList = new List<float>();
        var restList = new List<bool>();

        // head padding（首 note 起点之前，含越界前置辅音的空间）。
        durSec.Add(Math.Max(0, notes[0].StartTime - renderStart));
        midiList.Add(notes[0].Pitch);
        restList.Add(true);

        double prevEnd = notes[0].StartTime;
        for (int i = 0; i < notes.Count; i++)
        {
            var note = notes[i];
            double gap = note.StartTime - prevEnd;
            if (gap > 0)
            {
                durSec.Add(gap);
                midiList.Add(note.Pitch);
                restList.Add(true);
            }
            durSec.Add(Math.Max(0, note.EndTime - note.StartTime));
            midiList.Add(note.Pitch);
            if ((note.Lyric ?? string.Empty).StartsWith("+"))
            {
                restList.Add(restList[^1]);
            }
            else
            {
                bool isRest = true;
                foreach (var p in phones)
                {
                    if (p.NoteIndex != i) continue;
                    if (p.Symbol != "AP" && p.Symbol != "SP" && v.IsVowel(p.Symbol)) { isRest = false; break; }
                }
                restList.Add(isRest);
            }
            prevEnd = note.EndTime;
        }

        // tail padding。
        durSec.Add(tail * frameSec);
        midiList.Add(notes[^1].Pitch);
        restList.Add(true);

        var midi = midiList.ToArray();
        var rest = restList.ToArray();
        FillRestTones(midi, rest);

        var durFrames = DiffSingerFrames.FitDurationSum(
            DiffSingerFrames.DurationsToFrames(durSec, frameSec), totalFrames);
        return (midi, durFrames, rest);
    }

    // rest 组 tone 填充：每段连续 rest 用最近的非 rest note tone 填（首段用其后、末段用其前、中间段从中点劈半）。
    static void FillRestTones(float[] midi, bool[] rest)
    {
        int n = rest.Length;
        if (rest.All(r => r)) { Array.Fill(midi, 60f); return; }

        var groups = new List<(int start, int end)>();
        for (int i = 0; i < n; i++)
        {
            if (!rest[i]) continue;
            int j = i + 1;
            while (j < n && rest[j]) j++;
            groups.Add((i, j));
            i = j;
        }
        foreach (var (start, end) in groups)
        {
            if (start == 0)
                Array.Fill(midi, midi[end], 0, end);
            else if (end == n)
                Array.Fill(midi, midi[start - 1], start, n - start);
            else
            {
                int mid = (start + end + 1) / 2;
                Array.Fill(midi, midi[start - 1], start, mid - start);
                Array.Fill(midi, midi[end], mid, end - mid);
            }
        }
    }

    static void AddAccel(List<NamedOnnxValue> inputs, InferenceSession model, VoicebankConfig cfg, int steps)
    {
        if (cfg.UseContinuousAcceleration)
        {
            if (model.InputMetadata.ContainsKey("steps"))
                inputs.Add(NamedOnnxValue.CreateFromTensor("steps",
                    new DenseTensor<long>(new[] { (long)steps }, new[] { 1 })));
        }
        else if (model.InputMetadata.ContainsKey("speedup"))
        {
            long speedup = Math.Max(1, 1000 / Math.Max(1, steps));
            while (1000 % speedup != 0 && speedup > 1) speedup--;
            inputs.Add(NamedOnnxValue.CreateFromTensor("speedup",
                new DenseTensor<long>(new[] { speedup }, new[] { 1 })));
        }
    }

    static string PhonemeLanguage(string phoneme)
    {
        int slash = phoneme.IndexOf('/');
        return slash > 0 ? phoneme[..slash] : string.Empty;
    }

    static NamedOnnxValue NvL(string name, long[] data, int n)
        => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, new[] { 1, n }));
    static NamedOnnxValue NvF(string name, float[] data, int n)
        => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(data, new[] { 1, n }));
}

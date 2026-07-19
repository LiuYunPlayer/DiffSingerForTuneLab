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
//   不吃用户音高（用户编辑事后合并）；seed 轨驱动帧级 retake mask（对齐 acoustic 语义）：
//   全保留/全重摇单趟全量预测（pitch 基值全 60、retake 全 true），混合时先 seed=0 算 take-0 参照、
//   再正式趟 retake=逐帧 mask + pitch 口喂参照 ⇒ 保留帧被条件钉住（硬局部化，输出=take-0）。
//   PEXP（expr）本阶段喂中性 1.0（满表现力）；steps 暂与声学共用（OpenUtau 另有 DiffSingerStepsPitch，后续统一）。
// 调用方（Render）拿到预测轮廓后，用它替代自由区（用户音高 NaN 处）的矩形 note-step 兜底；用户已画处用户值覆盖，
// PITD/vibrato 永远叠加在上（见共识：自由区填 f0 + 回显、事后合并）。无 dspitch ⇒ 返回 null 降级。
public static class DiffSingerPitch
{
    // phones = body 音素（不含 head/tail）；phDur = padded 帧（len=phones+2，和=totalFrames）；
    // renderStart = 渲染起点秒（phones[0].StartTime - head*frameSec）。返回逐帧 MIDI 音高（len=totalFrames），无预测器返回 null。
    public static float[]? Predict(
        DiffSingerPredictor? v, IReadOnlyList<PhonemeSpan> phones,
        IReadOnlyList<VoiceSynthesisNoteSnapshot> notes, int[] phDur,
        double renderStart, double frameSec, DiffSingerSpeakerMix mix, VoicebankConfig cfg, int steps, uint[] seedPerFrame, bool tensorCache,
        float[][]? blendRows = null)
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
        //   base 与（若有混合）目标 token 流各跑一次；结构输入(word_div/word_dur/ph_dur/languages)共享，只换 tokens。
        List<NamedOnnxValue> BuildLing(long[] toks)
        {
            var li = new List<NamedOnnxValue> { NvL("tokens", toks, nTokens) };
            if (v.LinguisticUsesWordBoundary)
            {
                var isVowel = phones.Select(p => p.IsVowel).ToArray();   // 结构按 base 音素分组（目标共用）；类型取 phonemizer 定型值
                var (wordDiv, wordDur) = DiffSingerFrames.PaddedWordDivAndDur(isVowel, phDur);
                li.Add(NvL("word_div", wordDiv, wordDiv.Length));
                li.Add(NvL("word_dur", wordDur, wordDur.Length));
            }
            else
            {
                li.Add(NvL("ph_dur", phDur.Select(x => (long)x).ToArray(), nTokens));
            }
            if (v.Linguistic.HasInput("languages"))
            {
                var langs = phones.Select(p => v.LangId(PhonemeLanguage(p.Symbol))).Prepend(0L).Append(0L).ToArray();
                li.Add(NvL("languages", langs, nTokens));
            }
            // linguistic 若把 tokens_b/blend 列为必需输入（P1-a 导出遗留），喂空操作满足——帧级混合在 role 模型做，不在此。
            if (v.Linguistic.HasInput("tokens_b"))
            {
                li.Add(NvL("tokens_b", (long[])toks.Clone(), nTokens));
                li.Add(NvF("blend", new float[nTokens], nTokens));
            }
            return li;
        }
        DenseTensor<float> Encode(long[] toks)
        {
            var o = DiffSingerTensorCache.Run(v.Linguistic, v.LinguisticHash, BuildLing(toks), tensorCache);
            var e = o.First(x => x.Name == "encoder_out").AsTensor<float>();
            return new DenseTensor<float>(e.ToArray(), e.Dimensions.ToArray());
        }
        var encDense = Encode(tokens);

        // —— note 序列：head padding + 各 note（间隙插 rest）+ tail padding；rest 组 tone 由最近非 rest 填充 ——
        var (noteMidi, noteDur, noteRest) = BuildNotes(phones, notes, renderStart, frameSec, totalFrames, head, tail);

        // —— pitch 模型共享条件（不含 pitch/retake/noise，那三个按下方重摇逻辑逐趟追加）——
        var model = v.Model("pitch");
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_out", encDense),
            NvF("note_midi", noteMidi, noteMidi.Length),
            NvL("note_dur", noteDur.Select(x => (long)x).ToArray(), noteDur.Length),
            NvL("ph_dur", phDur.Select(x => (long)x).ToArray(), nTokens),
        };

        // 音素混合（帧级，N 槽）：role 模型把 encoder_out_b/blend 列为**必需**输入（导出恒有），故只要模型声明就**总是**喂。
        //   逐槽 r：目标流(base 换该槽目标)编码出一行 encoder_out_b[r]（该槽无目标 ⇒ 复用 base，no-op）；blend[r]=blendRows[r]。
        //   S=mixRows（≥1）；无混合时 S=1 且 blend 全 0、目标=base ⇒ base 权重=1、等价不混（避免缺必需输入崩溃）。
        if (model.HasInput("encoder_out_b"))
        {
            int mixRows = blendRows is { Length: > 0 } ? blendRows.Length : 1;
            var ebFlat = new float[mixRows * nTokens * hidden];
            for (int r = 0; r < mixRows; r++)
            {
                var tgt = DiffSingerPredictor.BuildMixTargetTokens(v, phones, tokens, r, out bool anyMix);
                var arr = (anyMix ? Encode(tgt) : encDense).ToArray();
                Array.Copy(arr, 0, ebFlat, r * nTokens * hidden, nTokens * hidden);
            }
            inputs.Add(NamedOnnxValue.CreateFromTensor("encoder_out_b",
                new DenseTensor<float>(ebFlat, new[] { mixRows, nTokens, hidden })));
            var blFlat = new float[mixRows * totalFrames];
            for (int r = 0; r < mixRows; r++)
                if (blendRows != null && r < blendRows.Length)
                    Array.Copy(blendRows[r], 0, blFlat, r * totalFrames, totalFrames);
            inputs.Add(NamedOnnxValue.CreateFromTensor("blend", new DenseTensor<float>(blFlat, new[] { mixRows, totalFrames })));
        }

        AddAccel(inputs, model, cfg, steps);

        // 表现力（PEXP）：本阶段喂中性 1.0（满表现力）；可编辑 PEXP 轨后续再加。
        if (model.HasInput("expr"))
        {
            var expr = new float[totalFrames];
            Array.Fill(expr, 1f);
            inputs.Add(NvF("expr", expr, totalFrames));
        }
        if (model.HasInput("spk_embed"))
        {
            var spk = mix.ToEmbedding(v.GetEmbedding, hidden);
            inputs.Add(NamedOnnxValue.CreateFromTensor("spk_embed",
                new DenseTensor<float>(spk, new[] { 1, totalFrames, hidden })));
        }
        if (model.HasInput("note_rest"))
            inputs.Add(NamedOnnxValue.CreateFromTensor("note_rest",
                new DenseTensor<bool>(noteRest, new[] { 1, noteRest.Length })));

        // —— 帧级 seed retake（与 session 声学段同语义）：seed≠0 的帧重摇、seed=0 的帧保留 take-0。
        //   openvpi 导出保留帧条件 = base_pitch*retake + pitch*~retake（melody encoder 变体 delta_pitch*~retake），
        //   故保留帧把 take-0 参照喂进 pitch 口即被钉住；重摇帧的 pitch 输入被 ~retake 乘 0 屏蔽、喂什么都无效。
        //   · 非混合（全保留 or 全重摇）→ 单趟：pitch 全 60、retake 全 true、噪声按 seed 轨（全 0 即 take-0）。
        //   · 混合 → 先算 take-0 参照（同单趟但噪声 seed=0；输入恒定 → 张量缓存只算一次），
        //     再正式趟（retake=逐帧、pitch=参照、噪声按 seed 轨）。参照确定性重算，不记忆上次输出。
        float[] RunPitch(float[] basePitch, bool[] retakeMask, uint[]? seeds)
        {
            var ins = new List<NamedOnnxValue>(inputs)
            {
                NvF("pitch", basePitch, totalFrames),
                NamedOnnxValue.CreateFromTensor("retake", new DenseTensor<bool>(retakeMask, new[] { 1, totalFrames })),
            };
            if (seeds != null)
                DiffSingerNoise.AddNoise(ins, model, seeds, DiffSingerNoise.StagePitch, totalFrames);
            else
                DiffSingerNoise.AddNoise(ins, model, 0u, DiffSingerNoise.StagePitch, totalFrames);
            return DiffSingerTensorCache.Run(model, v.ModelHash("pitch"), ins, tensorCache)
                .First().AsTensor<float>().ToArray();
        }

        var allTrue = new bool[totalFrames];
        Array.Fill(allTrue, true);
        var basePitch60 = new float[totalFrames];
        Array.Fill(basePitch60, 60f);

        var mask = new bool[totalFrames];
        bool anyKeep = false, anyReroll = false;
        for (int f = 0; f < totalFrames; f++)
        {
            bool rr = seedPerFrame[Math.Min(f, seedPerFrame.Length - 1)] != 0;
            mask[f] = rr;
            anyReroll |= rr;
            anyKeep |= !rr;
        }

        if (!(anyKeep && anyReroll))
            return RunPitch(basePitch60, allTrue, seedPerFrame);

        var refPitch = RunPitch(basePitch60, allTrue, null);   // take-0 参照
        return RunPitch(refPitch, mask, seedPerFrame);
    }

    // note 序列构造（移植 OpenUtau，叠加重叠扩展）：head padding(rest) + 各 note（间隙插 rest note）+ tail padding(rest)。
    //   note_rest：slur（歌词 +）继承前一个；否则该 note 的音素全为辅音/AP/SP（无真元音）⇒ rest。
    //   note_midi：rest 组的 tone 由最近的非 rest note 填充（全 rest ⇒ 全填 60）。
    //   头盖尾（OpenUtau 无、本插件扩展）：note 与后一 note 重叠时，有效终点截到后一 note 起点，使 note_dur 与
    //   phonemizer/声学侧的截断时间线同口径——否则 pitch 模型按 note 全长走、轮廓越过后一 note 起点（不让位）。
    //   同起点和弦退化为 dur=0 塌缩（排序长者在前先塌，短者存活）。
    static (float[] midi, int[] durFrames, bool[] rest) BuildNotes(
        IReadOnlyList<PhonemeSpan> phones,
        IReadOnlyList<VoiceSynthesisNoteSnapshot> notes,
        double renderStart, double frameSec, int totalFrames, int head, int tail)
    {
        var durSec = new List<double>();
        var midiList = new List<float>();
        var restList = new List<bool>();

        // melisma ⟺ 无归属自己的音素（phonemizer 对发声 note 恒发至少一个符号、SP 兜底，故可由产物直接推导，
        //   与 phonemize 的延音决策天然一对一，不引第二份判定）。
        var hasOwnPhones = new bool[notes.Count];
        foreach (var p in phones)
            hasOwnPhones[p.NoteIndex] = true;

        // head padding（首 note 起点之前，含越界前置辅音的空间）。
        durSec.Add(Math.Max(0, notes[0].StartTime - renderStart));
        midiList.Add(notes[0].Pitch);
        restList.Add(true);

        double prevEnd = notes[0].StartTime;
        for (int i = 0; i < notes.Count; i++)
        {
            var note = notes[i];
            // 头盖尾：与后一 note 重叠时，本 note 有效终点截到后一 note 起点（同起点 ⇒ 截到自身起点 ⇒ dur=0 塌缩）。
            double effectiveEnd = i + 1 < notes.Count
                ? Math.Min(note.EndTime, notes[i + 1].StartTime)
                : note.EndTime;
            double gap = note.StartTime - prevEnd;
            if (gap > 0)
            {
                durSec.Add(gap);
                midiList.Add(note.Pitch);
                restList.Add(true);
            }
            durSec.Add(Math.Max(0, effectiveEnd - note.StartTime));
            midiList.Add(note.Pitch);
            // 延音符（melisma，由产物推导：无归属音素）：沿用前一个 note 的 rest 状态（前为发声 ⇒ 本帧也发声、携自身 MIDI 滑过去）。
            if (!hasOwnPhones[i])
            {
                restList.Add(restList[^1]);
            }
            else
            {
                bool isRest = true;
                foreach (var p in phones)
                {
                    if (p.NoteIndex != i) continue;
                    if (p.Symbol != "AP" && p.Symbol != "SP" && p.IsVowel) { isRest = false; break; }
                }
                restList.Add(isRest);
            }
            prevEnd = effectiveEnd;
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

    static void AddAccel(List<NamedOnnxValue> inputs, IModelSession model, VoicebankConfig cfg, int steps)
    {
        if (cfg.UseContinuousAcceleration)
        {
            if (model.HasInput("steps"))
                inputs.Add(NamedOnnxValue.CreateFromTensor("steps",
                    new DenseTensor<long>(new[] { (long)steps }, new[] { 1 })));
        }
        else if (model.HasInput("speedup"))
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

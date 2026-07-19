using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DiffSingerForTuneLab;

// variance 预测器（dsvariance）：忠实移植 OpenUtau DsVariance.Process。
//   linguistic(词模式：word_div/word_dur 由已知 ph_dur 按元音分组) + variance；pitch 输入为半音；输出预测曲线。
//   seed 轨驱动帧级 retake mask（对齐 acoustic/pitch）：全保留/全重摇单趟全量预测（基值 0、retake 全 true），
//   混合时先 seed=0 算 take-0 参照、再正式趟 retake=逐帧 mask（单轨广播到全部通道）+ 各通道口喂参照 ⇒ 保留帧钉住。
// 当前阶段只把预测值喂声学（无用户编辑/回显——留作下一阶段）。
public readonly record struct VarianceCurves(float[]? Energy, float[]? Breathiness, float[]? Voicing, float[]? Tension)
{
    // 按声学输入名取对应预测曲线（无该通道返回 null）。
    public float[]? this[string name] => name switch
    {
        "energy" => Energy,
        "breathiness" => Breathiness,
        "voicing" => Voicing,
        "tension" => Tension,
        _ => null,
    };
}

public static class DiffSingerVariance
{
    // phones = body 音素（不含 head/tail）；phDur = padded 帧（len=phones+2）；pitchSemis = totalFrames 半音曲线。
    //   收 PhonemeSpan（含逐槽 MixSymbols）；blendRows[r] = 第 r 槽逐帧包络比例（与 acoustic 同源、已归一），
    //   帧级混合在 role 模型条件级做（每槽跑一次 linguistic 目标流 + base）。
    public static VarianceCurves Predict(
        DiffSingerPredictor? v, IReadOnlyList<PhonemeSpan> phones, int[] phDur,
        float[] pitchSemis, DiffSingerSpeakerMix mix, VoicebankConfig cfg, int steps, uint[] seedPerFrame, bool tensorCache,
        float[][]? blendRows = null)
    {
        if (v is null || !v.HasModel("variance") || phones.Count == 0)
            return default;

        int hidden = v.HiddenSize;
        int nTokens = phones.Count + 2;
        int totalFrames = phDur.Sum();

        var tokens = phones.Select(p => (long)v.PhonemeToken(p.Symbol)).Prepend((long)v.PhonemeToken("SP"))
            .Append((long)v.PhonemeToken("SP")).ToArray();
        // —— linguistic（词模式吃 word_div/word_dur，否则音素模式吃 ph_dur；同 dspitch 按编码器实际输入判）——
        //   base 与（若有混合）目标 token 流各跑一次；结构输入(word_div/word_dur/ph_dur/languages)共享，只换 tokens。
        List<NamedOnnxValue> BuildLing(long[] toks)
        {
            var li = new List<NamedOnnxValue> { NvL("tokens", toks, nTokens) };
            if (v.LinguisticUsesWordBoundary)
            {
                var isVowel = phones.Select(p => p.IsVowel).ToArray();   // 用 phonemizer 定型的类型（dur 表+引擎补全），本预测器 dsdict 可能缺跨语言符号
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

        // —— variance ——
        var model = v.Model("variance");
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_out", encDense),
            NvL("ph_dur", phDur.Select(x => (long)x).ToArray(), nTokens),
            NvF("pitch", pitchSemis, totalFrames),
        };

        // 预测通道（顺序固定 energy→breathiness→voicing→tension，仅 predict_* 为真者参与）。
        //   通道基值 / retake / noise 不进共享条件，按下方重摇逻辑逐趟追加。
        var channels = new List<string>();
        if (cfg.PredictEnergy) channels.Add("energy");
        if (cfg.PredictBreathiness) channels.Add("breathiness");
        if (cfg.PredictVoicing) channels.Add("voicing");
        if (cfg.PredictTension) channels.Add("tension");
        int numVar = channels.Count;

        // 音素混合（帧级，N 槽）：role 模型把 encoder_out_b/blend 列为**必需**输入（导出恒有），故只要模型声明就**总是**喂。
        //   逐槽 r：目标流编码出一行 encoder_out_b[r]（该槽无目标 ⇒ 复用 base）；blend[r]=blendRows[r]。S=mixRows(≥1)；
        //   无混合时 S=1、blend 全 0、目标=base ⇒ 等价不混。
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

        if (model.HasInput("spk_embed"))
        {
            var spk = mix.ToEmbedding(v.GetEmbedding, hidden);
            inputs.Add(NamedOnnxValue.CreateFromTensor("spk_embed",
                new DenseTensor<float>(spk, new[] { 1, totalFrames, hidden })));
        }

        // —— 帧级 seed retake（与 acoustic/pitch 同语义）：seed≠0 的帧重摇、seed=0 的帧保留 take-0。
        //   openvpi 导出保留帧条件 = variance_embed(基值)*~retake ⇒ 保留帧把 take-0 参照喂进各通道口即被钉住；
        //   重摇帧基值被 ~retake 乘 0 屏蔽。retake 2D [1,T,numVar]：单一 seed 轨逐帧广播到全部通道。
        //   · 非混合 → 单趟：基值 0、retake 全 true、噪声按 seed 轨（全 0 即 take-0）。
        //   · 混合 → 先算 take-0 参照（同单趟但噪声 seed=0；输入恒定 → 张量缓存只算一次），
        //     再正式趟（retake=逐帧、基值=参照、噪声按 seed 轨）。参照确定性重算，不记忆上次输出。
        VarianceCurves RunVariance(VarianceCurves bases, bool[] frameMask, uint[]? seeds)
        {
            var ins = new List<NamedOnnxValue>(inputs);
            foreach (var name in channels)
                if (model.HasInput(name))
                    ins.Add(NvF(name, bases[name] ?? new float[totalFrames], totalFrames));
            var retake = new bool[totalFrames * numVar];
            for (int f = 0; f < totalFrames; f++)
                if (frameMask[f])
                    for (int c = 0; c < numVar; c++)
                        retake[f * numVar + c] = true;
            ins.Add(NamedOnnxValue.CreateFromTensor("retake",
                new DenseTensor<bool>(retake, new[] { 1, totalFrames, numVar })));
            if (seeds != null)
                DiffSingerNoise.AddNoise(ins, model, seeds, DiffSingerNoise.StageVariance, totalFrames);
            else
                DiffSingerNoise.AddNoise(ins, model, 0u, DiffSingerNoise.StageVariance, totalFrames);
            var outputs = DiffSingerTensorCache.Run(model, v.ModelHash("variance"), ins, tensorCache);
            float[]? Out(bool predict, string name)
                => predict ? outputs.First(o => o.Name == name).AsTensor<float>().ToArray() : null;
            return new VarianceCurves(
                Out(cfg.PredictEnergy, "energy_pred"),
                Out(cfg.PredictBreathiness, "breathiness_pred"),
                Out(cfg.PredictVoicing, "voicing_pred"),
                Out(cfg.PredictTension, "tension_pred"));
        }

        var allTrue = new bool[totalFrames];
        Array.Fill(allTrue, true);
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
            return RunVariance(default, allTrue, seedPerFrame);

        var reference = RunVariance(default, allTrue, null);   // take-0 参照
        return RunVariance(reference, mask, seedPerFrame);
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

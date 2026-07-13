using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DiffSingerForTuneLab;

// variance 预测器（dsvariance）：忠实移植 OpenUtau DsVariance.Process。
//   linguistic(词模式：word_div/word_dur 由已知 ph_dur 按元音分组) + variance；
//   energy/breathiness/voicing/tension 基值喂 0 + retake 全 true → 全量预测；pitch 输入为半音；输出预测曲线。
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
    //   收 PhonemeSpan（非纯符号）以取音素混合目标（MixSymbol）；blendPerFrame = 逐帧包络比例（与 acoustic 同源），
    //   帧级混合在 role 模型条件级做（跑两次 linguistic：base + 目标流）。
    public static VarianceCurves Predict(
        DiffSingerPredictor? v, IReadOnlyList<PhonemeSpan> phones, int[] phDur,
        float[] pitchSemis, DiffSingerSpeakerMix mix, VoicebankConfig cfg, int steps, uint[] seedPerFrame, bool tensorCache,
        float[]? blendPerFrame = null)
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
                var isVowel = phones.Select(p => v.IsVowel(p.Symbol)).ToArray();
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

        // 音素混合（帧级，条件级）：目标 token 流编码出 encoder_out_b，逐帧 blend 在 role 模型混合。
        var tokensTgt = DiffSingerPredictor.BuildMixTargetTokens(v, phones, tokens, out bool anyMix);
        var encTgt = anyMix && blendPerFrame != null ? Encode(tokensTgt) : null;

        // —— variance ——
        var model = v.Model("variance");
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_out", encDense),
            NvL("ph_dur", phDur.Select(x => (long)x).ToArray(), nTokens),
            NvF("pitch", pitchSemis, totalFrames),
        };

        // 预测通道（顺序固定 energy→breathiness→voicing→tension，仅 predict_* 为真者参与）。
        var channels = new List<string>();
        void Channel(bool predict, string name)
        {
            if (!predict) return;
            channels.Add(name);
            if (model.HasInput(name))
                inputs.Add(NvF(name, new float[totalFrames], totalFrames));   // 基值 0（retake 全 true 故忽略）
        }
        Channel(cfg.PredictEnergy, "energy");
        Channel(cfg.PredictBreathiness, "breathiness");
        Channel(cfg.PredictVoicing, "voicing");
        Channel(cfg.PredictTension, "tension");

        int numVar = channels.Count;
        var retake = new bool[totalFrames * numVar];
        Array.Fill(retake, true);
        inputs.Add(NamedOnnxValue.CreateFromTensor("retake",
            new DenseTensor<bool>(retake, new[] { 1, totalFrames, numVar })));

        // 音素混合（帧级）：目标流 encoder_out_b [1,nTokens,H] + 逐帧 blend [1,totalFrames] 喂 role 模型（条件级、去噪一次）。
        if (encTgt != null && model.HasInput("encoder_out_b"))
        {
            inputs.Add(NamedOnnxValue.CreateFromTensor("encoder_out_b", encTgt));
            inputs.Add(NvF("blend", blendPerFrame!, totalFrames));
        }

        AddAccel(inputs, model, cfg, steps);

        if (model.HasInput("spk_embed"))
        {
            var spk = mix.ToEmbedding(v.GetEmbedding, hidden);
            inputs.Add(NamedOnnxValue.CreateFromTensor("spk_embed",
                new DenseTensor<float>(spk, new[] { 1, totalFrames, hidden })));
        }

        DiffSingerNoise.AddNoise(inputs, model, seedPerFrame, DiffSingerNoise.StageVariance, totalFrames);

        var outputs = DiffSingerTensorCache.Run(model, v.ModelHash("variance"), inputs, tensorCache);
        float[]? Out(bool predict, string name)
            => predict ? outputs.First(o => o.Name == name).AsTensor<float>().ToArray() : null;
        return new VarianceCurves(
            Out(cfg.PredictEnergy, "energy_pred"),
            Out(cfg.PredictBreathiness, "breathiness_pred"),
            Out(cfg.PredictVoicing, "voicing_pred"),
            Out(cfg.PredictTension, "tension_pred"));
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

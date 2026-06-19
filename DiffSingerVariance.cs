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
    // symbols = body 音素（不含 head/tail）；phDur = padded 帧（len=symbols+2）；pitchSemis = totalFrames 半音曲线。
    public static VarianceCurves Predict(
        DiffSingerPredictor? v, IReadOnlyList<string> symbols, int[] phDur,
        float[] pitchSemis, DiffSingerSpeakerMix mix, VoicebankConfig cfg, int steps, bool tensorCache)
    {
        if (v is null || !v.HasModel("variance") || symbols.Count == 0)
            return default;

        int hidden = v.HiddenSize;
        int nTokens = symbols.Count + 2;
        int totalFrames = phDur.Sum();

        var tokens = symbols.Select(s => (long)v.PhonemeToken(s)).Prepend((long)v.PhonemeToken("SP"))
            .Append((long)v.PhonemeToken("SP")).ToArray();
        var langs = symbols.Select(s => v.LangId(PhonemeLanguage(s))).Prepend(0L).Append(0L).ToArray();
        var isVowel = symbols.Select(v.IsVowel).ToArray();

        // —— linguistic（词模式）——
        var (wordDiv, wordDur) = DiffSingerFrames.PaddedWordDivAndDur(isVowel, phDur);
        var lingInputs = new List<NamedOnnxValue>
        {
            NvL("tokens", tokens, nTokens),
            NvL("word_div", wordDiv, wordDiv.Length),
            NvL("word_dur", wordDur, wordDur.Length),
        };
        if (v.Linguistic.InputMetadata.ContainsKey("languages"))
            lingInputs.Add(NvL("languages", langs, nTokens));
        var lingOut = DiffSingerTensorCache.Run(v.Linguistic, v.LinguisticHash, lingInputs, tensorCache);
        var enc = lingOut.First(o => o.Name == "encoder_out").AsTensor<float>();
        var encDense = new DenseTensor<float>(enc.ToArray(), enc.Dimensions.ToArray());

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
            if (model.InputMetadata.ContainsKey(name))
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

        AddAccel(inputs, model, cfg, steps);

        if (model.InputMetadata.ContainsKey("spk_embed"))
        {
            var spk = mix.ToEmbedding(v.GetEmbedding, hidden);
            inputs.Add(NamedOnnxValue.CreateFromTensor("spk_embed",
                new DenseTensor<float>(spk, new[] { 1, totalFrames, hidden })));
        }

        var outputs = DiffSingerTensorCache.Run(model, v.ModelHash("variance"), inputs, tensorCache);
        float[]? Out(bool predict, string name)
            => predict ? outputs.First(o => o.Name == name).AsTensor<float>().ToArray() : null;
        return new VarianceCurves(
            Out(cfg.PredictEnergy, "energy_pred"),
            Out(cfg.PredictBreathiness, "breathiness_pred"),
            Out(cfg.PredictVoicing, "voicing_pred"),
            Out(cfg.PredictTension, "tension_pred"));
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

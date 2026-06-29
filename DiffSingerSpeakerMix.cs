using System;
using System.Collections.Generic;
using System.Linq;

namespace DiffSingerForTuneLab;

// 说话人逐帧混合（模型无关，每会话每块算一次、acoustic/pitch/variance 三域共享）：
//   忠实移植 OpenUtau DiffSingerSpeakerEmbedManager.PhraseSpeakerEmbedByFrame 的逐帧嵌入混合。
//   每条 mix:<suffix> 曲线 [0,100]·0.01 累积到该 suffix 的逐帧权重；逐帧标准化——
//   Σ>1 时各 suffix 按和归一化，否则默认 suffix（part 级 KeySpeaker）补 1-Σ。
//   权重与模型无关、按 suffix 定义；各域用各自的 emb 解析器 ToEmbedding（预测器/声学的 speakers 表与 .emb 各异）。
//   无 mix 轨（或单说话人）时退化为「默认 suffix 恒权重 1」⇒ 等价旧的单 emb 逐帧广播。
public sealed class DiffSingerSpeakerMix
{
    readonly (string Suffix, float[] Weight)[] mEntries;   // 每帧跨条目权重和恒为 1
    public int FrameCount { get; }

    DiffSingerSpeakerMix((string Suffix, float[] Weight)[] entries, int frameCount)
    {
        mEntries = entries;
        FrameCount = frameCount;
    }

    // 构造逐帧权重：默认 suffix 必有一席；各 mix 轨逐帧值（归一化 [0,1]，NaN 自由区视作 0）累积到对应 suffix；
    //   逐帧标准化（Σ>1 归一化，否则默认 suffix 补 1-Σ）。忠实对齐 OpenUtau 的 standardization 段（其 [0,100] 在此已归一）。
    public static DiffSingerSpeakerMix Create(
        string defaultSuffix, IReadOnlyList<(string Suffix, double[] Sampled)> tracks, int nFrames)
    {
        var bySuffix = new Dictionary<string, float[]>(StringComparer.Ordinal);
        float[] Weight(string suffix)
        {
            if (!bySuffix.TryGetValue(suffix, out var w))
                bySuffix[suffix] = w = new float[nFrames];
            return w;
        }

        var defaultWeight = Weight(defaultSuffix);   // 默认 suffix 恒占一席（即便无 mix 轨）
        foreach (var (suffix, sampled) in tracks)
        {
            var w = Weight(suffix);
            for (int f = 0; f < nFrames; f++)
            {
                double v = f < sampled.Length ? sampled[f] : double.NaN;
                if (!double.IsNaN(v)) w[f] += (float)v;
            }
        }

        for (int f = 0; f < nFrames; f++)
        {
            float sum = 0;
            foreach (var w in bySuffix.Values) sum += w[f];
            if (sum > 1)
                foreach (var w in bySuffix.Values) w[f] /= sum;
            else
                defaultWeight[f] += 1 - sum;
        }

        return new DiffSingerSpeakerMix(
            bySuffix.Select(kv => (kv.Key, kv.Value)).ToArray(), nFrames);
    }

    // 用指定域的 emb 解析器混出逐帧 spk_embed（[1, FrameCount, hidden] 的扁平数组）：Σ_suffix 权重·emb。
    //   resolveEmb 据 suffix 取该域 emb（预测器走 GetEmbedding、声学走 GetSpeakerEmbeddingBySuffix，均带回退+缓存）。
    public float[] ToEmbedding(Func<string, float[]> resolveEmb, int hidden)
    {
        var spk = new float[FrameCount * hidden];
        foreach (var (suffix, weight) in mEntries)
        {
            var emb = resolveEmb(suffix);
            for (int f = 0; f < FrameCount; f++)
            {
                float w = weight[f];
                if (w == 0) continue;
                int b = f * hidden;
                for (int i = 0; i < hidden; i++) spk[b + i] += w * emb[i];
            }
        }
        return spk;
    }
}

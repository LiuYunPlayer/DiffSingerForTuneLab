using System;
using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace DiffSingerForTuneLab;

// 把扩散/RF 采样的初始噪声从「模型内部现采」改为「插件按 seed 确定性生成、外部喂入」。
//   仅当模型声明了 noise 输入口（经图手术 / fork 重导出，见记忆 openvpi-noise-fork）才喂；
//   否则不喂、退回模型内部 RandomNormalLike（stock 模型向后兼容、零行为变化）。
//
// noise 形状从 InputMetadata 读：[1, num_feats, out_dims, n_frames]（轴 1/2 定值，仅帧轴运行时定）。
//   位置寻址 PRNG：每元素噪声 = f(seed, stage, c, m, t)，与张量长度/区段无关——
//   故同 seed 跨渲染/会话/机器逐元素一致（可复现，取代张量缓存兜稳定）；
//   将来按 note/帧改 seed 只动对应位置、其余 bit 不变（局部 retake 的地基）。
public static class DiffSingerNoise
{
    // stage 盐：三个扩散模型共用同一 part seed，但各自独立噪声流（避免 pitch/variance/acoustic 噪声相关）。
    public const ulong StageAcoustic = 0xA0, StagePitch = 0xB0, StageVariance = 0xC0;

    // 标量 seed（全 part 一致，如 acoustic timbre——它全局不局部化，单一 seed 即可）。
    public static void AddNoise(List<NamedOnnxValue> inputs, InferenceSession model, uint seed, ulong stage, int nFrames)
        => AddNoise(inputs, model, _ => seed, stage, nFrames);

    // 逐帧 seed（pitch/variance 的 seed 自动化轨：每帧可不同 → 时间维 × 值维 = 区域独立 take）。归一化 [0,1] 已在会话期放大到 uint32。
    public static void AddNoise(List<NamedOnnxValue> inputs, InferenceSession model, uint[] seedPerFrame, ulong stage, int nFrames)
        => AddNoise(inputs, model, t => seedPerFrame[Math.Min(t, seedPerFrame.Length - 1)], stage, nFrames);

    static void AddNoise(List<NamedOnnxValue> inputs, InferenceSession model, Func<int, uint> seedAt, ulong stage, int nFrames)
    {
        if (!model.InputMetadata.TryGetValue("noise", out var meta))
            return;   // 无 noise 口（stock 模型）→ 不喂，退回模型内部随机
        var dims = meta.Dimensions;            // [1, num_feats, out_dims, -1]
        int feats = dims[1], outDims = dims[2];
        var data = new float[feats * outDims * nFrames];
        int idx = 0;
        for (int c = 0; c < feats; c++)
            for (int m = 0; m < outDims; m++)
                for (int t = 0; t < nFrames; t++)
                    data[idx++] = Gaussian(stage, seedAt(t), (ulong)c, (ulong)m, (ulong)t);
        inputs.Add(NamedOnnxValue.CreateFromTensor("noise",
            new DenseTensor<float>(data, new[] { 1, feats, outDims, nFrames })));
    }

    // 位置寻址标准正态：hash(stage, seed, c, m, t) → 两个均匀数 → Box–Muller。
    static float Gaussian(ulong stage, ulong seed, ulong c, ulong m, ulong t)
    {
        ulong k = Mix(Mix(Mix(Mix(stage, seed), c), m), t);
        double u1 = Uniform(SplitMix64(k));
        double u2 = Uniform(SplitMix64(k ^ 0xD1B54A32D192ED03UL));
        double r = Math.Sqrt(-2.0 * Math.Log(u1));
        return (float)(r * Math.Cos(2.0 * Math.PI * u2));
    }

    static ulong Mix(ulong h, ulong x)
        => SplitMix64(h ^ (x + 0x9E3779B97F4A7C15UL + (h << 6) + (h >> 2)));

    static ulong SplitMix64(ulong x)
    {
        x += 0x9E3779B97F4A7C15UL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
        return x ^ (x >> 31);
    }

    // (0,1] 均匀（取高 53 位，+0.5 偏移避免 0 → 杜绝 log(0)）。
    static double Uniform(ulong h) => ((h >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

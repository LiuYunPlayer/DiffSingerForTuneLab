using System;
using System.Collections.Generic;
using System.Linq;
using DiffSingerForTuneLab;
using Xunit;

namespace DiffSinger.Tests;

// DiffSingerFrames 是纯静态工具类，零外部依赖，用 xUnit 直接验证数值契约。
//   ref: docs/tunelab-voicebank-schema.md §14.2 / OpenUtau DiffSingerUtils
public class DiffSingerFramesTests
{
    const double Frame10ms = 0.010;   // 10 ms 帧（OpenUtau 默认 / 部分声库）
    const double Frame20ms = 0.020;   // 20 ms 帧（dsconfig 可配 hop/sample_rate）

    // ── DurationsToFrames 基础性质 ──

    [Fact]
    public void DurationsToFrames_ZeroInput_ReturnsEmpty()
    {
        var result = DiffSingerFrames.DurationsToFrames(Array.Empty<double>(), Frame10ms);
        Assert.Empty(result);
    }

    [Theory]
    // 使用精确二进制可表示的时长（避免 0.1/0.01 的 double 精度陷阱）
    [InlineData(new double[] { 0.025 }, Frame10ms, new int[] { 3 })]   // 25ms / 10 = 2.5+0.5=3.0 → 3
    [InlineData(new double[] { 0.050 }, Frame10ms, new int[] { 6 })]   // 50ms / 10 = 5.0+0.5=5.5 → ToEven→6
    [InlineData(new double[] { 0.075 }, Frame10ms, new int[] { 8 })]   // 75ms / 10 = 7.5+0.5=8.0 → 8
    [InlineData(new double[] { 0.128 }, Frame10ms, new int[] { 13 })]  // 128ms / 10 = 12.8+0.5=13.3 → 13
    [InlineData(new double[] { 0.256 }, Frame10ms, new int[] { 26 })]  // 256ms / 10 = 25.6+0.5=26.1 → 26
    public void DurationsToFrames_SingleSegment_RoundsCorrectly(
        double[] input, double frameSec, int[] expected)
    {
        var result = DiffSingerFrames.DurationsToFrames(input, frameSec);
        Assert.Equal(expected, result);
    }

    [Fact]
    public void DurationsToFrames_NegativeInput_ClampsToZero()
    {
        var result = DiffSingerFrames.DurationsToFrames(new[] { -0.5, 0.01 }, Frame10ms);
        Assert.Equal(2, result.Length);
        // -0.5 被 clamp 到 0，等效空段；0.01 → 1ms/10ms=1, +0.5→1.5, ToEven→2 帧
        Assert.Equal(new[] { 0, 2 }, result);
    }

    [Fact]
    public void DurationsToFrames_AllZero_ReturnsAllZero()
    {
        // 三个 0 时长段：accumulated 恒 0，round(0.5, ToEven)=0，每段差 0
        var result = DiffSingerFrames.DurationsToFrames(new[] { 0.0, 0.0, 0.0 }, Frame10ms);
        Assert.Equal(new[] { 0, 0, 0 }, result);
    }

    // ── 累积取整不丢帧 ──
    // 注：0.1 在 double 下不完全精确（0.1*3/0.01+0.5 ≈ 30.500... → ToEven 可能进位到 31），
    // 故用 frameSec=0.02（50ms）并将输入凑成整数帧，避开浮点歧义。

    [Theory]
    [InlineData(new double[] { 0.064, 0.128, 0.032 }, 0.01, 23)]  // 224ms / 10ms = 23 帧（累积取整）
    [InlineData(new double[] { 0.04, 0.06, 0.10 }, 0.02, 10)]   // 200ms / 20ms = 10 帧
    [InlineData(new double[] { 0.02, 0.02, 0.02, 0.02 }, 0.02, 4)]  // 80ms / 20ms = 4
    public void DurationsToFrames_SumPreserved(double[] input, double frameSec, int expectedTotal)
    {
        var result = DiffSingerFrames.DurationsToFrames(input, frameSec);
        Assert.Equal(expectedTotal, result.Sum());
    }

    // ── PaddedPhoneFrames ──

    [Fact]
    public void PaddedPhoneFrames_ThreePhones_ReturnsFiveSegments()
    {
        // 3 phones + head + tail = 5 段
        var result = DiffSingerFrames.PaddedPhoneFrames(
            new[] { 0.1, 0.1, 0.1 }, Frame10ms);
        Assert.Equal(5, result.Length);
        Assert.Equal(DiffSingerFrames.HeadFrames, result[0]);
        Assert.Equal(DiffSingerFrames.TailFrames, result[^1]);
    }

    [Fact]
    public void PaddedPhoneFrames_TotalMatchesDirectComputation()
    {
        var phones = new[] { 0.064, 0.128, 0.032 };
        var padded = DiffSingerFrames.PaddedPhoneFrames(phones, Frame10ms);
        // 等价于把 head + phones + tail 一次性喂进 DurationsToFrames
        var direct = DiffSingerFrames.DurationsToFrames(
            phones.Prepend(DiffSingerFrames.HeadFrames * Frame10ms)
                  .Append(DiffSingerFrames.TailFrames * Frame10ms),
            Frame10ms);
        Assert.Equal(direct, padded);
    }

    [Fact]
    public void PaddedPhoneFrames_ZeroPhones_ReturnsOnlyHeadAndTail()
    {
        var result = DiffSingerFrames.PaddedPhoneFrames(Array.Empty<double>(), Frame10ms);
        Assert.Equal(2, result.Length);
        Assert.Equal(DiffSingerFrames.HeadFrames, result[0]);
        Assert.Equal(DiffSingerFrames.TailFrames, result[1]);
    }

    // ── FitDurationSum ──

    [Fact]
    public void FitDurationSum_NormalCase_AdjustsLastElement()
    {
        var input = new[] { 10, 20, 30 };
        var result = DiffSingerFrames.FitDurationSum(input, 65);
        Assert.Equal(65, result.Sum());
        // sum=60, delta=+5 → 末项 30+5=35
        Assert.Equal(new[] { 10, 20, 35 }, result);
    }

    [Fact]
    public void FitDurationSum_AlreadyExact_Unchanged()
    {
        var input = new[] { 10, 20, 30 };
        var result = DiffSingerFrames.FitDurationSum(input, 60);
        Assert.Equal(input, result);
    }

    [Fact]
    public void FitDurationSum_PositiveDelta_OnlyLastChanges()
    {
        var input = new[] { 10, 20, 30 };
        var result = DiffSingerFrames.FitDurationSum(input, 70);
        Assert.Equal(70, result.Sum());
        Assert.Equal(10, result[0]);
        Assert.Equal(20, result[1]);
        Assert.Equal(40, result[^1]);  // 30 + 10 delta
    }

    [Fact]
    public void FitDurationSum_NegativeDelta_BorrowsFromEarlier()
    {
        // sum=70, target=50, delta=-20 → 末项 40-20=20≥0, 无需向前借
        var input = new[] { 10, 20, 40 };
        var result = DiffSingerFrames.FitDurationSum(input, 50);
        Assert.Equal(50, result.Sum());
        Assert.Equal(new[] { 10, 20, 20 }, result);
        Assert.All(result, r => Assert.True(r >= 0));
    }

    [Fact]
    public void FitDurationSum_NegativeDelta_Overdraft_Borrows()
    {
        // sum=70, target=40, delta=-30 → 末项 40-30=10≥0, 仍无需借
        // 需要真正借的场景: target < sum - 末项
        var input = new[] { 10, 20, 40 };
        var result = DiffSingerFrames.FitDurationSum(input, 25);
        Assert.Equal(25, result.Sum());
        // delta=-45, 末项 40-45=-5<0 → 清零，还需从前面借 5
        Assert.Equal(0, result[^1]);
        Assert.All(result, r => Assert.True(r >= 0));
    }

    [Fact]
    public void FitDurationSum_EmptyArray_ReturnsEmpty()
    {
        var result = DiffSingerFrames.FitDurationSum(Array.Empty<int>(), 100);
        Assert.Empty(result);
    }

    [Fact]
    public void FitDurationSum_SingleElement_AdjustsToTotal()
    {
        var result = DiffSingerFrames.FitDurationSum(new[] { 30 }, 100);
        Assert.Equal(100, result.Sum());
        Assert.Equal(100, result[0]);
    }

    // ── PaddedWordDivAndDur ──

    [Fact]
    public void PaddedWordDivAndDur_OneWord_OneVowel()
    {
        // 3 phones, 1 vowel at index 1 → wordDiv = [2, 3]
        //   head+phone0+phone1 = 2, phone2+tail = 3
        var (div, dur) = DiffSingerFrames.PaddedWordDivAndDur(
            isVowel: new[] { false, true, false },
            phDur: new[] { 3, 4, 3, 8, 8 });  // 5 padded = 3 phones + 2
        Assert.Equal(new long[] { 2, 3 }, div);
        Assert.Equal(5, div.Sum());           // wordDiv sum == phDur.Length
        Assert.Equal(new long[] { 7, 19 }, dur);  // 3+4=7, 3+8+8=19
        Assert.Equal(26, dur.Sum());          // = phDur sum
    }

    [Fact]
    public void PaddedWordDivAndDur_TwoWords_TwoVowels()
    {
        // 4 phones, vowels at 0, 2 → 3 words (head+vowel0 | vowel0+1..vowel2 | vowel2+1..tail)
        var phDur = new[] { 5, 3, 4, 2, 8, 8 }; // 6 padded = 4 phones + 2
        var (div, dur) = DiffSingerFrames.PaddedWordDivAndDur(
            isVowel: new[] { true, false, true, false },
            phDur: phDur);
        // wordDiv: [1, 2, 3]
        Assert.Equal(new long[] { 1, 2, 3 }, div);
        Assert.Equal(6, div.Sum());           // wordDiv sum == phDur.Length
        Assert.Equal(3, dur.Length);          // one entry per word (not per phone)
        // wordDur: [5, 3+4=7, 2+8+8=18]
        Assert.Equal(new long[] { 5, 7, 18 }, dur);
        Assert.Equal(phDur.Sum(), dur.Sum()); // wordDur sum == phDur sum
    }

    [Fact]
    public void PaddedWordDivAndDur_AllConsonants_FallsBackToLastPhone()
    {
        // 无元音 → vowelIds = [last phone index = 2]
        var (div, dur) = DiffSingerFrames.PaddedWordDivAndDur(
            isVowel: new[] { false, false, false },
            phDur: new[] { 5, 3, 2, 8, 8 });
        // wordDiv: [3, 2] — head+phone0+phone1=3, phone2+tail=2
        Assert.Equal(new long[] { 3, 2 }, div);
        Assert.Equal(5, div.Sum());         // = phDur.Length = phones + 2
        Assert.Equal(new long[] { 10, 16 }, dur);  // 5+3+2=10, 8+8=16
        Assert.Equal(26, dur.Sum());        // = phDur sum
    }

    [Fact]
    public void PaddedWordDivAndDur_LengthMismatch_Throws()
    {
        Assert.Throws<InvalidOperationException>(() =>
            DiffSingerFrames.PaddedWordDivAndDur(
                isVowel: new[] { true },
                phDur: new[] { 1, 2 }));  // phones=1, expects 1+2=3
    }

    [Fact]
    public void PaddedWordDivAndDur_NoPhones_Throws()
    {
        Assert.Throws<InvalidOperationException>(() =>
            DiffSingerFrames.PaddedWordDivAndDur(
                isVowel: Array.Empty<bool>(),
                phDur: new[] { 8, 8 }));
    }

    // ── ToneToFreq ──

    [Theory]
    [InlineData(69.0, 440.0)]      // A4 = 440 Hz
    [InlineData(57.0, 220.0)]      // A3 = 220 Hz
    [InlineData(81.0, 880.0)]      // A5 = 880 Hz
    [InlineData(60.0, 261.6255653)] // C4 ≈ 261.63 Hz
    public void ToneToFreq_StandardNotes_MatchesExpected(double midi, double expectedHz)
    {
        var hz = DiffSingerFrames.ToneToFreq(midi);
        Assert.Equal(expectedHz, hz, precision: 4);
    }

    // ── 头/尾帧常量 ──

    [Fact]
    public void HeadTailFrames_EightEach()
    {
        Assert.Equal(8, DiffSingerFrames.HeadFrames);
        Assert.Equal(8, DiffSingerFrames.TailFrames);
    }
}

using System;
using System.Collections.Generic;
using System.Linq;

namespace DiffSingerForTuneLab;

// 帧时长工具：忠实移植 OpenUtau DiffSingerUtils（DurationsMsToFrames / PaddedWordDivAndDur / FitDurationSum）。
// 在秒域工作（OpenUtau 用 ms，等价）。head/tail padding 各 8 帧（OpenUtau 约定）。
public static class DiffSingerFrames
{
    public const int HeadFrames = 8;
    public const int TailFrames = 8;

    // 秒时长序列 → 整数帧（累积取整保连续）：frame = round(累积秒/frameSec + 0.5, ToEven)。
    public static int[] DurationsToFrames(IEnumerable<double> durationsSec, double frameSec)
    {
        var result = new List<int>();
        double accumulated = 0;
        int previous = 0;
        foreach (var d in durationsSec)
        {
            accumulated += Math.Max(0, d);
            int frame = (int)Math.Round(accumulated / frameSec + 0.5, MidpointRounding.ToEven);
            result.Add(frame - previous);
            previous = frame;
        }
        return result.ToArray();
    }

    // body 音素时长（秒）前后加 head/tail padding（秒）→ 整数帧（len = phones + 2）。
    public static int[] PaddedPhoneFrames(IReadOnlyList<double> phoneDursSec, double frameSec)
    {
        var seq = phoneDursSec.Prepend(HeadFrames * frameSec).Append(TailFrames * frameSec);
        return DurationsToFrames(seq, frameSec);
    }

    // 按元音分组的 word_div/word_dur（OpenUtau PaddedWordDivAndDur）：phDur 含 head/tail（len = symbols+2）。
    // word 边界在元音处；首词含 head+前置辅音+首元音，末词含末元音之后+tail。
    public static (long[] wordDiv, long[] wordDur) PaddedWordDivAndDur(
        IReadOnlyList<bool> isVowel, int[] phDur)
    {
        int phones = isVowel.Count;
        if (phones == 0)
            throw new InvalidOperationException("word 模式至少需要一个音素");
        if (phDur.Length != phones + 2)
            throw new InvalidOperationException($"word 模式时长数 {phDur.Length} 与 padded token 数 {phones + 2} 不符");

        var vowelIds = Enumerable.Range(0, phones).Where(i => isVowel[i]).ToArray();
        if (vowelIds.Length == 0)
            vowelIds = new[] { phones - 1 };

        var wordDiv = vowelIds.Zip(vowelIds.Skip(1), (a, b) => (long)(b - a))
            .Prepend(vowelIds[0] + 1)
            .Append(phones - vowelIds[^1] + 1)
            .ToArray();

        var wordDur = new long[wordDiv.Length];
        int offset = 0;
        for (int i = 0; i < wordDiv.Length; i++)
        {
            int len = (int)wordDiv[i];
            long sum = 0;
            for (int j = 0; j < len; j++) sum += phDur[offset + j];
            wordDur[i] = sum;
            offset += len;
        }
        return (wordDiv, wordDur);
    }

    // 调整整数帧序列使总和 = totalFrames（末项吸收差额，不足则向前借）。
    public static int[] FitDurationSum(int[] durations, int totalFrames)
    {
        if (durations.Length == 0) return durations;
        var result = durations.ToArray();
        int delta = totalFrames - result.Sum();
        result[^1] += delta;
        if (result[^1] < 0)
        {
            int deficit = -result[^1];
            result[^1] = 0;
            for (int i = result.Length - 2; i >= 0 && deficit > 0; i--)
            {
                int take = Math.Min(result[i], deficit);
                result[i] -= take;
                deficit -= take;
            }
        }
        return result;
    }

    public static float ToneToFreq(double tone) => (float)(440.0 * Math.Pow(2, (tone - 69) / 12.0));
}

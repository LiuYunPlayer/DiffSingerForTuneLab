using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TuneLab.Foundation;
using TuneLab.SDK;
using static DiffSingerForTuneLab.DiffSingerDeclarations;

namespace DiffSingerForTuneLab;

// 一条 part 的合成会话。
// 关键设计：
//   · 区段式重渲染：修改某个音素时，仅以该音素为中心前后各扩展一个音素作为「重渲染区段」，
//     渲染后将新 mel 通过频谱过渡拼贴到原序列的 mel 谱上，避免整个序列被改变。
//   · pitch 锁定：用户修改 pitch 曲线后，自动音高预测（dspitch）被锁定不再重新生成，
//     仅缓存的预测作为 NaN 自由区的回退。除非用户选择 Retake 或 RedrawPitch。
//   · dur 忠于 UI：音素时间线由用户界面（note 时长 / 钉死音素）决定，首次渲染后只有显式请求才重新 phonemize。
public sealed class DiffSingerSynthesisSession : ISynthesisSession
{
    readonly VoicebankConfig mConfig;
    readonly ISynthesisContext mContext;
    readonly string mVoiceId;
    readonly DiffSingerModelCache mModelCache;
    readonly int mSamplingSteps;

    internal string VoiceId => mVoiceId;

    enum RenderMode { Normal, Retake }
    volatile RenderMode mRenderMode;
    volatile bool mRenderModeConsumed;

    readonly OrderedMap<string, AutomationConfig> mAutomationConfigs;
    readonly OrderedMap<string, AutomationConfig> mReadbackConfigs;

    readonly IDisposable mNotesSubscription;
    readonly List<ILiveAutomation> mSubscribedAutomations = new();
    readonly Dictionary<ILiveNote, NoteHandlers> mNoteHandlers = new();
    readonly List<Piece> mPieces = new();
    bool mNeedResegment;

    // 缓存有效标志
    bool mHasValidCache;

    // 受影响的 time range（来自 OnRangeModified 或 note 修改），供 mel 拼贴使用
    double mAffectedStartTime = double.NaN;
    double mAffectedEndTime = double.NaN;

    public DiffSingerSynthesisSession(VoicebankConfig config, ISynthesisContext context,
        string voiceId, DiffSingerModelCache modelCache, int samplingSteps)
    {
        mConfig = config;
        mContext = context;
        mVoiceId = voiceId;
        mModelCache = modelCache;
        mSamplingSteps = samplingSteps;

        mAutomationConfigs = BuildAutomationConfigs(config);
        mReadbackConfigs = BuildReadbackConfigs(config);

        mNotesSubscription = NotifiableExtensions.WhenAny(context.Notes, SubscribeNote, UnsubscribeNote);
        context.Notes.ItemAdded += OnNotesStructureChanged;
        context.Notes.ItemRemoved += OnNotesStructureChanged;
        context.PartProperties.Modified += MarkAllDirtyAndResegment;
        context.Pitch.RangeModified += OnRangeModified;
        context.PitchDeviation.RangeModified += OnRangeModified;
        context.Committed += OnCommitted;

        foreach (var key in mAutomationConfigs.Keys)
            if (context.TryGetAutomation(key, out var automation))
            {
                automation.RangeModified += OnRangeModified;
                mSubscribedAutomations.Add(automation);
            }

        mNeedResegment = true;
    }

    public string DefaultLyric => "a";

    // 根据区间找出应处理的 piece 集合（NaN 表示全曲）
    IEnumerable<Piece> PiecesInRange(double start, double end)
    {
        if (double.IsNaN(start) || double.IsNaN(end))
            return mPieces;
        return mPieces.Where(p => p.StartTime < end && p.EndTime > start);
    }

    internal void RequestRetake()
    {
        mHasValidCache = false;
        ClearPieceCaches(PiecesInRange(mAffectedStartTime, mAffectedEndTime));
        StatusChanged?.Invoke();
    }

    internal void RequestRetakeScoped(double scopeStart, double scopeEnd)
    {
        mHasValidCache = false;
        SetAffectedRange(scopeStart, scopeEnd);
        ClearPieceCaches(PiecesInRange(scopeStart, scopeEnd));
        StatusChanged?.Invoke();
    }

    internal void RequestRedrawPitch()
    {
        foreach (var piece in PiecesInRange(mAffectedStartTime, mAffectedEndTime))
        {
            if (piece.CachedPhones == null) continue;
            piece.Dirty = true; piece.Failed = false;
            piece.CachedPitchPrediction = null;
            piece.RedrawPitchRequested = true;
        }
        StatusChanged?.Invoke();
    }

    internal void RequestRedrawPitchScoped(double scopeStart, double scopeEnd)
    {
        SetAffectedRange(scopeStart, scopeEnd);
        foreach (var piece in PiecesInRange(scopeStart, scopeEnd))
        {
            if (piece.CachedPhones == null) continue;
            piece.Dirty = true; piece.Failed = false;
            piece.CachedPitchPrediction = null;
            piece.RedrawPitchRequested = true;
        }
        StatusChanged?.Invoke();
    }

    void ClearPieceCaches(IEnumerable<Piece> pieces)
    {
        foreach (var piece in pieces)
        {
            piece.Dirty = true; piece.Failed = false;
            piece.CachedPitchPrediction = null;
            piece.CachedVarianceCurves = default;
            piece.CachedPhones = null;
            piece.CachedMel = null; piece.CachedMelDims = null;
            piece.CachedAudio = null;
            piece.CachedPitchReadback = null;
            piece.CachedVarianceReadback = new Dictionary<string, IReadOnlyList<Point>>();
            piece.RedrawPitchRequested = false;
        }
    }

    public SynthesisSegment? GetNextSegment(double startTime, double endTime)
        => FindNextDirtyPiece(startTime, endTime) is { } p ? new SynthesisSegment(p.StartTime, p.EndTime) : null;

    Piece? FindNextDirtyPiece(double startTime, double endTime)
    {
        if (mNeedResegment) Resegment();
        foreach (var piece in mPieces)
        {
            if (!piece.Dirty || piece.Failed || piece.Synthesizing) continue;
            if (piece.EndTime < startTime || piece.StartTime > endTime) continue;
            return piece;
        }
        return null;
    }

    public async Task SynthesizeNext(SynthesisSegment segment, CancellationToken cancellation = default)
    {
        if (FindNextDirtyPiece(segment.StartTime, segment.EndTime) is not { } piece)
            return;

        var snapshot = mContext.GetSnapshot(piece.Notes, piece.Notes[0].StartTime.Value, piece.Notes.Max(n => n.EndTime.Value));
        piece.Dirty = false;
        piece.Synthesizing = true;
        piece.Progress = 0;
        StatusChanged?.Invoke();

        var report = new Progress<double>(p => { piece.Progress = p; StatusChanged?.Invoke(); });
        try
        {
            var rendered = await Task.Run(() => Render(snapshot, piece.Notes, piece, report, cancellation), CancellationToken.None);
            if (rendered != null && mPieces.Contains(piece))
            {
                int rate = rendered.SampleRate;

                // 区段式音频拼贴：仅替换受影响的 time range，其余保持旧音频不变
                var stitchedAudio = StitchAudio(rendered.Audio, piece.CachedAudio,
                    rendered.StartTime, rate, mAffectedStartTime, mAffectedEndTime);

                // 缓存旧音频供下次拼贴
                piece.CachedAudio = stitchedAudio;

                piece.Segment?.Dispose();
                piece.Segment = mContext.CreateAudioSegment((long)(rendered.StartTime * rate), stitchedAudio.Length, rate);
                piece.Segment.Write(0, stitchedAudio);
                piece.Segment.Commit();
                piece.Phonemes = rendered.Phonemes;

                // 回显曲线也做区间拼贴：未修改区间的 pitch/tension 与缓存的旧曲线一致
                piece.PitchReadback = StitchPoints(rendered.PitchReadback, piece.CachedPitchReadback,
                    mAffectedStartTime, mAffectedEndTime);
                piece.CachedPitchReadback = piece.PitchReadback;

                if (rendered.VarianceReadback.Count > 0)
                {
                    var stitchedVar = new Dictionary<string, IReadOnlyList<Point>>();
                    foreach (var kvp in rendered.VarianceReadback)
                    {
                        piece.CachedVarianceReadback.TryGetValue(kvp.Key, out var oldVar);
                        stitchedVar[kvp.Key] = StitchPoints(kvp.Value, oldVar,
                            mAffectedStartTime, mAffectedEndTime);
                    }
                    piece.VarianceReadback = stitchedVar;
                    piece.CachedVarianceReadback = stitchedVar;
                }
            }
        }
        catch (Exception ex)
        {
            piece.Failed = true;
            piece.Error = ex.Message;
            TuneLabContext.Global.GetLogger().Warning($"DiffSinger：合成失败 [{piece.StartTime:F2}s]：{ex}");
        }
        finally
        {
            piece.Synthesizing = false;
            StatusChanged?.Invoke();
        }
    }

    // —— 音频拼贴 ——
    // 将新音频的 affected 区间（前后各扩展 3 帧过渡）替换到旧音频中。
    // 若旧音频不存在（首次渲染）则直接返回新音频。
    static float[] StitchAudio(float[] newAudio, float[]? oldAudio,
        double renderStartSec, int sampleRate,
        double affectedStart, double affectedEnd)
    {
        if (oldAudio == null || oldAudio.Length == 0)
            return newAudio;
        if (double.IsNaN(affectedStart) || double.IsNaN(affectedEnd))
            return newAudio; // 无明确 affected 区间时全量替换
        if (oldAudio.Length != newAudio.Length)
            return newAudio; // 长度不同无法拼贴

        // 计算 affected 区间的采样点范围（前后扩展 3 帧 = 3 * hop_size 采样点）
        int hop = 512; // DiffSinger 标准 hop_size
        int fadeSamples = 3 * hop;
        int startSample = Math.Max(0, (int)((affectedStart - renderStartSec) * sampleRate) - fadeSamples);
        int endSample = Math.Min(newAudio.Length, (int)((affectedEnd - renderStartSec) * sampleRate) + fadeSamples);

        if (startSample >= endSample)
            return newAudio;

        var result = new float[oldAudio.Length];
        Array.Copy(oldAudio, result, oldAudio.Length);

        // 拷贝 affected 区间的新音频
        int copyLen = endSample - startSample;
        Array.Copy(newAudio, startSample, result, startSample, copyLen);

        // 前过渡区：第一帧（线性渐入）
        int fadeLen = Math.Min(fadeSamples, copyLen / 2);
        for (int i = 0; i < fadeLen; i++)
        {
            float t = (float)(i + 1) / (fadeLen + 1);
            int idx = startSample + i;
            if (idx >= 0 && idx < result.Length)
                result[idx] = oldAudio[idx] * (1 - t) + newAudio[idx] * t;
        }

        // 后过渡区：最后一帧（线性渐出）
        for (int i = 0; i < fadeLen; i++)
        {
            float t = (float)(i + 1) / (fadeLen + 1);
            int idx = endSample - 1 - i;
            if (idx >= 0 && idx < result.Length)
                result[idx] = oldAudio[idx] * (1 - t) + newAudio[idx] * t;
        }

        return result;
    }

    // —— 推理链（worker，只读冻结快照）——

    // 推理结果
    sealed record RenderResultEx(float[] Audio, double StartTime, int SampleRate,
        List<SynthesizedPhoneme> Phonemes, List<Point> PitchReadback,
        Dictionary<string, IReadOnlyList<Point>> VarianceReadback,
        int[] MelDims, float[] Mel);

    RenderResultEx? Render(SynthesisSnapshot snapshot, IReadOnlyList<ILiveNote> origins, Piece piece,
        IProgress<double>? progress, CancellationToken cancellation)
    {
        var notes = snapshot.Notes;
        if (notes.Count == 0 || cancellation.IsCancellationRequested)
            return null;

        var models = mModelCache.GetOrLoad(mVoiceId, mConfig);
        int hop = models.HopSize, sr = models.SampleRate, hidden = models.HiddenSize;
        double frameSec = (double)hop / sr;
        int head = DiffSingerFrames.HeadFrames;
        int numMelBins = models.NumMelBins;

        string partLang = snapshot.PartProperties.GetString(KeyLanguage, string.Empty);
        string speaker = snapshot.PartProperties.GetString(KeySpeaker, mConfig.Speakers.Count > 0 ? mConfig.Speakers[0] : string.Empty);
        var noteLang = notes.Select(nt => nt.Properties.GetString(KeyLanguage, partLang)).ToArray();

        // 渲染模式（仅 Retake 用会话级，RedrawPitch 由 piece 级标记驱动）
        bool isRetake = false;
        if (!mRenderModeConsumed && mRenderMode == RenderMode.Retake)
        {
            isRetake = true;
            mRenderModeConsumed = true;
            mRenderMode = RenderMode.Normal;
        }
        bool isRedrawPitch = piece.RedrawPitchRequested;
        piece.RedrawPitchRequested = false;

        bool pieceHasNoCache = piece.CachedPhones == null;
        bool needFullPredict = isRetake || !mHasValidCache || pieceHasNoCache;
        bool needPitchPredict = needFullPredict || isRedrawPitch;

        // —— 模型优先级：dsdur/dspitch 提级模型 ——
        var durPred = models.GetPredictor("dsdur");
        var pitchPred = models.GetPredictor("dspitch");
        var varPred = models.GetPredictor("dsvariance");

        // —— Phonemizer（needFullPredict 时才重新运行，否则复用缓存）——
        List<PhonemeSpan> phones;
        if (needFullPredict)
        {
            phones = durPred != null
                ? DiffSingerPhonemizer.Phonemize(durPred, notes, noteLang, speaker, hop, sr)
                : FallbackPhonemes(models, notes, noteLang);
            piece.CachedPhones = phones;
        }
        else
        {
            phones = piece.CachedPhones ?? new List<PhonemeSpan>();
        }
        if (phones.Count == 0)
            return null;
        progress?.Report(0.2);
        if (cancellation.IsCancellationRequested)
            return null;

        // —— 帧布局 ——
        var phoneDurSec = phones.Select(p => Math.Max(0, p.EndTime - p.StartTime)).ToArray();
        var durations = DiffSingerFrames.PaddedPhoneFrames(phoneDurSec, frameSec);
        int nTokens = durations.Length;
        int nFrames = durations.Sum();
        double renderStart = phones[0].StartTime - head * frameSec;

        // 逐帧时刻 + 说话人逐帧混合
        var frameTimes = new double[nFrames];
        for (int f = 0; f < nFrames; f++) frameTimes[f] = renderStart + (f + 0.5) * frameSec;
        var mixTracks = new List<(string Suffix, double[] Sampled)>();
        foreach (var (key, suffix) in SpeakerMixTracks(mConfig))
            if (snapshot.TryGetAutomation(key, out var mixAuto))
                mixTracks.Add((suffix, mixAuto.Evaluator.Evaluate(frameTimes)));
        var speakerMix = DiffSingerSpeakerMix.Create(Suffix(speaker), mixTracks, nFrames);

        // tokens/languages
        var tokens = new long[nTokens];
        var langs = new long[nTokens];
        tokens[0] = AcousticToken(models, "SP");
        tokens[nTokens - 1] = AcousticToken(models, "SP");
        for (int i = 0; i < phones.Count; i++)
        {
            tokens[i + 1] = AcousticToken(models, phones[i].Symbol);
            langs[i + 1] = models.TryGetLanguage(PhonemeLang(phones[i].Symbol), out var lid) ? lid : 0;
        }

        // 逐帧 note 音高回退
        var framePitch = new double[nFrames];
        int fi = 0;
        for (int seg = 0; seg < nTokens; seg++)
        {
            int ni = seg == 0 ? phones[0].NoteIndex
                : seg == nTokens - 1 ? phones[^1].NoteIndex
                : phones[seg - 1].NoteIndex;
            int pitch = notes[ni].Pitch;
            for (int k = 0; k < durations[seg]; k++) framePitch[fi++] = pitch;
        }

        // —— 自动音高预测（仅在 needPitchPredict 时跑；否则复用缓存）——
        float[]? predictedPitch;
        if (needPitchPredict)
        {
            predictedPitch = DiffSingerPitch.Predict(pitchPred, phones, notes, durations,
                renderStart, frameSec, speakerMix, mConfig, mSamplingSteps);
            piece.CachedPitchPrediction = predictedPitch;
        }
        else
        {
            predictedPitch = piece.CachedPitchPrediction;
        }
        progress?.Report(0.28);
        if (cancellation.IsCancellationRequested)
            return null;

        // 逐帧 f0(Hz) + 半音曲线
        var pitchCurve = snapshot.Pitch.Evaluator.Evaluate(frameTimes);
        var deviation = snapshot.PitchDeviation.Evaluator.Evaluate(frameTimes);
        var f0 = new float[nFrames];
        var semis = new float[nFrames];
        var pitchReadback = new List<Point>(nFrames);

        // —— 初始生成自动音高时排除前后 SP 音高数据（不画到用户界面）——
        //   判断哪些帧属于 head SP（第 0 个 token）和 tail SP（最后一个 token）
        int spHeadFrames = durations[0];    // head SP 的帧数
        int spTailFrames = durations[^1];   // tail SP 的帧数

        for (int f = 0; f < nFrames; f++)
        {
            double fallback;
            if (predictedPitch != null)
                fallback = f < predictedPitch.Length ? predictedPitch[f] : predictedPitch[^1];
            else
                fallback = framePitch[f];

            double semitone = (double.IsNaN(pitchCurve[f]) ? fallback : pitchCurve[f]) + deviation[f];
            semis[f] = (float)semitone;
            f0[f] = DiffSingerFrames.ToneToFreq(semitone);

            // 排除前后 SP 帧的音高回显（不在 pitchReadback 中添加）
            bool isHeadOrTailSp = f < spHeadFrames || f >= nFrames - spTailFrames;
            if (!isHeadOrTailSp)
            {
                pitchReadback.Add(new Point(frameTimes[f], semitone));
            }
        }
        progress?.Report(0.3);
        if (cancellation.IsCancellationRequested)
            return null;

        // —— variance 预测（needFullPredict 时重新预测；否则复用缓存）——
        VarianceCurves varCurves;
        if (needFullPredict)
        {
            varCurves = DiffSingerVariance.Predict(varPred, phones.Select(p => p.Symbol).ToList(),
                durations, semis, speakerMix, mConfig, mSamplingSteps);
            piece.CachedVarianceCurves = varCurves;
        }
        else
        {
            varCurves = piece.CachedVarianceCurves;
        }
        progress?.Report(0.45);
        if (cancellation.IsCancellationRequested)
            return null;

        // —— 声学输入 ——
        var ac = models.Acoustic;
        var inputs = new List<NamedOnnxValue>();
        void AddL(string name, long[] data, int[] dims)
        { if (ac.InputMetadata.ContainsKey(name)) inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, dims))); }
        void AddF(string name, float[] data, int[] dims)
        { if (ac.InputMetadata.ContainsKey(name)) inputs.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(data, dims))); }

        AddL("tokens", tokens, new[] { 1, nTokens });
        AddL("languages", langs, new[] { 1, nTokens });
        AddL("durations", durations.Select(x => (long)x).ToArray(), new[] { 1, nTokens });
        AddF("f0", f0, new[] { 1, nFrames });

        // —— variance：使用（缓存的）预测 + 用户 delta 合成喂声学 ——
        var varReadback = new Dictionary<string, IReadOnlyList<Point>>();
        foreach (var spec in Variances)
        {
            float[]? predicted = varCurves[spec.Key];
            double[]? user = snapshot.TryGetAutomation(spec.Key, out var auto)
                ? auto.Evaluator.Evaluate(frameTimes)
                : null;

            if (ac.InputMetadata.ContainsKey(spec.Key))
                AddF(spec.Key, CombineVariance(spec, predicted, user, nFrames), new[] { 1, nFrames });

            if (spec.Use(mConfig) && spec.Predict(mConfig) && predicted != null)
                varReadback[spec.Key] = BuildReadbackSegment(spec, predicted, frameTimes, nFrames);
        }

        // —— gender / velocity ——
        AddF("gender", BuildCurveInput(snapshot, KeyGender, GenderBaseline, GenderConvert(), frameTimes, nFrames), new[] { 1, nFrames });
        AddF("velocity", BuildCurveInput(snapshot, KeySpeed, SpeedBaseline, SpeedConvert, frameTimes, nFrames), new[] { 1, nFrames });

        if (ac.InputMetadata.ContainsKey("spk_embed"))
        {
            var spk = speakerMix.ToEmbedding(models.GetSpeakerEmbeddingBySuffix, hidden);
            inputs.Add(NamedOnnxValue.CreateFromTensor("spk_embed", new DenseTensor<float>(spk, new[] { 1, nFrames, hidden })));
        }
        if (mConfig.UseContinuousAcceleration)
        {
            if (ac.InputMetadata.ContainsKey("depth") && mConfig.UseVariableDepth)
                inputs.Add(NamedOnnxValue.CreateFromTensor("depth", new DenseTensor<float>(new[] { (float)models.MaxDepth }, new[] { 1 })));
            if (ac.InputMetadata.ContainsKey("steps"))
                inputs.Add(NamedOnnxValue.CreateFromTensor("steps", new DenseTensor<long>(new[] { (long)mSamplingSteps }, new[] { 1 })));
        }
        else if (ac.InputMetadata.ContainsKey("speedup"))
        {
            long speedup = Math.Max(1, 1000 / Math.Max(1, mSamplingSteps));
            while (1000 % speedup != 0 && speedup > 1) speedup--;
            inputs.Add(NamedOnnxValue.CreateFromTensor("speedup", new DenseTensor<long>(new[] { speedup }, new[] { 1 })));
        }

        // —— 声学模型：产 mel ——
        using var melOut = ac.Run(inputs);
        var melTensor = melOut.First(v => v.Name == "mel").AsTensor<float>();
        var melDims = melTensor.Dimensions.ToArray();
        var newMel = melTensor.ToArray();

        // —— 区段式 mel 拼贴：仅替换受影响的 time range，其余保持旧 mel ——
        float[] finalMel;
        if (piece.CachedMel != null && piece.CachedMelDims != null
            && melDims.SequenceEqual(piece.CachedMelDims) && piece.CachedMel.Length == newMel.Length)
        {
            finalMel = StitchMel(newMel, piece.CachedMel, frameTimes, nFrames, numMelBins, mAffectedStartTime, mAffectedEndTime);
        }
        else
        {
            finalMel = newMel;
        }

        // 更新 mel 缓存
        piece.CachedMel = finalMel;
        piece.CachedMelDims = melDims;

        progress?.Report(0.75);
        if (cancellation.IsCancellationRequested)
            return null;

        // —— 声码器（使用原始 mel 形状创建张量）——
        var voc = models.Vocoder;
        var melShape = new int[melDims.Length];
        Array.Copy(melDims, melShape, melDims.Length);
        var vInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("mel", new DenseTensor<float>(finalMel, melShape))
        };
        if (voc.InputMetadata.ContainsKey("f0"))
            vInputs.Add(NamedOnnxValue.CreateFromTensor("f0", new DenseTensor<float>(f0, new[] { 1, nFrames })));
        using var wavOut = voc.Run(vInputs);
        var audio = wavOut.First(v => v.Name == "waveform").AsTensor<float>().ToArray();
        progress?.Report(1.0);

        // —— 音素产物 ——
        var phonemes = phones.Select(p => new SynthesizedPhoneme
        {
            Symbol = p.Symbol,
            StartTime = p.StartTime,
            EndTime = p.EndTime,
            Note = origins[p.NoteIndex],
            StretchWeight = p.IsVowel ? 1 : 0,
        }).ToList();

        // 标记缓存有效（首次渲染成功后保持）
        mHasValidCache = true;

        return new RenderResultEx(audio, renderStart, sr, phonemes, pitchReadback, varReadback,
            melDims, finalMel);
    }

    // —— 区段式 mel 拼贴 ——
    // 将新 mel 的「受影响的 time range」替换到旧 mel 中，边界做 3 帧交叉过渡。
    // 旧 mel 为 null 或区间无效时直接返回新 mel。
    static float[] StitchMel(float[] newMel, float[]? oldMel, double[] frameTimes,
        int nFrames, int numMelBins, double affectedStart, double affectedEnd)
    {
        if (oldMel == null || oldMel.Length != newMel.Length)
            return newMel;
        if (double.IsNaN(affectedStart) || double.IsNaN(affectedEnd))
            return newMel;
        if (affectedEnd <= frameTimes[0] || affectedStart >= frameTimes[^1])
            return newMel; // affected 区间完全在渲染范围外

        const int fadeFrames = 3;
        int totalFrames = nFrames;

        // 找 affected 区间对应的帧范围（前后各扩展 fadeFrames 帧）
        int startFrame = totalFrames - 1;
        int endFrame = 0;
        for (int f = 0; f < totalFrames; f++)
        {
            if (frameTimes[f] >= affectedStart && frameTimes[f] <= affectedEnd)
            {
                if (f < startFrame) startFrame = f;
                if (f > endFrame) endFrame = f;
            }
        }
        startFrame = Math.Max(0, startFrame - fadeFrames);
        endFrame = Math.Min(totalFrames - 1, endFrame + fadeFrames);

        if (startFrame >= endFrame)
            return newMel;

        var result = new float[newMel.Length];
        Array.Copy(oldMel, result, newMel.Length);

        // 区间内直接替换为新 mel
        for (int f = startFrame + fadeFrames; f <= endFrame - fadeFrames; f++)
            for (int b = 0; b < numMelBins; b++)
                result[f * numMelBins + b] = newMel[f * numMelBins + b];

        // 前过渡区（fadeFrames 帧线性渐入）
        for (int i = 0; i < fadeFrames && startFrame + i < totalFrames; i++)
        {
            float t = (float)(i + 1) / (fadeFrames + 1);
            int f = startFrame + i;
            for (int b = 0; b < numMelBins; b++)
            {
                int idx = f * numMelBins + b;
                result[idx] = oldMel[idx] * (1 - t) + newMel[idx] * t;
            }
        }

        // 后过渡区（fadeFrames 帧线性渐出）
        for (int i = 0; i < fadeFrames && endFrame - i >= 0; i++)
        {
            float t = (float)(i + 1) / (fadeFrames + 1);
            int f = endFrame - i;
            for (int b = 0; b < numMelBins; b++)
            {
                int idx = f * numMelBins + b;
                result[idx] = oldMel[idx] * (1 - t) + newMel[idx] * t;
            }
        }

        return result;
    }

    // —— Point 列表拼贴 ——
    // 将受 affected 区间内的 Point 替换为新列表中的值，其余保持旧列表不变。
    // 旧列表为 null 或帧结构改变（点数不同）时直接返回新列表。
    static IReadOnlyList<Point> StitchPoints(IReadOnlyList<Point> newPoints, IReadOnlyList<Point>? oldPoints,
        double affectedStart, double affectedEnd)
    {
        if (oldPoints == null || oldPoints.Count == 0)
            return newPoints;
        if (double.IsNaN(affectedStart) || double.IsNaN(affectedEnd))
            return newPoints;
        if (newPoints.Count == 0)
            return oldPoints;
        if (newPoints.Count != oldPoints.Count)
            return newPoints;
        if (Math.Abs(newPoints[0].X - oldPoints[0].X) > 0.001)
            return newPoints;

        int oldStart = 0, oldEnd = oldPoints.Count;
        while (oldStart < oldPoints.Count && oldPoints[oldStart].X < affectedStart) oldStart++;
        while (oldEnd > 0 && oldPoints[oldEnd - 1].X > affectedEnd) oldEnd--;

        if (oldStart >= oldEnd)
            return newPoints;

        int newStart = 0, newEnd = newPoints.Count;
        while (newStart < newPoints.Count && newPoints[newStart].X < affectedStart) newStart++;
        while (newEnd > 0 && newPoints[newEnd - 1].X > affectedEnd) newEnd--;

        if (newStart >= newEnd)
            return oldPoints;

        // 拼接并在边界做 3 点线性过渡，避免 pitch 断层
        const int fadeCount = 3;
        var result = new List<Point>(oldStart + (newEnd - newStart) + (oldPoints.Count - oldEnd));

        // 旧区间（前段）
        for (int i = 0; i < oldStart; i++) result.Add(oldPoints[i]);

        // 前过渡：fadeCount 个点的旧→新渐变
        for (int i = 0; i < fadeCount && newStart + i < newEnd; i++)
        {
            float t = (float)(i + 1) / (fadeCount + 1);
            double x = newPoints[newStart + i].X;
            double y = oldPoints[oldStart + i].Y * (1 - t) + newPoints[newStart + i].Y * t;
            result.Add(new Point(x, y));
        }

        // 中间段：全量新值（仅当区间足够长时）
        for (int i = newStart + fadeCount; i < newEnd - fadeCount; i++)
            result.Add(newPoints[i]);

        // 后过渡：fadeCount 个点的新→旧渐变（按时间递增顺序）
        int backStart = Math.Max(newStart, newEnd - fadeCount);
        for (int i = backStart; i < newEnd; i++)
        {
            float t = (float)(newEnd - i) / (fadeCount + 1);
            int oi = oldEnd - (newEnd - i);
            if (oi < 0 || oi >= oldPoints.Count) continue;
            double x = newPoints[i].X;
            double y = oldPoints[oi].Y * (1 - t) + newPoints[i].Y * t;
            result.Add(new Point(x, y));
        }

        // 旧区间（后段）
        for (int i = oldEnd; i < oldPoints.Count; i++) result.Add(oldPoints[i]);

        return result;
    }

    // 无 dur 预测器兜底：每 note 一元音、占满 note 时长（无对齐/无 head/tail 之外的处理）。
    static List<PhonemeSpan> FallbackPhonemes(VoiceModels models, IReadOnlyList<SynthesisNoteSnapshot> notes, string[] noteLang)
    {
        var result = new List<PhonemeSpan>(notes.Count);
        for (int i = 0; i < notes.Count; i++)
        {
            string sym = PickVowelSymbol(models, noteLang[i]);
            result.Add(new PhonemeSpan(sym, notes[i].StartTime, notes[i].EndTime, i, true));
        }
        return result;
    }

    static long AcousticToken(VoiceModels models, string symbol)
        => models.TryGetPhoneme(symbol, out var id) ? id : 0;

    static string PhonemeLang(string phoneme)
    {
        int slash = phoneme.IndexOf('/');
        return slash > 0 ? phoneme[..slash] : string.Empty;
    }

    // 预测 x 与用户值 y（UI 单位，NaN 自由区代入中性）按 OpenUtau delta 函数合成，clamp 到声学值域。
    //   预测缺失（null，即 !Predict 而声学仍需该输入）→ 以 0 为基线降级，仅叠加用户 delta。
    static float[] CombineVariance(VarianceSpec spec, float[]? predicted, double[]? user, int n)
    {
        var result = new float[n];
        for (int f = 0; f < n; f++)
        {
            float x = predicted == null ? 0f : (f < predicted.Length ? predicted[f] : predicted[^1]);
            double y = user != null && !double.IsNaN(user[f]) ? user[f] : spec.Neutral;
            result[f] = (float)Math.Clamp(spec.Delta(x, (float)y), spec.AcousticMin, spec.AcousticMax);
        }
        return result;
    }

    // 纯用户曲线 → 帧级声学输入：按帧求值用户轨（无轨 / NaN 自由区 → 中性），逐帧 convert。
    //   不 clamp（OpenUtau 亦不 clamp，连续轨的 UI 量程已界定取值范围）。
    static float[] BuildCurveInput(SynthesisSnapshot snapshot, string key, double neutral,
        Func<double, double> convert, double[] frameTimes, int n)
    {
        double[]? user = snapshot.TryGetAutomation(key, out var auto)
            ? auto.Evaluator.Evaluate(frameTimes)
            : null;
        var result = new float[n];
        for (int f = 0; f < n; f++)
        {
            double y = user != null && !double.IsNaN(user[f]) ? user[f] : neutral;
            result[f] = (float)convert(y);
        }
        return result;
    }

    // GENC convert（OpenUtau DiffSingerRenderer）：正 = formant 下移；缩放由声库增广范围 KeyShift*（=range）定。
    //   range 某端为 0 ⇒ 该方向 scale=0（不移位）。闭包按当前声库现算（每会话固定）。
    Func<double, double> GenderConvert()
    {
        double posScale = mConfig.KeyShiftMax == 0 ? 0 : 12 / mConfig.KeyShiftMax / 100;
        double negScale = mConfig.KeyShiftMin == 0 ? 0 : -12 / mConfig.KeyShiftMin / 100;
        return x => x < 0 ? -x * posScale : -x * negScale;
    }

    // VELC convert（OpenUtau DiffSingerRenderer）：对数标度，100 = 原速，每 +100 速度 ×2。
    static double SpeedConvert(double x) => Math.Pow(2, (x - 100) / 100);

    // 回显段：纯预测值（不含用户编辑），clamp 到声学值域，逐帧 (全局秒, 值)。
    static List<Point> BuildReadbackSegment(VarianceSpec spec, float[] predicted, double[] frameTimes, int n)
    {
        var points = new List<Point>(n);
        for (int f = 0; f < n; f++)
        {
            float x = f < predicted.Length ? predicted[f] : predicted[^1];
            points.Add(new Point(frameTimes[f], Math.Clamp(x, spec.AcousticMin, spec.AcousticMax)));
        }
        return points;
    }

    // G2P 查无时的单音素兜底：优先该语言的 /a，回退裸 a，再回退 SP（静音）。
    static string PickVowelSymbol(VoiceModels models, string lang)
    {
        string keyed = string.IsNullOrEmpty(lang) ? "a" : $"{lang}/a";
        if (models.TryGetPhoneme(keyed, out _)) return keyed;
        if (models.TryGetPhoneme("a", out _)) return "a";
        return "SP";
    }

    // —— 产物（数据线程发布、可跨线程读）——
    public IReadOnlyList<IReadOnlyList<Point>> SynthesizedPitch
        => mPieces.Where(p => p.PitchReadback.Count > 0).Select(p => p.PitchReadback).ToList();

    // 回显产物（数据线程发布、可跨线程读）：按声明的回显轨 key 聚合各 piece 的纯预测段（每 piece 一段、段间断开）。
    public IReadOnlyMap<string, SynthesizedParameter> SynthesizedParameters
    {
        get
        {
            var map = new Map<string, SynthesizedParameter>();
            foreach (var kvp in mReadbackConfigs)
            {
                var segments = new List<IReadOnlyList<Point>>();
                foreach (var piece in mPieces)
                    if (piece.VarianceReadback.TryGetValue(kvp.Key, out var segment) && segment.Count > 0)
                        segments.Add(segment);
                if (segments.Count > 0)
                    map.Add(kvp.Key, new SynthesizedParameter { Segments = segments });
            }
            return map;
        }
    }

    public IReadOnlyList<SynthesizedPhoneme> Phonemes
        => mPieces.SelectMany(p => p.Phonemes).ToList();

    public IReadOnlyList<SynthesisStatusSegment> GetStatus()
    {
        var result = new List<SynthesisStatusSegment>(mPieces.Count);
        foreach (var piece in mPieces)
        {
            var status = piece.Failed ? SynthesisSegmentStatus.Failed
                : piece.Synthesizing ? SynthesisSegmentStatus.Synthesizing
                : piece.Dirty || piece.Segment == null ? SynthesisSegmentStatus.Pending
                : SynthesisSegmentStatus.Synthesized;
            result.Add(new SynthesisStatusSegment
            {
                StartTime = piece.StartTime,
                EndTime = piece.EndTime,
                Status = status,
                Message = piece.Failed ? piece.Error : piece.Synthesizing ? L.Tr("Synthesizing") : null,
                Progress = piece.Synthesizing ? piece.Progress : 0,
            });
        }
        return result;
    }

    public event Action? StatusChanged;

    public void Dispose()
    {
        mNotesSubscription.Dispose();
        mContext.Notes.ItemAdded -= OnNotesStructureChanged;
        mContext.Notes.ItemRemoved -= OnNotesStructureChanged;
        mContext.PartProperties.Modified -= MarkAllDirtyAndResegment;
        mContext.Pitch.RangeModified -= OnRangeModified;
        mContext.PitchDeviation.RangeModified -= OnRangeModified;
        foreach (var automation in mSubscribedAutomations)
            automation.RangeModified -= OnRangeModified;
        mContext.Committed -= OnCommitted;
        foreach (var piece in mPieces)
            piece.Segment?.Dispose();
        mPieces.Clear();
        // 模型会话归引擎级缓存所有、跨会话共享，不在此释放（引擎 Destroy 统一释放）。
    }

    // —— 分块（数据线程；按 note 间隙分块，note 集等价的块保留缓存与状态）——见 §5.9 重叠陷阱 ——
    void Resegment()
    {
        mNeedResegment = false;

        var groups = new List<List<ILiveNote>>();
        List<ILiveNote>? current = null;
        double groupMaxEnd = 0;
        foreach (var note in mContext.Notes)
        {
            if (current == null || note.StartTime.Value > groupMaxEnd)
            {
                current = new List<ILiveNote>();
                groups.Add(current);
                groupMaxEnd = note.EndTime.Value;
            }
            else
            {
                groupMaxEnd = Math.Max(groupMaxEnd, note.EndTime.Value);
            }
            current.Add(note);
        }

        var newPieces = new List<Piece>(groups.Count);
        foreach (var groupNotes in groups)
        {
            double pieceEnd = groupNotes.Max(n => n.EndTime.Value);
            var existing = mPieces.FirstOrDefault(p => p.Notes.SequenceEqual(groupNotes));
            if (existing != null)
            {
                mPieces.Remove(existing);
                existing.StartTime = groupNotes[0].StartTime.Value;
                existing.EndTime = pieceEnd;
                newPieces.Add(existing);
            }
            else
            {
                newPieces.Add(new Piece
                {
                    Notes = groupNotes,
                    StartTime = groupNotes[0].StartTime.Value,
                    EndTime = pieceEnd,
                    Dirty = true,
                });
            }
        }

        foreach (var piece in mPieces)
            piece.Segment?.Dispose();
        mPieces.Clear();
        mPieces.AddRange(newPieces);
        StatusChanged?.Invoke();
    }

    sealed record NoteHandlers(Action OnDur, Action OnPitch, Action OnLyric, Action OnProps);

    void SubscribeNote(ILiveNote note)
    {
        Action onDur = () =>
        {
            SetAffectedRange(note.StartTime.Value, note.EndTime.Value);
            MarkPieceDirty(note, clearPhones: true, clearPitch: false, clearVariance: true);
            mNeedResegment = true;
        };
        Action onPitch = () =>
        {
            SetAffectedRange(note.StartTime.Value, note.EndTime.Value);
            MarkPieceDirty(note, clearPhones: false, clearPitch: false, clearVariance: false);
        };
        Action onLyric = () =>
        {
            SetAffectedRange(note.StartTime.Value, note.EndTime.Value);
            MarkPieceDirty(note, clearPhones: true, clearPitch: true, clearVariance: true);
            mNeedResegment = true;
        };
        Action onProps = () =>
        {
            SetAffectedRange(note.StartTime.Value, note.EndTime.Value);
            MarkPieceDirty(note, clearPhones: true, clearPitch: false, clearVariance: true);
            mNeedResegment = true;
        };

        note.StartTime.Modified += onDur;
        note.EndTime.Modified += onDur;
        note.Phonemes.Modified += onDur;
        note.Pitch.Modified += onPitch;
        note.Lyric.Modified += onLyric;
        note.Properties.Modified += onProps;

        mNoteHandlers[note] = new NoteHandlers(onDur, onPitch, onLyric, onProps);
    }

    void MarkPieceDirty(ILiveNote note, bool clearPhones, bool clearPitch, bool clearVariance)
    {
        foreach (var piece in mPieces)
        {
            if (!piece.Notes.Contains(note)) continue;
            piece.Dirty = true; piece.Failed = false;
            if (clearPhones) piece.CachedPhones = null;
            if (clearPitch) piece.CachedPitchPrediction = null;
            if (clearVariance) piece.CachedVarianceCurves = default;
            return;
        }
    }

    void SetAffectedRange(double start, double end)
    {
        double pad = 0.1;
        mAffectedStartTime = start - pad;
        mAffectedEndTime = end + pad;
    }

    void UnsubscribeNote(ILiveNote note)
    {
        if (mNoteHandlers.Remove(note, out var h))
        {
            note.StartTime.Modified -= h.OnDur;
            note.EndTime.Modified -= h.OnDur;
            note.Phonemes.Modified -= h.OnDur;
            note.Pitch.Modified -= h.OnPitch;
            note.Lyric.Modified -= h.OnLyric;
            note.Properties.Modified -= h.OnProps;
        }
    }

    void OnNotesStructureChanged(ILiveNote note) { mNeedResegment = true; }

    void MarkAllDirtyAndResegment()
    {
        foreach (var piece in mPieces) { piece.Dirty = true; piece.Failed = false; }
        mNeedResegment = true;
    }

    void OnCommitted()
    {
        if (mNeedResegment)
            Resegment();
    }

    void OnRangeModified(double startTime, double endTime)
    {
        // 记录受影响的 time range（自动化曲线修改）
        SetAffectedRange(startTime, endTime);

        foreach (var piece in mPieces)
        {
            if (piece.EndTime < startTime || piece.StartTime > endTime)
                continue;
            piece.Dirty = true;
            piece.Failed = false;
        }
        StatusChanged?.Invoke();
    }

    sealed class Piece
    {
        public required IReadOnlyList<ILiveNote> Notes;
        public double StartTime;
        public double EndTime;
        public bool Dirty;
        public bool Failed;
        public bool Synthesizing;
        public string? Error;
        public double Progress;
        public IAudioSegment? Segment;
        public IReadOnlyList<SynthesizedPhoneme> Phonemes = [];
        public IReadOnlyList<Point> PitchReadback = [];
        public IReadOnlyDictionary<string, IReadOnlyList<Point>> VarianceReadback = new Dictionary<string, IReadOnlyList<Point>>();
        // 回显曲线缓存（区间拼贴用）
        public IReadOnlyList<Point>? CachedPitchReadback;
        public IReadOnlyDictionary<string, IReadOnlyList<Point>> CachedVarianceReadback = new Dictionary<string, IReadOnlyList<Point>>();

        // 缓存：note 代数（判断缓存有效性）
        // 模型预测缓存（增量渲染时复用）
        public float[]? CachedPitchPrediction;
        public VarianceCurves CachedVarianceCurves;
        public List<PhonemeSpan>? CachedPhones;
        // mel 缓存（用于交叉过渡）
        public float[]? CachedMel;
        public int[]? CachedMelDims;
        public float[]? CachedAudio;
        // piece 级 RedrawPitch 请求标记
        public bool RedrawPitchRequested;
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 一条 part 的合成会话。本阶段实现「声明面」：四个声明方法是选中声库能力集（VoicebankConfig）的纯函数——
// 据 use_*_embed 暴露可编辑曲线、据 predict_* 暴露只读回显轨、据 speakers/languages 暴露 part/note 属性。
// 调度与 6 级合成管线、产物发布为后续阶段：GetNextSegment 暂报「无待合成」，故宿主不驱动 SynthesizeNext，
// 会话呈现属性面板与轨但不产音——诚实的中间态。
public sealed class DiffSingerSynthesisSession : ISynthesisSession
{
    // —— 暴露给用户的参数键（避开宿主保留名 Volume / VibratoEnvelope）——
    const string KeyGender = "gender";
    const string KeySpeed = "speed";
    const string KeySpeaker = "speaker";
    const string KeyLanguage = "language";

    // Gender(GENC) / Speed(VELC) 连续轨：忠实采 OpenUtau 原生 UI 量程（非半音/倍率），convert 据此逐字移植。
    //   · GENC ∈ [-100,100]，基线 0 = 不移位，正 = formant 下移；增广范围 KeyShift* 仅入 convert 缩放，不做轨边界。
    //     与 energy/breath/tension 轨的 [-100,100] 一致。
    //   · VELC ∈ [0,200]，基线 100 = 原速，对数（每 +100 速度 ×2）；纯帧级声学条件，不改 durations / 帧数。
    const double GenderBaseline = 0, GenderMin = -100, GenderMax = 100;
    const double SpeedBaseline = 100, SpeedMin = 0, SpeedMax = 200;

    // 四个 variance 量的声明规格 + 合成语义（单一真相源，忠实移植 OpenUtau）：
    //   · 可编辑轨 = 增量 delta（OpenUtau UI 单位 [EditMin,EditMax]）；连续轨、基线=Neutral（处处有值）。
    //     Use(config) ⇒ 暴露此轨。日后若加「绝对覆盖式实参」，在 Render 的合成处另开分支即可，量程表不动。
    //   · 合成喂声学 = clamp(Delta(预测 x, 用户 y), AcousticMin, AcousticMax)；Neutral 代入 Delta 恒得纯预测。
    //   · 回显轨 = 纯预测值（同声学单位 [AcousticMin,AcousticMax]）；Use && Predict ⇒ 附此只读轨。
    // 注：voicing 的中性值为 100（OpenUtau 语义：默认 100、量程 [0,100]、只降不升），故其 NaN 自由区对应 100=纯预测。
    readonly record struct VarianceSpec(
        string Key, string Display, string Color,
        Func<VoicebankConfig, bool> Use, Func<VoicebankConfig, bool> Predict,
        double EditMin, double EditMax, double Neutral,
        double AcousticMin, double AcousticMax,
        Func<float, float, float> Delta);

    static readonly VarianceSpec[] Variances =
    {
        new("energy",      "Energy",      "#E573A5", c => c.UseEnergyEmbed,      c => c.PredictEnergy,      -100, 100,   0, -96, 0, (x, y) => x + y * 12 / 100),
        new("breathiness", "Breathiness", "#73E5C2", c => c.UseBreathinessEmbed, c => c.PredictBreathiness, -100, 100,   0, -96, 0, (x, y) => x + y * 12 / 100),
        new("voicing",     "Voicing",     "#C2E573", c => c.UseVoicingEmbed,     c => c.PredictVoicing,        0, 100, 100, -96, 0, (x, y) => x + (y - 100) * 12 / 100),
        new("tension",     "Tension",     "#A573E5", c => c.UseTensionEmbed,     c => c.PredictTension,     -100, 100,   0, -10, 10, (x, y) => x + y / 20),
    };

    readonly VoicebankConfig mConfig;
    readonly ISynthesisContext mContext;
    readonly string mVoiceId;
    readonly DiffSingerModelCache mModelCache;
    readonly int mSamplingSteps;

    // 轨集合与 part 面板只依赖声库能力集（每会话固定），构造期算一次即可；note 面板默认值随 part 当前值变，逐次构建。
    readonly OrderedMap<string, AutomationConfig> mAutomationConfigs = new();
    readonly OrderedMap<string, AutomationConfig> mReadbackConfigs = new();
    readonly ObjectConfig mPartConfig;

    // —— 调度状态（数据线程；按 note 间隙分块，账本式托管失效与产物）——
    readonly IDisposable mNotesSubscription;
    readonly List<ILiveAutomation> mSubscribedAutomations = new();   // 已订阅 RangeModified 的可编辑轨（Dispose 退订）
    readonly Dictionary<ILiveNote, Action> mNoteHandlers = new();
    readonly List<Piece> mPieces = new();
    bool mNeedResegment;
    bool mAutomationsWired;   // 可编辑轨 RangeModified 是否已惰性订阅（见 EnsureAutomationSubscriptions）

    public DiffSingerSynthesisSession(VoicebankConfig config, ISynthesisContext context,
        string voiceId, DiffSingerModelCache modelCache, int samplingSteps)
    {
        mConfig = config;
        mContext = context;
        mVoiceId = voiceId;
        mModelCache = modelCache;
        mSamplingSteps = samplingSteps;

        // 可编辑曲线：variance 量（连续 delta，基线=中性值=纯预测）+ Gender/Speed（连续，基线为中性值）。
        //   必须连续（非分段）：宿主仅把连续轨接进合成（存 mAutomations、快照可读、RangeModified 触发失效）；
        //   非音高分段轨在宿主里读不到也不失效（仅 Pitch 被特判）。中性基线上画偏移正是 delta 的天然形态。
        foreach (var v in Variances)
            if (v.Use(config))
                mAutomationConfigs.Add(v.Key, Continuous(v.Display, v.Color, v.Neutral, v.EditMin, v.EditMax));

        if (config.UseKeyShiftEmbed)
            mAutomationConfigs.Add(KeyGender, Continuous("Gender", "#E5A573", GenderBaseline, GenderMin, GenderMax));
        if (config.UseSpeedEmbed)
            mAutomationConfigs.Add(KeySpeed, Continuous("Speed", "#73B5E5", SpeedBaseline, SpeedMin, SpeedMax));

        // 只读回显轨：仅当声学接受该量为输入且方差器能产基线时——显示方差器纯预测（内容感知基线，真实声学单位）。
        foreach (var v in Variances)
            if (v.Use(config) && v.Predict(config))
                mReadbackConfigs.Add(v.Key, Piecewise(v.Display, v.Color, v.AcousticMin, v.AcousticMax));

        mPartConfig = BuildPartConfig();

        // 变更接线（handler 只做廉价标脏；重活延迟到 Committed 重分块）——见 §5.9。
        mNotesSubscription = NotifiableExtensions.WhenAny(context.Notes, SubscribeNote, UnsubscribeNote);
        context.Notes.ItemAdded += OnNotesStructureChanged;
        context.Notes.ItemRemoved += OnNotesStructureChanged;
        context.PartProperties.Modified += MarkAllDirtyAndResegment;
        context.Pitch.RangeModified += OnRangeModified;
        context.PitchDeviation.RangeModified += OnRangeModified;
        // 可编辑轨（variance / gender / speed）的订阅推迟到首次调度：宿主在 session 构造之后才
        // RefreshDeclarations 填充 Voice.AutomationConfigs，构造期 TryGetAutomation 必失败（proxy 未建）。
        context.Committed += OnCommitted;
        mNeedResegment = true;
    }

    // 新建 note 的默认歌词：中性占位，待词典 G2P 阶段按声库词典择一有效词细化。
    public string DefaultLyric => "a";

    public IReadOnlyOrderedMap<string, AutomationConfig> GetAutomationConfigs(IPartPropertyContext context) => mAutomationConfigs;
    public IReadOnlyOrderedMap<string, AutomationConfig> GetSynthesizedParameterConfigs(IPartPropertyContext context) => mReadbackConfigs;
    public ObjectConfig GetPartPropertyConfig(IPartPropertyContext context) => mPartConfig;

    // note 级：多语言声库暴露 per-note 语言覆盖；默认值取 part 当前默认语言（依赖 part 值 ⇒ 逐次构建）。
    public ObjectConfig GetNotePropertyConfig(INotePropertyContext context)
    {
        var properties = new OrderedMap<string, IControllerConfig>();
        if (HasLanguageChoice)
        {
            var partDefault = context.PartProperties.GetString(KeyLanguage, mConfig.Languages[0]);
            properties.Add(KeyLanguage, LanguageCombo(partDefault));
        }
        return new ObjectConfig { Properties = properties };
    }

    // —— 调度：窗内第一个脏块的纯值边界（peek 廉价、确定性）——
    public SynthesisSegment? GetNextSegment(double startTime, double endTime)
    {
        EnsureAutomationSubscriptions();
        return FindNextDirtyPiece(startTime, endTime) is { } p ? new SynthesisSegment(p.StartTime, p.EndTime) : null;
    }

    // 惰性订阅可编辑轨的区间编辑（否则画了曲线不重渲染）：宿主转发 RangeModified 需 live proxy，
    // 而 proxy 由 TryGetAutomation 惰性创建、其又依赖 Voice.AutomationConfigs 已填充（构造后才发生）。
    // 故首次调度（已过 RefreshDeclarations）时订阅；订上即止（轨集合每会话固定）。数据线程调用。
    void EnsureAutomationSubscriptions()
    {
        if (mAutomationsWired)
            return;

        bool any = false;
        foreach (var key in mAutomationConfigs.Keys)
            if (mContext.TryGetAutomation(key, out var automation))
            {
                automation.RangeModified += OnRangeModified;
                mSubscribedAutomations.Add(automation);
                any = true;
            }
        if (any || mAutomationConfigs.Count == 0)
            mAutomationsWired = true;
    }

    // peek 与 commit 共用同一查找（确定性 + 同调度 tick 无编辑 ⇒ commit 重算得到 peek 报出的同一块）。
    Piece? FindNextDirtyPiece(double startTime, double endTime)
    {
        if (mNeedResegment)
            Resegment();

        foreach (var piece in mPieces)
        {
            if (!piece.Dirty || piece.Failed || piece.Synthesizing)
                continue;
            if (piece.EndTime < startTime || piece.StartTime > endTime)
                continue;
            return piece;
        }
        return null;
    }

    public async Task SynthesizeNext(SynthesisSegment segment, CancellationToken cancellation = default)
    {
        if (FindNextDirtyPiece(segment.StartTime, segment.EndTime) is not { } piece)
            return;

        // 同步前缀（数据线程）：物化不可变快照（本块 note 全集 + 按 note 范围开窗）。
        var snapshot = mContext.GetSnapshot(piece.Notes, piece.Notes[0].StartTime.Value, piece.Notes.Max(n => n.EndTime.Value));
        piece.Dirty = false;
        piece.Synthesizing = true;
        piece.Progress = 0;
        StatusChanged?.Invoke();

        var report = new Progress<double>(p => { piece.Progress = p; StatusChanged?.Invoke(); });
        try
        {
            // offload：worker 只读冻结快照跑 ONNX（绝不碰活视图）；模型懒加载经引擎级缓存（首载触发原生加载）。
            var rendered = await Task.Run(() => Render(snapshot, piece.Notes, report, cancellation), CancellationToken.None);
            if (rendered != null && mPieces.Contains(piece))
            {
                int rate = rendered.SampleRate;
                piece.Segment?.Dispose();
                piece.Segment = mContext.CreateAudioSegment((long)(rendered.StartTime * rate), rendered.Audio.Length, rate);
                piece.Segment.Write(0, rendered.Audio);
                piece.Segment.Commit();
                piece.Phonemes = rendered.Phonemes;
                piece.PitchReadback = rendered.PitchReadback;
                piece.VarianceReadback = rendered.VarianceReadback;
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

    // 推理链（worker，只读冻结快照）：忠实移植 OpenUtau phonemizer + renderer（见记忆 openutau-is-authority）。
    //   phonemizer(dsdur) → 音素时间线；renderer 加 head/tail SP padding、tokens[SP..SP]、durations[8..8]、
    //   f0(Hz over totalFrames)、variance 预测+用户 delta 合成喂声学（纯预测产回显轨）、spk by frame、depth/steps。
    //   gender/velocity 走用户曲线 + OpenUtau GENC/VELC convert；pitch 走用户曲线（stage 4 接预测器）。
    RenderResult? Render(SynthesisSnapshot snapshot, IReadOnlyList<ILiveNote> origins,
        IProgress<double>? progress, CancellationToken cancellation)
    {
        var notes = snapshot.Notes;
        if (notes.Count == 0 || cancellation.IsCancellationRequested)
            return null;

        var models = mModelCache.GetOrLoad(mVoiceId, mConfig);
        int hop = models.HopSize, sr = models.SampleRate, hidden = models.HiddenSize;
        double frameSec = (double)hop / sr;
        int head = DiffSingerFrames.HeadFrames;

        string partLang = snapshot.PartProperties.GetString(KeyLanguage, mConfig.Languages.Count > 0 ? mConfig.Languages[0] : string.Empty);
        string speaker = snapshot.PartProperties.GetString(KeySpeaker, mConfig.Speakers.Count > 0 ? mConfig.Speakers[0] : string.Empty);
        var noteLang = notes.Select(nt => nt.Properties.GetString(KeyLanguage, partLang)).ToArray();

        // —— Phonemizer：歌词 → 音素时间线（绝对秒、含前置辅音越界）——
        var durPred = models.GetPredictor("dsdur");
        var phones = durPred != null
            ? DiffSingerPhonemizer.Phonemize(durPred, notes, noteLang, speaker, hop, sr)
            : FallbackPhonemes(models, notes, noteLang);   // 无 dur 预测器：每 note 一元音兜底
        if (phones.Count == 0)
            return null;
        progress?.Report(0.2);
        if (cancellation.IsCancellationRequested)
            return null;

        // —— 帧布局：[head SP][...phones...][tail SP]，累积取整 → durations（len=phones+2）——
        var phoneDurSec = phones.Select(p => Math.Max(0, p.EndTime - p.StartTime)).ToArray();
        var durations = DiffSingerFrames.PaddedPhoneFrames(phoneDurSec, frameSec);
        int nTokens = durations.Length;          // phones + 2
        int nFrames = durations.Sum();
        double renderStart = phones[0].StartTime - head * frameSec;

        // tokens/languages：声学表，前后加 SP。
        var tokens = new long[nTokens];
        var langs = new long[nTokens];
        tokens[0] = AcousticToken(models, "SP");
        tokens[nTokens - 1] = AcousticToken(models, "SP");
        for (int i = 0; i < phones.Count; i++)
        {
            tokens[i + 1] = AcousticToken(models, phones[i].Symbol);
            langs[i + 1] = models.TryGetLanguage(PhonemeLang(phones[i].Symbol), out var lid) ? lid : 0;
        }

        // 逐帧 note 音高回退（head→首 note，phone i→其 note，tail→末 note）。
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

        // 逐帧 f0(Hz) + 半音曲线（variance 用）：帧中心采样双通道音高，NaN 自由区回退 note 音高。
        var frameTimes = new double[nFrames];
        for (int f = 0; f < nFrames; f++) frameTimes[f] = renderStart + (f + 0.5) * frameSec;
        var pitchCurve = snapshot.Pitch.Evaluator.Evaluate(frameTimes);
        var deviation = snapshot.PitchDeviation.Evaluator.Evaluate(frameTimes);
        var f0 = new float[nFrames];
        var semis = new float[nFrames];
        var pitchReadback = new List<Point>(nFrames);
        for (int f = 0; f < nFrames; f++)
        {
            double semitone = (double.IsNaN(pitchCurve[f]) ? framePitch[f] : pitchCurve[f]) + deviation[f];
            semis[f] = (float)semitone;
            f0[f] = DiffSingerFrames.ToneToFreq(semitone);
            pitchReadback.Add(new Point(frameTimes[f], semitone));
        }
        progress?.Report(0.3);
        if (cancellation.IsCancellationRequested)
            return null;

        // —— variance 预测（基线；下方与用户 delta 合成喂声学、纯预测产回显）——
        var varCurves = DiffSingerVariance.Predict(
            models.GetPredictor("dsvariance"), phones.Select(p => p.Symbol).ToList(),
            durations, semis, speaker, mConfig, mSamplingSteps);
        progress?.Report(0.45);
        if (cancellation.IsCancellationRequested)
            return null;

        // —— 声学输入（按 InputMetadata 条件构造）——
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

        // —— variance：预测 + 用户 delta 合成喂声学，同时产纯预测回显 ——
        //   用户曲线按帧求值（连续轨：未编辑处=中性基线 → Delta 恒得纯预测；编辑处 → 叠加），clamp 到声学值域。
        //   回显（Use && Predict）= 纯预测值，不含用户编辑。
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

        // —— gender / velocity：纯用户曲线（无方差器基线），按帧 convert 喂声学（忠实移植 OpenUtau GENC/VELC）——
        //   无轨 / NaN 自由区 → 中性 → convert 得中性 embed（gender 0、velocity 1）；OpenUtau 不 clamp（UI 量程已界定）。
        AddF("gender", BuildCurveInput(snapshot, KeyGender, GenderBaseline, GenderConvert(), frameTimes, nFrames), new[] { 1, nFrames });
        AddF("velocity", BuildCurveInput(snapshot, KeySpeed, SpeedBaseline, SpeedConvert, frameTimes, nFrames), new[] { 1, nFrames });

        if (ac.InputMetadata.ContainsKey("spk_embed"))
        {
            var emb = models.GetSpeakerEmbedding(speaker);
            var spk = new float[nFrames * hidden];
            for (int f = 0; f < nFrames; f++) Array.Copy(emb, 0, spk, f * hidden, hidden);
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

        using var melOut = ac.Run(inputs);
        var mel = melOut.First(v => v.Name == "mel").AsTensor<float>();
        progress?.Report(0.75);
        if (cancellation.IsCancellationRequested)
            return null;

        // —— 声码器：mel (+ f0) → 波形 ——
        var voc = models.Vocoder;
        var vInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("mel", mel) };
        if (voc.InputMetadata.ContainsKey("f0"))
            vInputs.Add(NamedOnnxValue.CreateFromTensor("f0", new DenseTensor<float>(f0, new[] { 1, nFrames })));
        using var wavOut = voc.Run(vInputs);
        var audio = wavOut.First(v => v.Name == "waveform").AsTensor<float>().ToArray();
        progress?.Report(1.0);

        // —— 音素产物（绝对秒、韵核吸收伸缩）——
        var phonemes = phones.Select(p => new SynthesizedPhoneme
        {
            Symbol = p.Symbol,
            StartTime = p.StartTime,
            EndTime = p.EndTime,
            Note = origins[p.NoteIndex],
            StretchWeight = p.IsVowel ? 1 : 0,
        }).ToList();

        return new RenderResult(audio, renderStart, sr, phonemes, pitchReadback, varReadback);
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

    void SubscribeNote(ILiveNote note)
    {
        void Handler()
        {
            foreach (var piece in mPieces)
                if (piece.Notes.Contains(note)) { piece.Dirty = true; piece.Failed = false; }
            mNeedResegment = true;
        }
        mNoteHandlers[note] = Handler;
        note.StartTime.Modified += Handler;
        note.EndTime.Modified += Handler;
        note.Pitch.Modified += Handler;
        note.Lyric.Modified += Handler;
        note.Phonemes.Modified += Handler;
        note.Properties.Modified += Handler;
    }

    void UnsubscribeNote(ILiveNote note)
    {
        if (!mNoteHandlers.Remove(note, out var handler))
            return;
        note.StartTime.Modified -= handler;
        note.EndTime.Modified -= handler;
        note.Pitch.Modified -= handler;
        note.Lyric.Modified -= handler;
        note.Phonemes.Modified -= handler;
        note.Properties.Modified -= handler;
    }

    void OnNotesStructureChanged(ILiveNote note) => mNeedResegment = true;

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
        foreach (var piece in mPieces)
        {
            if (piece.EndTime < startTime || piece.StartTime > endTime)
                continue;
            piece.Dirty = true;
            piece.Failed = false;
        }
        StatusChanged?.Invoke();
    }

    sealed record RenderResult(float[] Audio, double StartTime, int SampleRate,
        List<SynthesizedPhoneme> Phonemes, List<Point> PitchReadback,
        Dictionary<string, IReadOnlyList<Point>> VarianceReadback);

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
    }

    // —— 构建辅助 ——
    bool HasLanguageChoice => mConfig.UseLanguageId && mConfig.Languages.Count > 1;

    ObjectConfig BuildPartConfig()
    {
        var properties = new OrderedMap<string, IControllerConfig>();

        if (mConfig.Speakers.Count > 1)
            properties.Add(KeySpeaker, new ComboBoxConfig
            {
                DisplayText = L.Tr("Speaker"),
                Options = SpeakerOptions(mConfig.Speakers),
            });

        if (HasLanguageChoice)
            properties.Add(KeyLanguage, LanguageCombo(mConfig.Languages[0]));

        return new ObjectConfig { Properties = properties };
    }

    ComboBoxConfig LanguageCombo(string defaultValue) => new()
    {
        DisplayText = L.Tr("Language"),
        Options = ToOptions(mConfig.Languages),
        DefaultOption = PropertyValue.Create(defaultValue),
    };

    static List<ComboBoxOption> ToOptions(IReadOnlyList<string> values)
    {
        var options = new List<ComboBoxOption>(values.Count);
        foreach (var value in values)
            options.Add(value);   // 隐式转换：string → ComboBoxOption（值即显示文本）
        return options;
    }

    // 说话人：值保留 dsconfig 原始条目（下游据此选 .emb），显示去模型名前缀（"260509a.Miku" → "Miku"）。
    // 注：character.yaml 的 subbanks 带更友好的本地化名（如「01: 初音未来」），按 suffix 关联可作后续增强。
    static List<ComboBoxOption> SpeakerOptions(IReadOnlyList<string> speakers)
    {
        var options = new List<ComboBoxOption>(speakers.Count);
        foreach (var speaker in speakers)
        {
            int dot = speaker.LastIndexOf('.');
            var display = dot >= 0 && dot < speaker.Length - 1 ? speaker[(dot + 1)..] : speaker;
            options.Add(new ComboBoxOption(PropertyValue.Create(speaker), display));
        }
        return options;
    }

    static AutomationConfig Piecewise(string display, string color, double min, double max) => new()
    {
        DisplayText = L.Tr(display),
        DefaultValue = double.NaN,   // 分段：无基线、段间断开（NaN 自由区）
        MinValue = min,
        MaxValue = max,
        Color = color,
    };

    static AutomationConfig Continuous(string display, string color, double baseline, double min, double max) => new()
    {
        DisplayText = L.Tr(display),
        DefaultValue = baseline,
        MinValue = min,
        MaxValue = max,
        Color = color,
    };
}

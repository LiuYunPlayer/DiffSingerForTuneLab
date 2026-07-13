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

// 一条 part 的合成会话。本阶段实现「声明面」：四个声明方法是选中声库能力集（VoicebankConfig）的纯函数——
// 据 use_*_embed 暴露可编辑曲线、据 predict_* 暴露只读回显轨、据 speakers/languages 暴露 part/note 属性。
// 调度与 6 级合成管线、产物发布为后续阶段：GetNextSegment 暂报「无待合成」，故宿主不驱动 SynthesizeNext，
// 会话呈现属性面板与轨但不产音——诚实的中间态。
// 声明面（轨集合/属性面板）已上移到 DiffSingerVoiceEngine（经 DiffSingerDeclarations）；本会话仅承载运行时：
// 调度、6 级推理管线、产物发布。轨 key 与 variance/gender/speed 规格复用 DiffSingerDeclarations（using static 引入）。
public sealed class DiffSingerSynthesisSession : IVoiceSynthesisSession
{
    readonly IVoiceSynthesisContext mContext;
    readonly string mVoiceId;
    readonly Func<ResolveProps, PartContext?> mResolve;   // (model/version 选择) → 解析到具体物理包能力集；运行时支持换 model/version
    readonly VoicebankConfig? mConfig;   // 构造期解析包（驱动固定轨订阅/回显轨集合）；运行时 Render 按 part 属性另行解析
    readonly DiffSingerModelCache mModelCache;
    readonly int mSamplingSteps;
    readonly bool mTensorCache;        // 张量缓存总开关（引擎设置 tensor_cache）
    readonly int mCacheMaxSizeMb;      // 缓存体积上限（MB）；0 = 不限制（引擎设置 cache_max_size_mb）

    // 运行时复用的声明派生物（每会话固定，构造期据声库能力集算一次）：
    //   可编辑轨集合（构造期订阅其区间编辑）+ 回显轨集合（产物 SynthesizedParameters 按其 key 聚合）。
    readonly OrderedMap<PropertyKey, AutomationConfig> mReadbackConfigs;

    // —— 调度状态（数据线程；按 note 间隙分块，账本式托管失效与产物）——
    readonly IActionEvent<IVoiceSynthesisNote> mNoteFieldModified;   // 「任一 note 任一字段变」聚合事件（宿主 WhenAnyItem，成员增删自动接线/退订）
    // 已订阅 RangeModified 的全部自动化轨（key → automation）。固定轨(variance/gender/speed)、seed、混音轨**皆**随
    //   解析包(model/version)能力与 part 属性变化而增减，故统一幂等 re-sync——见 SyncAutomationSubscriptions。
    readonly Dictionary<string, ISynthesisAutomation> mSubscriptions = new();
    readonly List<Piece> mPieces = new();
    bool mNeedResegment;

    public DiffSingerSynthesisSession(string voiceId, IVoiceSynthesisContext context,
        Func<ResolveProps, PartContext?> resolve, DiffSingerModelCache modelCache,
        int samplingSteps, bool tensorCache, int cacheMaxSizeMb)
    {
        mVoiceId = voiceId;
        mContext = context;
        mResolve = resolve;
        mModelCache = modelCache;
        mSamplingSteps = samplingSteps;
        mTensorCache = tensorCache;
        mCacheMaxSizeMb = cacheMaxSizeMb;

        // 构造期解析一次（驱动固定轨订阅与回显轨集合）；运行时 Render 按 part 属性另行解析、支持 model/version 切换。
        var pc = resolve(PropsOf(context.PartProperties));
        mConfig = pc?.Config;
        mReadbackConfigs = pc != null ? BuildReadbackConfigs(pc.Config) : new OrderedMap<PropertyKey, AutomationConfig>();

        // 变更接线（handler 只做廉价标脏；重活延迟到 Committed 重分块）——见 §5.9。
        //   note 字段变更：宿主 WhenAnyItem 把「每个现有/未来成员的这几个属性事件」聚合成一个带成员标识的事件，
        //   成员增删时自动接线/退订（取代旧的手工 SubscribeNote/UnsubscribeNote 簿记）。
        mNoteFieldModified = context.Notes.WhenAnyItem(
            n => n.StartTime.Modified,
            n => n.EndTime.Modified,
            n => n.Pitch.Modified,
            n => n.Lyric.Modified,
            n => n.LeadingPhonemes.Modified,   // 钉死音素双列表 + 结合线偏移无合并信号，须同订三者（SDK 契约）
            n => n.BodyPhonemes.Modified,
            n => n.BodyOffset.Modified,
            n => n.Properties.Modified);
        mNoteFieldModified.Subscribe(OnNoteModified);
        context.Notes.MembershipModified.Subscribe(OnNotesStructureChanged);   // 成员增删聚合信号 → 重分块
        context.PartProperties.Modified.Subscribe(MarkAllDirtyAndResegment);
        context.Pitch.RangeModified.Subscribe(OnRangeModified);
        context.PitchDeviation.RangeModified.Subscribe(OnRangeModified);
        context.Committed.Subscribe(OnCommitted);

        // 区间编辑失效订阅：固定轨(variance/gender/speed，按能力 gating)、seed 轨(按 retake gating)、混音轨(按 part 属性)
        //   皆随解析包(model/version)与 part 属性增减，故统一幂等 re-sync——构造期订一次，part 属性变再同步。
        //   SDK 把声明上移到引擎后，宿主在「建会话之前」即 RefreshDeclarations 填好 context.Automations（见 MidiPart 时序）。
        SyncAutomationSubscriptions();

        mNeedResegment = true;
    }

    // 新建 note 的默认歌词：中性占位，待词典 G2P 阶段按声库词典择一有效词细化。
    public string DefaultLyric => "a";

    // 延音判定（SDK 唯一通道，宿主照单消费：显示布局/编辑手势同源读本值）。判定绑定本引擎合成行为——
    //   phonemize 的快照域自判（DiffSingerPhonemizer.ComputeContinuation）与本函数同语义，改一处必改两处。
    // 语义：歌词 "-"（trim 后）∧ 经不断裂相接链回溯到内容 note。**只看歌词与位置，不看钉死音素**——
    //   与 SDK 参考语义的刻意偏离：钉死 `-` 的现实来源只有陈旧残留（乘客无音素可显示、UI 钉不了；
    //   宿主改歌词也不清钉死），"la"(带钉死)→改词 "-" 时用户最新意图就是 melisma，钉死排除会让旧钉死
    //   压过 "-" 编辑；忽略语义下钉死随 "-" 休眠不销毁、歌词改回即复活（无损可逆）。判定为延续的 note
    //   宿主本就不读其音素数据，休眠钉死不可见不可闻，与宿主机制同向。
    //   · 相接 = 前有效末 ≥ 后起，严格比较无容差（note 边界同源 tick 换算，相接即精确相等）——
    //     与 Resegment 的严格间隙判据同构，故判定为延续的 note 恒与其链头同块，快照域自判等价。
    //   · "+"（拆音节推进）是发声 note、恒非延续；空/未知歌词发 SP、也是内容 note。
    public bool IsContinuation(IVoiceSynthesisNote note)
    {
        if (!IsContinuationMarker(note.Lyric.Value))
            return false;
        var cur = note;
        while (true)
        {
            var prev = cur.Last;
            if (prev == null)
                return false;                                      // 链跑出开头、无内容 note → 孤儿
            if (prev.EndTime.Value < cur.StartTime.Value)
                return false;                                      // 空隙断链（严格比较）→ 孤儿
            if (!IsContinuationMarker(prev.Lyric.Value))
                return true;                                       // 回溯到内容 note（非记号歌词）→ 生效
            cur = prev;
        }
    }

    // 延音记号（本引擎约定，编辑器 "-" 录入约定的对应判定）。快照域自判共用。
    internal static bool IsContinuationMarker(string? lyric) => lyric?.Trim() == "-";

    // —— 调度：窗内第一个脏块的纯值边界（peek 廉价、确定性）——
    public SynthesisRange? GetNextSegment(double startTime, double endTime)
        => FindNextDirtyPiece(startTime, endTime) is { } p ? new SynthesisRange(p.StartTime, p.EndTime) : null;

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

    public async Task SynthesizeNext(double startTime, double endTime, CancellationToken cancellation = default)
    {
        if (FindNextDirtyPiece(startTime, endTime) is not { } piece)
            return;

        // 同步前缀（数据线程）：物化不可变快照（本块 note 全集 + 按 note 范围开窗）。
        var snapshot = mContext.GetSnapshot(piece.Notes, piece.Notes[0].StartTime.Value, piece.Notes.Max(n => n.EndTime.Value));
        piece.Dirty = false;
        piece.Synthesizing = true;
        piece.Progress = 0;
        mStatusChanged.Invoke();

        var report = new Progress<double>(p => { piece.Progress = p; mStatusChanged.Invoke(); });
        try
        {
            // offload：worker 只读冻结快照跑 ONNX（绝不碰活视图）；模型懒加载经引擎级缓存（首载触发原生加载）。
            //   合成毕在 worker 线程顺手做一次缓存体积上限逐出（仅开缓存且设了上限时；off 数据线程、尽力而为）。
            var rendered = await Task.Run(() =>
            {
                var result = Render(snapshot, piece.Notes, report, cancellation);
                if (mTensorCache && mCacheMaxSizeMb > 0)
                    DiffSingerTensorCache.EnforceSizeLimit(mCacheMaxSizeMb);
                return result;
            }, CancellationToken.None);
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
            // 块完成（产物写入）/ 失败（从产物中排除）→ 音素 / 参数 / 音高产物变化，连同状态一并通知。
            NotifyProducts();
        }
    }

    // 推理链（worker，只读冻结快照）：忠实移植 OpenUtau phonemizer + renderer（见记忆 openutau-is-authority）。
    //   phonemizer(dsdur) → 音素时间线；renderer 加 head/tail SP padding、tokens[SP..SP]、durations[8..8]、
    //   f0(Hz over totalFrames)、variance 预测+用户 delta 合成喂声学（纯预测产回显轨）、spk by frame、depth/steps。
    //   gender/velocity 走用户曲线 + OpenUtau GENC/VELC convert；pitch 自由区走 dspitch 预测轮廓、已画处用户值覆盖。
    RenderResult? Render(VoiceSynthesisSnapshot snapshot, IReadOnlyList<IVoiceSynthesisNote> origins,
        IProgress<double>? progress, CancellationToken cancellation)
    {
        var notes = snapshot.Notes;
        if (notes.Count == 0 || cancellation.IsCancellationRequested)
            return null;

        // 运行时解析：按 part 属性（model/version）定到具体物理包，支持会话存续期内换 model/version。
        if (mResolve(PropsOf(snapshot.PartProperties)) is not { } pc)
            return null;
        var config = pc.Config;
        var resolved = pc.Resolved;

        var models = mModelCache.GetOrLoad(config);
        int hop = models.HopSize, sr = models.SampleRate, hidden = models.HiddenSize;
        double frameSec = (double)hop / sr;
        int head = DiffSingerFrames.HeadFrames;

        var speakerSet = SpeakerSet.Compute(resolved);
        string partLang = snapshot.PartProperties.GetString(KeyLanguage, DefaultLanguageId(config, resolved));
        // 合成用说话人（嵌入解析）= 当前 voice 在该包的 dsconfig 后缀（单说话人模型为空串、模型无 spk_embed 时不喂）。
        string speaker = !string.IsNullOrEmpty(resolved.VoiceSpeaker)
            ? resolved.VoiceSpeaker
            : (config.Speakers.Count > 0 ? config.Speakers[0] : string.Empty);
        var noteLang = notes.Select(nt => nt.Properties.GetString(KeyLanguage, partLang)).ToArray();

        // —— Phonemizer：歌词 → 音素时间线（绝对秒、含前置辅音越界）——
        var durPred = models.GetPredictor("dsdur");
        var phones = durPred != null
            ? DiffSingerPhonemizer.Phonemize(durPred, notes, noteLang, speaker, hop, sr, mTensorCache)
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

        // 逐帧时刻 + 说话人逐帧混合（acoustic/pitch/variance 三域共享；未启用任何混合时退化为默认 speaker 恒权重）。
        //   遍历全量 mix:<suffix> 候选，snapshot.Automations 只含已声明轨——即用户在 part 面板已 + 的 speaker
        //   （speaker_mix 容器已选键），未选的 speaker 此处自然跳过、不参与混合。
        var frameTimes = new double[nFrames];
        for (int f = 0; f < nFrames; f++) frameTimes[f] = renderStart + (f + 0.5) * frameSec;
        var mixTracks = new List<(string Suffix, double[] Sampled)>();
        foreach (var (key, suffix) in SpeakerMixTracks(speakerSet))
            if (snapshot.Automations.TryGetValue(key, out var mixAuto))
                mixTracks.Add((suffix, mixAuto.Evaluator.Evaluate(frameTimes)));
        var speakerMix = DiffSingerSpeakerMix.Create(Suffix(speaker), mixTracks, nFrames);

        // pitch / variance 的 seed 自动化轨 → 逐帧 seed：归一化 [0,1] 放大到 uint32（哈希白化，刻度不影响质量）。
        //   平线 = 全局 take；画区段 = 该区独立 take（时间维 × 值维）。无轨/未画/NaN → 0（= 保留 take-0）。
        //   只有精确 v=0 映成 seed 0；任何画上去的 v>0 都映成非零 → 触发该帧重摇（acoustic retake mask 命门）。
        uint[] SampleSeedCurve(string key)
        {
            var seeds = new uint[nFrames];
            if (snapshot.Automations.TryGetValue(key, out var seedAuto))
            {
                var v = seedAuto.Evaluator.Evaluate(frameTimes);
                for (int f = 0; f < nFrames; f++)
                    seeds[f] = double.IsNaN(v[f]) ? 0u : (uint)Math.Round(Math.Clamp(v[f], 0, SeedCurveMax) * uint.MaxValue);
            }
            return seeds;
        }
        var seedPitchCurve = SampleSeedCurve(KeySeedPitch);
        var seedVarianceCurve = SampleSeedCurve(KeySeedVariance);
        var seedAcousticCurve = SampleSeedCurve(KeySeedAcoustic);

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

        // —— P3 包络：音素混合 → tokens_b（次要音素，逐 token）+ blend（逐帧比例，来自 part 级包络曲线）——
        //   目标符号来自 per-phoneme 属性（经 PhonemeSpan 透传，已解析）；比例来自 KeyMixCurve 逐帧采样。
        //   单槽（S=1）：tokens_b=[1,nTokens]、blend=[1,nFrames]。无目标的 token 其帧 blend 置 0（该 token tokens_b=base，
        //   即便非 0 也无效，置 0 只为语义清晰）。声学未声明 tokens_b/blend（老库）时下方 HasInput gating 静默跳过。
        var tokensB = (long[])tokens.Clone();
        var tokenHasMix = new bool[nTokens];
        for (int i = 0; i < phones.Count; i++)
        {
            var mixSym = phones[i].MixSymbol;
            if (!string.IsNullOrEmpty(mixSym) && models.TryGetPhoneme(mixSym, out _))
            {
                tokensB[i + 1] = AcousticToken(models, mixSym);
                tokenHasMix[i + 1] = true;
            }
        }
        var mixRatioCurve = snapshot.Automations.TryGetValue(KeyMixCurve, out var mixRatioAuto)
            ? mixRatioAuto.Evaluator.Evaluate(frameTimes) : null;
        var blend = new float[nFrames];   // 全 0 = 不混合（等价原模型）
        if (mixRatioCurve != null && tokenHasMix.Any(b => b))
        {
            int mf = 0;
            for (int seg = 0; seg < nTokens; seg++)
                for (int k = 0; k < durations[seg]; k++, mf++)
                    if (tokenHasMix[seg])
                        blend[mf] = (float)Math.Clamp(double.IsNaN(mixRatioCurve[mf]) ? 0 : mixRatioCurve[mf], 0, 1);
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

        // —— dspitch 自然音高预测（纯从音符、retake 全 true、不吃用户音高）：替代自由区的矩形 note-step 兜底 ——
        //   用户已画处（Pitch 非 NaN）用户值覆盖；NaN 自由区用预测轮廓（无 dspitch ⇒ 仍用矩形 framePitch）；PITD/vibrato 叠加在上。
        var predictedPitch = DiffSingerPitch.Predict(
            models.GetPredictor("dspitch"), phones, notes, durations,
            renderStart, frameSec, speakerMix, config, mSamplingSteps, seedPitchCurve, mTensorCache);
        progress?.Report(0.28);
        if (cancellation.IsCancellationRequested)
            return null;

        // 逐帧 f0(Hz) + 半音曲线（variance 用）：帧中心采样双通道音高，NaN 自由区回退预测轮廓（无则 note 音高）。
        var pitchCurve = snapshot.Pitch.Evaluator.Evaluate(frameTimes);
        var deviation = snapshot.PitchDeviation.Evaluator.Evaluate(frameTimes);
        var f0 = new float[nFrames];
        var semis = new float[nFrames];
        var pitchReadback = new List<Point>(nFrames);
        for (int f = 0; f < nFrames; f++)
        {
            double fallback = predictedPitch != null
                ? (f < predictedPitch.Length ? predictedPitch[f] : predictedPitch[^1])
                : framePitch[f];
            double semitone = (double.IsNaN(pitchCurve[f]) ? fallback : pitchCurve[f]) + deviation[f];
            semis[f] = (float)semitone;
            f0[f] = DiffSingerFrames.ToneToFreq(semitone);
            pitchReadback.Add(new Point(frameTimes[f], semitone));
        }
        progress?.Report(0.3);
        if (cancellation.IsCancellationRequested)
            return null;

        // —— variance 预测（基线；下方与用户 delta 合成喂声学、纯预测产回显）——
        var varCurves = DiffSingerVariance.Predict(
            models.GetPredictor("dsvariance"), phones,
            durations, semis, speakerMix, config, mSamplingSteps, seedVarianceCurve, mTensorCache);
        progress?.Report(0.45);
        if (cancellation.IsCancellationRequested)
            return null;

        // —— 声学条件输入（按 InputMetadata 条件构造；不含 retake/gt_mel/noise，那三个在下方按重摇逻辑追加）——
        var ac = models.Acoustic;
        var cond = new List<NamedOnnxValue>();
        void AddL(string name, long[] data, int[] dims)
        { if (ac.HasInput(name)) cond.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, dims))); }
        void AddF(string name, float[] data, int[] dims)
        { if (ac.HasInput(name)) cond.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<float>(data, dims))); }

        AddL("tokens", tokens, new[] { 1, nTokens });
        AddL("languages", langs, new[] { 1, nTokens });
        AddL("durations", durations.Select(x => (long)x).ToArray(), new[] { 1, nTokens });
        AddL("tokens_b", tokensB, new[] { 1, nTokens });   // 音素混合目标 [S=1, nTokens]（模型无此输入则静默跳过）
        AddF("blend", blend, new[] { 1, nFrames });        // 逐帧比例包络 [S=1, nFrames]
        AddF("f0", f0, new[] { 1, nFrames });

        // —— variance：预测 + 用户 delta 合成喂声学，同时产纯预测回显 ——
        //   用户曲线按帧求值（连续轨：未编辑处=中性基线 → Delta 恒得纯预测；编辑处 → 叠加），clamp 到声学值域。
        //   回显（Use && Predict）= 纯预测值，不含用户编辑。
        var varReadback = new Dictionary<string, IReadOnlyList<Point>>();
        foreach (var spec in Variances)
        {
            float[]? predicted = varCurves[spec.Key];
            double[]? user = snapshot.Automations.TryGetValue(spec.Key, out var auto)
                ? auto.Evaluator.Evaluate(frameTimes)
                : null;

            if (ac.HasInput(spec.Key))
                AddF(spec.Key, CombineVariance(spec, predicted, user, nFrames), new[] { 1, nFrames });

            if (spec.Use(config) && spec.Predict(config) && predicted != null)
                varReadback[spec.Key] = BuildReadbackSegment(spec, predicted, frameTimes, nFrames);
        }

        // —— gender / velocity：纯用户曲线（无方差器基线），按帧 convert 喂声学（忠实移植 OpenUtau GENC/VELC）——
        //   无轨 / NaN 自由区 → 中性 → convert 得中性 embed（gender 0、velocity 1）；OpenUtau 不 clamp（UI 量程已界定）。
        AddF("gender", BuildCurveInput(snapshot, KeyGender, GenderBaseline, GenderConvert(config), frameTimes, nFrames), new[] { 1, nFrames });
        AddF("velocity", BuildCurveInput(snapshot, KeySpeed, SpeedBaseline, SpeedConvert, frameTimes, nFrames), new[] { 1, nFrames });

        if (ac.HasInput("spk_embed"))
        {
            var spk = speakerMix.ToEmbedding(models.GetSpeakerEmbeddingBySuffix, hidden);
            cond.Add(NamedOnnxValue.CreateFromTensor("spk_embed", new DenseTensor<float>(spk, new[] { 1, nFrames, hidden })));
        }
        if (config.UseContinuousAcceleration)
        {
            if (ac.HasInput("depth") && config.UseVariableDepth)
                cond.Add(NamedOnnxValue.CreateFromTensor("depth", new DenseTensor<float>(new[] { (float)models.MaxDepth }, new[] { 1 })));
            if (ac.HasInput("steps"))
                cond.Add(NamedOnnxValue.CreateFromTensor("steps", new DenseTensor<long>(new[] { (long)mSamplingSteps }, new[] { 1 })));
        }
        else if (ac.HasInput("speedup"))
        {
            long speedup = Math.Max(1, 1000 / Math.Max(1, mSamplingSteps));
            while (1000 % speedup != 0 && speedup > 1) speedup--;
            cond.Add(NamedOnnxValue.CreateFromTensor("speedup", new DenseTensor<long>(new[] { speedup }, new[] { 1 })));
        }

        // —— note 级 acoustic retake（软条件，canonical take-0 参照；无记忆、关开工程可复现）——
        //   seed 轨逐帧：seed≠0 的帧重摇(retake=true)、seed=0 的帧保留(复刻 take-0 参照)。
        //   · 非混合（全保留 or 全重摇）→ 单趟：retake 全 true、gt_mel 全 0、噪声按 seed 轨
        //     （retake 全 true 时 gt_mel 被模型乘 0 屏蔽，故结果只取决于 seed 轨噪声，与历史/参照无关 → 一致可复现）。
        //   · 混合 → 先算 take-0 参照（retake 全 true、gt_mel 全 0、seed=0 噪声；输入恒定 → 张量缓存只算一次），
        //     再正式趟（retake=逐帧、gt_mel=参照、噪声按 seed 轨）。参照由 seed=0 确定性重算，不记忆上次 mel。
        //   扩散噪声仅当模型有 noise 口时喂入（fork 外置噪声）；retake/gt_mel 仅当模型有这俩口（训练 use_acoustic_retake）。
        Tensor<float> RunAcoustic(List<NamedOnnxValue> ins)
            => DiffSingerTensorCache.Run(ac, models.AcousticHash, ins, mTensorCache)
                .First(v => v.Name == "mel").AsTensor<float>();

        Tensor<float> mel;
        if (!(ac.HasInput("retake") && ac.HasInput("gt_mel")))
        {
            // stock / 仅噪声模型：无 retake 口，按 seed 轨喂噪声直接单趟（无 noise 口则退回内部随机）。
            var ins = new List<NamedOnnxValue>(cond);
            DiffSingerNoise.AddNoise(ins, ac, seedAcousticCurve, DiffSingerNoise.StageAcoustic, nFrames);
            mel = RunAcoustic(ins);
        }
        else
        {
            int melBins = ac.InputShape("gt_mel")[2];
            var retakeMask = new bool[nFrames];
            bool anyKeep = false, anyReroll = false;
            for (int f = 0; f < nFrames; f++)
            {
                bool rr = seedAcousticCurve[f] != 0;   // seed≠0 ⇒ 重摇该帧
                retakeMask[f] = rr;
                anyReroll |= rr;
                anyKeep |= !rr;
            }
            var allTrue = new bool[nFrames];
            Array.Fill(allTrue, true);
            var zeroMel = new float[nFrames * melBins];

            void AddRetake(List<NamedOnnxValue> dst, bool[] mask, float[] gtMel)
            {
                dst.Add(NamedOnnxValue.CreateFromTensor("retake", new DenseTensor<bool>(mask, new[] { 1, nFrames })));
                dst.Add(NamedOnnxValue.CreateFromTensor("gt_mel", new DenseTensor<float>(gtMel, new[] { 1, nFrames, melBins })));
            }

            if (!(anyKeep && anyReroll))
            {
                // 非混合：单趟。retake 全 true、gt_mel 全 0、噪声按 seed 轨（seed 全 0 即 take-0）。
                var ins = new List<NamedOnnxValue>(cond);
                AddRetake(ins, allTrue, zeroMel);
                DiffSingerNoise.AddNoise(ins, ac, seedAcousticCurve, DiffSingerNoise.StageAcoustic, nFrames);
                mel = RunAcoustic(ins);
            }
            else
            {
                // 混合：take-0 参照（seed=0 噪声，输入恒定可缓存）→ 正式趟（gt_mel=参照、噪声按 seed 轨）。
                var refIns = new List<NamedOnnxValue>(cond);
                AddRetake(refIns, allTrue, zeroMel);
                DiffSingerNoise.AddNoise(refIns, ac, 0u, DiffSingerNoise.StageAcoustic, nFrames);
                var refMel = RunAcoustic(refIns).ToArray();

                var ins = new List<NamedOnnxValue>(cond);
                AddRetake(ins, retakeMask, refMel);
                DiffSingerNoise.AddNoise(ins, ac, seedAcousticCurve, DiffSingerNoise.StageAcoustic, nFrames);
                mel = RunAcoustic(ins);
            }
        }
        progress?.Report(0.75);
        if (cancellation.IsCancellationRequested)
            return null;

        // —— 声码器：mel (+ f0) → 波形 ——
        var voc = models.Vocoder;
        var vInputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("mel", mel) };
        if (voc.HasInput("f0"))
            vInputs.Add(NamedOnnxValue.CreateFromTensor("f0", new DenseTensor<float>(f0, new[] { 1, nFrames })));
        var wavOut = DiffSingerTensorCache.Run(voc, models.VocoderHash, vInputs, mTensorCache);
        var audio = wavOut.First(v => v.Name == "waveform").AsTensor<float>().ToArray();
        progress?.Report(1.0);

        // —— 音素产物（按归属 note 键，每 note 一个 SynthesizedSyllable = 引导/主体双列表 + BodyOffset，只报标称
        //   几何；定位 / 跨 note 去重叠 / melisma 归宿主，见 §5.7）——
        //   时长 = 解析出的 EndTime − StartTime（核含 melisma 填充量、辅音为固定长）；权重核 1 / 辅音 0。
        //   引导 / 主体归属由 phonemizer 携于 PhonemeSpan.IsLeading（自由 note 按分组、钉死 note 按宿主双列表）；
        //   BodyOffset（有符号）= 结合线 junction − note 头，几何派生（主体存在取主体首起点、否则取引导末）——与旧 Preutterance
        //   口径同源、随解析几何自洽；BodyOffset=0 即核精确落 note 头。用户锁定回填后再喂 PhonemeLayout 与显示自洽（WYSIWYG）。
        //   归属：p.NoteIndex 对齐 snapshot.Notes，故以 origins[NoteIndex] 为键回指活 note（仅作身份 token）。
        var byNote = new Dictionary<int, List<PhonemeSpan>>();
        foreach (var p in phones)
        {
            if (!byNote.TryGetValue(p.NoteIndex, out var list))
                byNote[p.NoteIndex] = list = new List<PhonemeSpan>();
            list.Add(p);
        }
        var phonemes = new Map<IVoiceSynthesisNote, SynthesizedSyllable>();
        foreach (var kvp in byNote)
        {
            double noteHead = notes[kvp.Key].StartTime;
            var leading = new List<SynthesizedPhoneme>();
            var body = new List<SynthesizedPhoneme>();
            double junction = noteHead;   // 结合线 = 主体首音素起点；无主体则退取引导末、皆无则 note 头（正常不触发）
            foreach (var p in kvp.Value)   // phonemizer 按 note 内序（引导在前、主体在后）发射
            {
                var descriptor = new SynthesizedPhoneme
                {
                    Symbol = CleanSymbol(p.Symbol),   // 剥 lang/ 前缀：显示/钉死符号保持干净，语种走 per-phoneme 属性
                    Duration = Math.Max(0, p.EndTime - p.StartTime),
                    StretchWeight = p.IsVowel ? 1 : 0,
                };
                if (p.IsLeading) { leading.Add(descriptor); junction = p.EndTime; }
                else { if (body.Count == 0) junction = p.StartTime; body.Add(descriptor); }
            }
            phonemes.Add(origins[kvp.Key], new SynthesizedSyllable(leading, body, junction - noteHead));
        }

        return new RenderResult(audio, renderStart, sr, phonemes, pitchReadback, varReadback);
    }

    // 无 dur 预测器兜底：每 note 一元音、占满 note 时长（无对齐/无 head/tail 之外的处理）。
    static List<PhonemeSpan> FallbackPhonemes(VoiceModels models, IReadOnlyList<VoiceSynthesisNoteSnapshot> notes, string[] noteLang)
    {
        var result = new List<PhonemeSpan>(notes.Count);
        for (int i = 0; i < notes.Count; i++)
        {
            string sym = PickVowelSymbol(models, noteLang[i]);
            result.Add(new PhonemeSpan(sym, notes[i].StartTime, notes[i].EndTime, i, true, false));   // 元音起手核 = 主体（非引导）
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

    // 上报显示用：剥掉 lang/ 前缀（语种走 per-phoneme 属性，符号保持干净）。语言无关符号（SP/AP/cl…）无前缀、原样返回。
    static string CleanSymbol(string symbol)
    {
        int slash = symbol.IndexOf('/');
        return slash > 0 ? symbol[(slash + 1)..] : symbol;
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
    static float[] BuildCurveInput(VoiceSynthesisSnapshot snapshot, string key, double neutral,
        Func<double, double> convert, double[] frameTimes, int n)
    {
        double[]? user = snapshot.Automations.TryGetValue(key, out var auto)
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
    // 从 part 属性抽出 model/version 选择（解析入参）。两种属性形态：快照 PropertyObject（Render）与实时只读外观（构造/订阅）。
    static ResolveProps PropsOf(PropertyObject p)
        => new(NullIfEmpty(p.GetString(KeyModel, string.Empty)), NullIfEmpty(p.GetString(KeyVersion, string.Empty)));

    static ResolveProps PropsOf(IReadOnlyNotifiablePropertyObject p)
        => new(NullIfEmpty(LiveString(p, KeyModel)), NullIfEmpty(LiveString(p, KeyVersion)));

    static string LiveString(IReadOnlyNotifiablePropertyObject p, string key)
        => p.GetValue(key, PropertyValue.Create(string.Empty)).ToString(out var s) ? s : string.Empty;

    static string? NullIfEmpty(string s) => string.IsNullOrEmpty(s) ? null : s;

    // gender 归一化 [-1,1]（旧 [-100,100] 的 /100 已并入刻度）：正 = formant 下移；缩放由声库增广范围 KeyShift* 定。
    static Func<double, double> GenderConvert(VoicebankConfig config)
    {
        double posScale = config.KeyShiftMax == 0 ? 0 : 12 / config.KeyShiftMax;
        double negScale = config.KeyShiftMin == 0 ? 0 : -12 / config.KeyShiftMin;
        return x => x < 0 ? -x * posScale : -x * negScale;
    }

    // VELC convert（OpenUtau DiffSingerRenderer）：对数标度，speed 归一化后 1 = 原速，每 +1 速度 ×2。
    static double SpeedConvert(double x) => Math.Pow(2, x - 1);

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
    // 合成音高（具名富类型）：各已合成 piece 的逐帧回显聚为分段折线（每 piece 一段、段间断开）；失败 / 未合成块不报。
    public SynthesizedPitch SynthesizedPitch
    {
        get
        {
            var segments = new List<IReadOnlyList<Point>>();
            foreach (var piece in mPieces)
                if (!piece.Failed && piece.Segment != null && piece.PitchReadback.Count > 0)
                    segments.Add(piece.PitchReadback);
            return new SynthesizedPitch { Segments = segments };
        }
    }

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
                    if (piece.VarianceReadback.TryGetValue(kvp.Key.Id, out var segment) && segment.Count > 0)
                        segments.Add(segment);
                if (segments.Count > 0)
                    map.Add(kvp.Key.Id, new SynthesizedParameter { Segments = segments });
            }
            return map;
        }
    }

    // 合成音素（按归属 note 键）：各已合成 piece 的 note→音节（SynthesizedSyllable = 引导/主体双列表 + BodyOffset）
    //   并入一张 map（块间 note 不相交，直接并）；只报 Symbol / Duration / StretchWeight + note 级引导/主体归属与 BodyOffset——
    //   定位 / 去重叠 / melisma 归宿主（见 §5.7）。失败 / 未合成块不报。
    public IReadOnlyMap<IVoiceSynthesisNote, SynthesizedSyllable> SynthesizedPhonemes
    {
        get
        {
            var result = new Map<IVoiceSynthesisNote, SynthesizedSyllable>();
            foreach (var piece in mPieces)
            {
                if (piece.Failed || piece.Segment == null)
                    continue;
                foreach (var kvp in piece.Phonemes)
                    result.Add(kvp.Key, kvp.Value);
            }
            return result;
        }
    }

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

    // 更新信号按产物分离（SDK 契约）。本会话的产物三者同源——piece 完成 / 失败 / 重分块时一起变，故 NotifyProducts 一并 fire；
    //   高频的进度（StatusChanged，逐 tick）单独 fire、不带动产物重读。
    // 出方向产物/状态事件（SDK 已统一为 IActionEvent）：各以宿主 ActionEvent 具体类做后备，对外暴露只读订阅面、自身 Invoke。
    public IActionEvent SynthesizedPhonemesChanged => mSynthesizedPhonemesChanged;
    public IActionEvent SynthesizedParametersChanged => mSynthesizedParametersChanged;
    public IActionEvent SynthesizedPitchChanged => mSynthesizedPitchChanged;
    public IActionEvent StatusChanged => mStatusChanged;
    readonly ActionEvent mSynthesizedPhonemesChanged = new();
    readonly ActionEvent mSynthesizedParametersChanged = new();
    readonly ActionEvent mSynthesizedPitchChanged = new();
    readonly ActionEvent mStatusChanged = new();

    void NotifyProducts()
    {
        mSynthesizedPhonemesChanged.Invoke();
        mSynthesizedParametersChanged.Invoke();
        mSynthesizedPitchChanged.Invoke();
        mStatusChanged.Invoke();
    }

    public void Dispose()
    {
        mNoteFieldModified.Unsubscribe(OnNoteModified);
        mContext.Notes.MembershipModified.Unsubscribe(OnNotesStructureChanged);
        mContext.PartProperties.Modified.Unsubscribe(MarkAllDirtyAndResegment);
        mContext.Pitch.RangeModified.Unsubscribe(OnRangeModified);
        mContext.PitchDeviation.RangeModified.Unsubscribe(OnRangeModified);
        foreach (var automation in mSubscriptions.Values)
            automation.RangeModified.Unsubscribe(OnRangeModified);
        mContext.Committed.Unsubscribe(OnCommitted);
        foreach (var piece in mPieces)
            piece.Segment?.Dispose();
        mPieces.Clear();
        // 模型会话归引擎级缓存所有、跨会话共享，不在此释放（引擎 Destroy 统一释放）。
    }

    // —— 分块（数据线程；按 note 间隙分块，note 集等价的块保留缓存与状态）——见 §5.9 重叠陷阱 ——
    void Resegment()
    {
        mNeedResegment = false;

        var groups = new List<List<IVoiceSynthesisNote>>();
        List<IVoiceSynthesisNote>? current = null;
        double groupMaxEnd = 0;
        foreach (var note in mContext.Notes)
        {
            if (current == null || note.StartTime.Value > groupMaxEnd)
            {
                current = new List<IVoiceSynthesisNote>();
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
        // 重分块：块集合 / 脏态变 → 产物报告随之变化（旧块的音素 / 回显可能不再在新块集中）。
        NotifyProducts();
    }

    // 任一 note 的任一可订阅字段变更（WhenAnyItem 携带触发的成员）：精确标脏含该 note 的块。
    void OnNoteModified(IVoiceSynthesisNote note)
    {
        foreach (var piece in mPieces)
            if (piece.Notes.Contains(note)) { piece.Dirty = true; piece.Failed = false; }
        mNeedResegment = true;
    }

    void OnNotesStructureChanged() => mNeedResegment = true;

    void MarkAllDirtyAndResegment()
    {
        foreach (var piece in mPieces) { piece.Dirty = true; piece.Failed = false; }
        mNeedResegment = true;
        // part 属性变更（含 model/version 切换）可能增删了固定/seed/混音轨：补订新出现的、退订已消失的，使后续画曲线能标脏。
        //   时序安全：宿主 OnPartPropertiesModified（part 构造期订阅）先于本会话 handler（会话构造期订阅）执行，
        //   它已 RebuildAutomationConfigs 填好 Voice.AutomationConfigs，故此刻 context.Automations 已反映新解析包的轨集。
        SyncAutomationSubscriptions();
    }

    // 同步「区间编辑失效」订阅到当前应有的轨集：固定轨(variance/gender/speed，按解析包能力 gating)、seed 轨(按 retake
    //   gating)、说话人混合轨(按 part 属性选择)——三类都随 model/version 与 part 属性变化而增删。统一幂等 re-sync：
    //   只订阅宿主当前已声明(live)的轨（gating 掉的轨不在 context.Automations 里、无从订阅，故必须按出现/消失增删，
    //   而非构造期一次性订全），故 OnRangeModified 收到的恒是有效轨、无需再判条件。可反复调。
    void SyncAutomationSubscriptions()
    {
        if (mResolve(PropsOf(mContext.PartProperties)) is not { } pc)
            return;

        // 当前应订的轨集 = 固定轨(已 gating) ∪ 全量候选混音轨；混音候选里未选中的不会 live，下方按 live 过滤。
        var desired = new HashSet<string>(StringComparer.Ordinal);
        foreach (var key in BuildFixedAutomationConfigs(pc.Config, RetakeOf(pc.Resolved)).Keys)
            desired.Add(key.Id);
        foreach (var (key, _) in MixTrackKeys(pc.Resolved))
            desired.Add(key);

        // 补订：应有且宿主已声明(live)、尚未订阅的。
        foreach (var key in desired)
            if (!mSubscriptions.ContainsKey(key) && mContext.Automations.TryGetValue(key, out var automation))
            {
                automation.RangeModified.Subscribe(OnRangeModified);
                mSubscriptions[key] = automation;
            }
        // 退订：已订阅但不再应有（gating 掉）、或宿主已撤下该轨(混音被取消选择)的。
        foreach (var key in mSubscriptions.Keys.ToList())
            if (!desired.Contains(key) || !mContext.Automations.ContainsKey(key))
            {
                mSubscriptions[key].RangeModified.Unsubscribe(OnRangeModified);
                mSubscriptions.Remove(key);
            }
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
        mStatusChanged.Invoke();
    }

    sealed record RenderResult(float[] Audio, double StartTime, int SampleRate,
        IReadOnlyMap<IVoiceSynthesisNote, SynthesizedSyllable> Phonemes, List<Point> PitchReadback,
        Dictionary<string, IReadOnlyList<Point>> VarianceReadback);

    sealed class Piece
    {
        public required IReadOnlyList<IVoiceSynthesisNote> Notes;
        public double StartTime;
        public double EndTime;
        public bool Dirty;
        public bool Failed;
        public bool Synthesizing;
        public string? Error;
        public double Progress;
        public IAudioSegment? Segment;
        public IReadOnlyMap<IVoiceSynthesisNote, SynthesizedSyllable> Phonemes = new Map<IVoiceSynthesisNote, SynthesizedSyllable>();
        public IReadOnlyList<Point> PitchReadback = [];
        public IReadOnlyDictionary<string, IReadOnlyList<Point>> VarianceReadback = new Dictionary<string, IReadOnlyList<Point>>();
    }
}

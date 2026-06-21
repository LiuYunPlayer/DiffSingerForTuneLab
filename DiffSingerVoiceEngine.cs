using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// DiffSinger voice 引擎入口。
// 职责：
//   - 经扩展设置接收「声库根目录」与「执行设备」（随宿主持久化、与工程无关）；
//   - 扫描声库目录、缓存为 VoiceSourceInfos（get 仅返缓存、不阻塞，扫描在 Init/ApplySettings 期做）；
//   - 为每条 part 建一个合成会话（会话承载分块/失效/产物管线，后续实现）。
// 声库以原生 DiffSinger 目录形态存在、与本插件版本解耦：插件不打包声库，只按结构特征扫描发现。
public sealed class DiffSingerVoiceEngine : IVoiceEngine, IExtensionSettings
{
    const string KeyVoicebankDirs = "voicebank_dirs";
    const string KeyExecutionProvider = "execution_provider";
    const string KeySamplingSteps = "sampling_steps";
    const string KeyTensorCache = "tensor_cache";
    const string KeyCacheMaxSizeMb = "cache_max_size_mb";

    public IReadOnlyOrderedMap<string, VoiceSourceInfo> VoiceSourceInfos => mState.Infos;

    public void Init()
    {
        EnsureDefaultDirectory();
        Rescan();
    }

    public void Destroy()
    {
        mModelCache?.Dispose();
        mModelCache = null;
    }

    // —— 并发渲染限流（DirectML 多轨并发需要限制同时渲染的轨数）——
    SemaphoreSlim mRenderSemaphore = new(1, 1);

    internal void UpdateRenderSemaphore(int maxConcurrent)
    {
        maxConcurrent = Math.Max(1, maxConcurrent);
        var old = Interlocked.Exchange(ref mRenderSemaphore, new SemaphoreSlim(maxConcurrent, maxConcurrent));
        old.Dispose();
    }

    internal SemaphoreSlim RenderSemaphore => mRenderSemaphore;

    // —— 会话注册表（引擎级，跨会话共享）——
    readonly List<WeakReference<DiffSingerSynthesisSession>> mSessions = new();

    internal void RegisterSession(DiffSingerSynthesisSession session)
    {
        lock (mSessions)
        {
            // 清理已回收的弱引用
            mSessions.RemoveAll(wr => !wr.TryGetTarget(out _));
            mSessions.Add(new WeakReference<DiffSingerSynthesisSession>(session));
        }
    }

    internal void UnregisterSession(DiffSingerSynthesisSession session)
    {
        lock (mSessions)
        {
            mSessions.RemoveAll(wr => !wr.TryGetTarget(out var s) || s == session);
        }
    }

    internal DiffSingerSynthesisSession? FindSessionByVoiceId(string voiceId)
    {
        lock (mSessions)
        {
            // 清理已回收
            mSessions.RemoveAll(wr => !wr.TryGetTarget(out _));
            // 取最后一个匹配 voiceId 的会话（通常只有一个）
            DiffSingerSynthesisSession? found = null;
            foreach (var wr in mSessions)
            {
                if (wr.TryGetTarget(out var s) && s.VoiceId == voiceId)
                    found = s;
            }
            return found;
        }
    }

    public ISynthesisSession CreateSession(string voiceId, ISynthesisContext context)
    {
        if (!mState.Banks.ContainsKey(voiceId))
            throw new ArgumentException($"未知声库 voiceId: {voiceId}");

        // 推理走引擎级模型缓存（懒加载、按 voiceId 共享）；声明面（轨/面板）已上移到引擎方法、建会话前即填好。
        var config = ConfigFor(voiceId)!;
        var samplingSteps = mSettings.GetInt(KeySamplingSteps, 20);
        var tensorCache = mSettings.GetBool(KeyTensorCache, true);
        var cacheMaxSizeMb = mSettings.GetInt(KeyCacheMaxSizeMb, 4096);
        return new DiffSingerSynthesisSession(config, context, voiceId, EnsureModelCache(),
            samplingSteps, tensorCache, cacheMaxSizeMb);
    }

    // —— 声明（引擎层、纯函数 of (voiceId, part 值)；宿主在每次 part 参数 commit 时按当前值重算 diff 到 UI）——
    //   据 context.VoiceId 取声库能力集，委托 DiffSingerDeclarations 建轨/面板。未知声库 → 空声明（不抛，见接口契约）。
    public IReadOnlyOrderedMap<PropertyKey, AutomationConfig> GetAutomationConfigs(IPartPropertyContext context)
        => ConfigFor(context.VoiceId) is { } c ? DiffSingerDeclarations.BuildAutomationConfigs(c, context.PartProperties) : EmptyAutomations;

    public IReadOnlyOrderedMap<PropertyKey, AutomationConfig> GetSynthesizedParameterConfigs(IPartPropertyContext context)
        => ConfigFor(context.VoiceId) is { } c ? DiffSingerDeclarations.BuildReadbackConfigs(c) : EmptyAutomations;

    public ObjectConfig GetPartPropertyConfig(IPartPropertyContext context)
        => ConfigFor(context.VoiceId) is { } c ? DiffSingerDeclarations.BuildPartConfig(c, context) : EmptyConfig;

    public ObjectConfig GetNotePropertyConfig(INotePropertyContext context)
    {
        if (ConfigFor(context.VoiceId) is not { } c)
            return EmptyConfig;

        var baseConfig = DiffSingerDeclarations.BuildNoteConfig(c, context);
        var voiceId = context.VoiceId;
        double selStart = context.SelectionStartTime;
        double selEnd = context.SelectionEndTime;
        bool hasSelection = !double.IsNaN(selStart) && !double.IsNaN(selEnd);

        var props = new OrderedMap<string, IControllerConfig>();
        foreach (var kvp in baseConfig.Properties)
            props.Add(kvp.Key, kvp.Value);

        // —— 音素编辑器 ——
        var phonemesJson = context.NoteProperties.GetString("_phonemes", "[]");
        var phonemeEntries = ParsePhonemeJson(phonemesJson);
        if (phonemeEntries.Count == 0 && c.Languages.Count > 0 && !double.IsNaN(selStart))
        {
            // 未自定义音素时，尝试从会话获取默认 G2P 音素
            double midTime = (selStart + selEnd) / 2;
            var session = FindSessionByVoiceId(voiceId);
            if (session != null)
            {
                // 从上下文快照取段落语言（保证段落语言变更后能传新值给 G2P）
                string snapPartLang = context.PartProperties.GetString(DiffSingerDeclarations.KeyLanguage, string.Empty);
                phonemeEntries = session.GetDefaultPhonemesForNote(midTime, snapPartLang);
            }
        }
        if (c.Languages.Count > 0)
        {
            props.Add("_phoneme_editor", new PhonemeEditorConfig
            {
                DisplayText = L.Tr("Phonemes"),
                Phonemes = phonemeEntries,
                AvailableLanguages = c.Languages,
                LanguageDataKey = DiffSingerDeclarations.KeyLanguage,
                CanDeleteConsonant = phonemeEntries.Count(e => !e.IsVowel) > 1,
                CanDeleteVowel = phonemeEntries.Count(e => e.IsVowel) > 1,
                OnChanged = _ =>
                {
                    // 音素被编辑 → 触发区间重渲染
                    var s = FindSessionByVoiceId(voiceId);
                    if (s != null)
                    {
                        if (hasSelection)
                            s.RequestRetakeScoped(selStart, selEnd);
                        else
                            s.RequestRetake();
                    }
                },
            });
        }

        // 追加 Retake 和 Redraw Pitch 按钮（使用选中音符的区间而非全曲）
        props.Add("_retake", new ButtonConfig
        {
            DisplayText = L.Tr("Retake"),
            Action = () =>
            {
                var session = FindSessionByVoiceId(voiceId);
                if (session != null)
                {
                    if (hasSelection)
                        session.RequestRetakeScoped(selStart, selEnd);
                    else
                        session.RequestRetake();
                }
            },
        });
        props.Add("_redraw_pitch", new ButtonConfig
        {
            DisplayText = L.Tr("Redraw Pitch"),
            Action = () =>
            {
                var session = FindSessionByVoiceId(voiceId);
                if (session != null)
                {
                    if (hasSelection)
                        session.RequestRedrawPitchScoped(selStart, selEnd);
                    else
                        session.RequestRedrawPitch();
                }
            },
        });

        return new ObjectConfig { Properties = props };
    }

    // 声库能力集按 voiceId 缓存（声明每次 commit 都调，避免重复解析 dsconfig）；config 随声库不可变，扫描重建时清空。
    VoicebankConfig? ConfigFor(string voiceId)
    {
        if (mConfigCache.TryGetValue(voiceId, out var cached))
            return cached;
        if (!mState.Banks.TryGetValue(voiceId, out var bank))
            return null;
        var config = VoicebankConfig.Load(bank.RootPath, TuneLabContext.Global.GetLogger());
        mConfigCache[voiceId] = config;
        return config;
    }

    // 模型缓存按当前执行设备设置懒建；provider 变更则弃旧建新（旧缓存 Dispose 释放原生会话）。
    DiffSingerModelCache EnsureModelCache()
    {
        var provider = mSettings.GetString(KeyExecutionProvider, "directml");
        if (mModelCache == null || mProviderInUse != provider)
        {
            mModelCache?.Dispose();
            mModelCache = new DiffSingerModelCache(provider, TuneLabContext.Global.GetLogger());
            mProviderInUse = provider;
        }
        return mModelCache;
    }

    // —— 扩展设置（设置 > 扩展 面板，随宿主持久化、跨工程共享）——
    public ObjectConfig GetSettingsConfig(IExtensionSettingsContext context)
    {
        // 标签随键走（SDK 改动）：DisplayText 移到 PropertyKey、config 仅承载值/量程/选项。
        var properties = new OrderedMap<PropertyKey, IControllerConfig>
        {
            {
                (KeyVoicebankDirs, L.Tr("Voicebank directories (separate with ;)")),
                new TextBoxConfig { DefaultValue = "" }
            },
            {
                (KeyExecutionProvider, L.Tr("Execution device")),
                new ComboBoxConfig
                {
                    // 首项即默认值：DirectML（任意 DX12 GPU 可用），运行期初始化失败由合成层回退 CPU。
                    Options = new List<ComboBoxOption>
                    {
                        new("directml", L.Tr("GPU (DirectML)")),
                        new("cpu", L.Tr("CPU")),
                    },
                }
            },
            {
                // 浅扩散采样步数（质量↔速度旋钮）：全局、跨工程共享；由声学合成阶段消费，量程待该阶段校准。
                (KeySamplingSteps, L.Tr("Sampling steps")),
                new SliderConfig
                {
                    DefaultValue = 20, MinValue = 1, MaxValue = 1000, IsInteger = true,
                }
            },
            {
                // 张量缓存总开关：缓存各 ONNX 模型输出，反复合成（撤销重做/重开工程/改动不涉及某块）时复用、免重算。
                (KeyTensorCache, L.Tr("Tensor cache")),
                new CheckBoxConfig { DefaultValue = true }
            },
            {
                // 缓存体积上限（MB）：超限按最近访问时间逐出最旧缓存；0 = 不限制（持久累积、手动清理）。
                (KeyCacheMaxSizeMb, L.Tr("Cache size limit (MB, 0 = unlimited)")),
                new SliderConfig
                {
                    DefaultValue = 4096, MinValue = 0, MaxValue = 102400, IsInteger = true,
                }
            },
        };
        return new ObjectConfig { Properties = properties };
    }

    public void ApplySettings(PropertyObject settings)
    {
        mSettings = settings;
        var maxConcurrent = mSettings.GetInt(KeyMaxConcurrentRenderings, 1);
        UpdateRenderSemaphore(maxConcurrent);
        Rescan();
    }

    // —— 内部：扫描与状态 ——
    void Rescan()
    {
        var logger = TuneLabContext.Global.GetLogger();
        var roots = CollectRoots();
        var banks = VoicebankScanner.Scan(roots, logger);

        var infos = new OrderedMap<string, VoiceSourceInfo>();
        var byId = new Dictionary<string, DiscoveredVoicebank>(StringComparer.Ordinal);
        foreach (var bank in banks)
        {
            infos.Add(bank.VoiceId, bank.Info);
            byId[bank.VoiceId] = bank;
        }

        mState = new State(infos, byId);
        mConfigCache.Clear();   // 声库集变更：弃旧声明缓存（RootPath 可能变）
        logger.Info($"DiffSinger：在 {roots.Count} 个根目录下发现 {banks.Count} 个声库。");
    }

    // 扫描根 = 内置默认目录 + 用户在设置里配置的目录（; 分隔）。
    List<string> CollectRoots()
    {
        var roots = new List<string> { DefaultVoicebankDirectory };
        var configured = mSettings.GetString(KeyVoicebankDirs, string.Empty);
        roots.AddRange(configured.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
        return roots;
    }

    static string DefaultVoicebankDirectory => Path.Combine(DiffSingerDeclarations.UserDataRoot, "Voices");

    // 内置默认目录尽力创建一次，给用户一个放声库的落点；失败不影响其余根目录的扫描。
    static void EnsureDefaultDirectory()
    {
        try { Directory.CreateDirectory(DefaultVoicebankDirectory); }
        catch { }
    }

    // 解析音素 JSON：[{"s":"ja/b","v":false},...]
    static List<PhonemeEntry> ParsePhonemeJson(string json)
    {
        var result = new List<PhonemeEntry>();
        if (string.IsNullOrEmpty(json) || json.Length < 2) return result;
        try
        {
            // 简单的手动 JSON 解析（避免加依赖）
            int i = json.IndexOf('[');
            if (i < 0) return result;
            while (true)
            {
                i = json.IndexOf('{', i);
                if (i < 0) break;
                string entry = json.Substring(i, Math.Min(json.Length - i, json.IndexOf('}', i) - i + 1));
                var phoneme = new PhonemeEntry();
                int sIdx = entry.IndexOf("\"s\":\"");
                if (sIdx >= 0)
                {
                    sIdx += 5;
                    int eIdx = entry.IndexOf('"', sIdx);
                    if (eIdx > sIdx) phoneme.Symbol = entry.Substring(sIdx, eIdx - sIdx);
                }
                phoneme.IsVowel = entry.Contains("\"v\":true") || entry.Contains("\"v\": true");
                result.Add(phoneme);
                i = json.IndexOf('}', i) + 1;
            }
        }
        catch { }
        return result;
    }

    // 不可变扫描结果，整体替换发布：get 侧读引用、扫描侧建好新实例后一次性换上，无需锁。
    sealed record State(
        IReadOnlyOrderedMap<string, VoiceSourceInfo> Infos,
        IReadOnlyDictionary<string, DiscoveredVoicebank> Banks);

    volatile State mState = new(
        new OrderedMap<string, VoiceSourceInfo>(),
        new Dictionary<string, DiscoveredVoicebank>());

    PropertyObject mSettings = PropertyObject.Empty;

    // 声明缓存（声库能力集按 voiceId）与空声明兜底（未知声库 / 引擎未就绪时返回）。
    readonly Dictionary<string, VoicebankConfig> mConfigCache = new(StringComparer.Ordinal);
    static readonly OrderedMap<PropertyKey, AutomationConfig> EmptyAutomations = new();
    static readonly ObjectConfig EmptyConfig = new() { Properties = new OrderedMap<PropertyKey, IControllerConfig>() };

    // 引擎级模型缓存（跨会话共享、按 voiceId/声码器名缓存）；mProviderInUse 记当前缓存所用执行设备。
    DiffSingerModelCache? mModelCache;
    string mProviderInUse = string.Empty;
}

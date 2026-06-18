using System;
using System.Collections.Generic;
using System.IO;
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

    public ISynthesisSession CreateSession(string voiceId, ISynthesisContext context)
    {
        if (!mState.Banks.TryGetValue(voiceId, out var bank))
            throw new ArgumentException($"未知声库 voiceId: {voiceId}");

        // 声明面据声库能力集（dsconfig）暴露属性面板与自动化轨；推理走引擎级模型缓存（懒加载、按 voiceId 共享）。
        var config = VoicebankConfig.Load(bank.RootPath, TuneLabContext.Global.GetLogger());
        var samplingSteps = mSettings.GetInt(KeySamplingSteps, 20);
        return new DiffSingerSynthesisSession(config, context, voiceId, EnsureModelCache(), samplingSteps);
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
        var properties = new OrderedMap<string, IControllerConfig>
        {
            {
                KeyVoicebankDirs,
                new TextBoxConfig { DisplayText = L.Tr("Voicebank directories (separate with ;)"), DefaultValue = "" }
            },
            {
                KeyExecutionProvider,
                new ComboBoxConfig
                {
                    DisplayText = L.Tr("Execution device"),
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
                KeySamplingSteps,
                new SliderConfig
                {
                    DisplayText = L.Tr("Sampling steps"),
                    DefaultValue = 20, MinValue = 1, MaxValue = 1000, IsInteger = true,
                }
            },
        };
        return new ObjectConfig { Properties = properties };
    }

    public void ApplySettings(PropertyObject settings)
    {
        mSettings = settings;
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

    static string DefaultVoicebankDirectory => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "TuneLab", "DiffSinger", "Voices");

    // 内置默认目录尽力创建一次，给用户一个放声库的落点；失败不影响其余根目录的扫描。
    static void EnsureDefaultDirectory()
    {
        try { Directory.CreateDirectory(DefaultVoicebankDirectory); }
        catch { }
    }

    // 不可变扫描结果，整体替换发布：get 侧读引用、扫描侧建好新实例后一次性换上，无需锁。
    sealed record State(
        IReadOnlyOrderedMap<string, VoiceSourceInfo> Infos,
        IReadOnlyDictionary<string, DiscoveredVoicebank> Banks);

    volatile State mState = new(
        new OrderedMap<string, VoiceSourceInfo>(),
        new Dictionary<string, DiscoveredVoicebank>());

    PropertyObject mSettings = PropertyObject.Empty;

    // 引擎级模型缓存（跨会话共享、按 voiceId/声码器名缓存）；mProviderInUse 记当前缓存所用执行设备。
    DiffSingerModelCache? mModelCache;
    string mProviderInUse = string.Empty;
}

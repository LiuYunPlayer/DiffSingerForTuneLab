using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TuneLab.Foundation;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 把扫描到的物理模型包统一合并成「voice(人) → model → version」三级注册表（设计见 docs/tunelab-voicebank-schema.md §8）。
//   统一模型：每个包都按 speaker 产出 voice——
//     · 有 tunelab.yaml 且含 voices[] → 用其声明（全局 voice id、名字/ i18n、白名单、retake、model/version 合并）；
//     · 否则（legacy/未适配）→ 从 dsconfig speakers 自动生成 voice（id=模型id.suffix、名=suffix 或单说话人取 character 名、
//       无白名单/无 i18n/无 retake/单模型单版本）。
//   关键：有无 tunelab.yaml，part 的属性结构都一致（model/version/mix/language）——加 manifest 只丰富、不改结构。
// 注册表在 Init / ApplySettings 期一次性构建（不可变、整体替换发布），i18n 在构建期按宿主当前语言定下。
public sealed class VoiceRegistry
{
    public IReadOnlyOrderedMap<string, VoiceSourceInfo> Infos { get; }

    // 选择器分组布局（与 Infos 平行、只管「怎么摆」）：legacy 多说话人包收成组、其余顶层平铺。空 = 全平铺。
    public IReadOnlyList<VoiceSourceLayoutItem> Layout { get; }

    readonly IReadOnlyDictionary<string, VoiceNode> mVoices;   // 全局 voice id → 节点

    VoiceRegistry(OrderedMap<string, VoiceSourceInfo> infos, Dictionary<string, VoiceNode> voices, IReadOnlyList<VoiceSourceLayoutItem> layout)
    {
        Infos = infos;
        mVoices = voices;
        Layout = layout;
    }

    public static VoiceRegistry Empty { get; } = new(
        new OrderedMap<string, VoiceSourceInfo>(), new Dictionary<string, VoiceNode>(StringComparer.Ordinal), []);

    public bool Contains(string voiceId) => mVoices.ContainsKey(voiceId);

    // —— 构建 ——
    public static VoiceRegistry Build(IEnumerable<string> packagePaths, string? hostLang, ILogger logger)
    {
        var voices = new Dictionary<string, VoiceNode>(StringComparer.Ordinal);
        // 多角色模型（一个模型暴露 >1 个 voice——legacy 自动展开的 speaker、或 manifest 显式声明的 voices 皆算）：
        //   modelId → 本地化模型名。这些 voice 收进以模型名命名的一级组；单角色模型留顶层。分组归属唯一按 voice 的展示 model 定。
        var modelGroupNames = new Dictionary<string, string>(StringComparer.Ordinal);

        foreach (var root in packagePaths)
        {
            var manifest = TunelabManifest.Load(root, logger);
            var meta = CharacterMetadata.Read(root);
            var (model, entries) = PackageVoices(root, manifest, meta, logger);
            if (entries.Count > 1)
                modelGroupNames[model.ModelId] = I18n.Resolve(model.Name, model.NameI18n, hostLang);
            foreach (var v in entries)
            {
                var node = voices.TryGetValue(v.Id, out var existing) ? existing : voices[v.Id] = new VoiceNode(v.Id);
                node.Add(model, v, root, entries, hostLang, logger);
            }
        }

        foreach (var node in voices.Values)
        {
            node.Finish();
            node.Localize(hostLang);
        }

        var ordered = voices.Values.OrderBy(v => v.Info.Name, StringComparer.CurrentCulture).ToList();

        var infos = new OrderedMap<string, VoiceSourceInfo>();
        foreach (var node in ordered)
            infos.Add(node.VoiceId, node.Info);

        // 布局（按模型一级分组）：按展示序遍历；voice 的展示 model 是多角色模型 → 归入以模型名命名的组
        //   （组在首个成员处就位、保序），否则顶层。组内部序 = 遍历序（即声库名序）；voice 只归其展示 model 的组，
        //   故一个模型重导出成多文件夹、或某人跨多模型时都不产生空组。未来要模型内更细层级再在此加 layout。
        var layout = new List<VoiceSourceLayoutItem>();
        var groupItems = new Dictionary<string, List<VoiceSourceLayoutItem>>(StringComparer.Ordinal);
        foreach (var node in ordered)
        {
            if (modelGroupNames.TryGetValue(node.DisplayModelId, out var groupName))
            {
                if (!groupItems.TryGetValue(node.DisplayModelId, out var items))
                {
                    items = groupItems[node.DisplayModelId] = new List<VoiceSourceLayoutItem>();
                    layout.Add(VoiceSourceLayoutItem.Group(groupName, items));   // items 后续原地追加（引用共享）
                }
                items.Add(VoiceSourceLayoutItem.Voice(node.VoiceId));
            }
            else
            {
                layout.Add(VoiceSourceLayoutItem.Voice(node.VoiceId));
            }
        }

        logger.Info($"DiffSinger：注册表——{voices.Count} 个 voice（来自 {packagePaths.Count()} 个包），{groupItems.Count} 个模型分组。");
        return new VoiceRegistry(infos, voices, layout);
    }

    // 一个包暴露的 voice 列表 + 模型层信息。manifest 有 voices 用之；否则从 dsconfig speakers 自动生成。
    static (PkgModel Model, IReadOnlyList<ManifestVoice> Voices) PackageVoices(
        string root, TunelabManifest? manifest, CharacterMetadata.Result meta, ILogger logger)
    {
        if (manifest is not null && manifest.Voices.Count > 0)
        {
            var pm = new PkgModel(manifest.ModelId, manifest.Name.Length > 0 ? manifest.Name : manifest.ModelId,
                manifest.NameI18n, manifest.Version, manifest.VersionLabel, manifest.VersionLabelI18n,
                manifest.ReleasedKey, manifest);
            return (pm, manifest.Voices);
        }

        // legacy/未适配：模型 id 用 manifest.id（若有）否则文件夹名；voice 从 dsconfig speakers 派生。
        string modelId = manifest?.ModelId is { Length: > 0 } mid ? mid : new DirectoryInfo(root).Name;
        string folder = new DirectoryInfo(root).Name;
        string modelName = manifest is not null && manifest.Name.Length > 0 ? manifest.Name
            : (string.IsNullOrWhiteSpace(meta.Name) ? folder : meta.Name!);
        var pmL = new PkgModel(modelId, modelName,
            manifest?.NameI18n ?? EmptyMap, manifest?.Version ?? 0, manifest?.VersionLabel,
            manifest?.VersionLabelI18n ?? EmptyMap, LegacyReleasedKey(root, manifest), manifest);

        var config = VoicebankConfig.Load(root, logger);
        var suffixes = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var s in config.Speakers)
        {
            var suffix = DiffSingerDeclarations.Suffix(s);
            if (seen.Add(suffix)) suffixes.Add(s);   // 保留原始条目（含模型前缀）供 emb 解析
        }

        var list = new List<ManifestVoice>();
        if (suffixes.Count == 0)
        {
            // 无 speakers（单说话人模型，无 spk_embed）：整模型即一个 voice，id=模型id、名取 character。
            list.Add(new ManifestVoice(modelId, string.Empty, modelName, EmptyMap, null, meta.PortraitOrImage, null));
        }
        else
        {
            // voice id = speaker 后缀本身（不加文件夹前缀）⇒ 多个文件夹里的同一 speaker 按 voice id 聚合成一个 voice、
            //   各文件夹成为它的一个 model（文件夹/character 名即 model 显示名）。名用 suffix（model 名在模型下拉里区分）。
            foreach (var entry in suffixes)
            {
                var suffix = DiffSingerDeclarations.Suffix(entry);
                list.Add(new ManifestVoice(suffix, entry, suffix, EmptyMap, null, meta.PortraitOrImage, null));
            }
        }
        return (pmL, list);
    }

    static readonly IReadOnlyDictionary<string, string> EmptyMap =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

    // legacy 跨模型排序键：manifest 有 released 用之；否则取文件夹最后修改时间（最近重导出/更新的当最新）。
    //   定宽串 yyyy-MM-dd-HH-mm-ss，与 manifest 的 YYYY-MM-DD 同走字典序比较。注意：mtime 是本机启发式、非可移植
    //   （重新解压/拷贝会变），仅影响未钉死时的默认 model；要稳定/可移植的新旧请用 manifest 的 released。
    static string LegacyReleasedKey(string root, TunelabManifest? manifest)
    {
        if (manifest?.ReleasedKey is { Length: > 0 } rk)
            return rk;
        try { return Directory.GetLastWriteTimeUtc(root).ToString("yyyy-MM-dd-HH-mm-ss"); }
        catch { return string.Empty; }
    }

    // —— 解析：voiceId + 选择 → 具体物理包 ——
    public ResolvedVoice? Resolve(string voiceId, ResolveProps props)
    {
        if (!mVoices.TryGetValue(voiceId, out var node))
            return null;

        string? pinModel = props.Model;
        var model = (pinModel != null ? node.Models.FirstOrDefault(m => m.ModelId == pinModel) : null)
                    ?? node.Models[0];

        VersionNode version = model.Versions[0];
        if (!string.IsNullOrEmpty(props.Version) && int.TryParse(props.Version, out var pinV))
            version = model.Versions.FirstOrDefault(v => v.Version == pinV) ?? model.Versions[0];

        return new ResolvedVoice(version.RootPath, version.Manifest, version.Speaker, version.Voice, version.PackageVoices);
    }

    // —— 下拉选项（声明面用）——
    // model 下拉：首项 "最新 (当前最新模型名)" sentinel（值=""，浮动跟随最新 released 模型），其后各具体模型（新→旧）。
    //   即便单模型也给出——让用户能在当前状态钉死，将来装了新模型也不被带走。null = 未知 voice。
    public IReadOnlyList<(string Value, string Display)>? ModelOptions(string voiceId)
    {
        if (!mVoices.TryGetValue(voiceId, out var node))
            return null;
        var list = new List<(string, string)> { ("", $"{L.Tr("Latest")} ({node.Models[0].DisplayName})") };
        list.AddRange(node.Models.Select(m => (m.ModelId, m.DisplayName)));
        return list;
    }

    // version 下拉（针对当前选中 model）：首项 "最新 (当前最新版本标签)" sentinel（值=""），其后各具体版本（新→旧）。
    //   即便单版本也给出。null = 未知 voice。
    public IReadOnlyList<(string Value, string Display)>? VersionOptions(string voiceId, string? selectedModelId)
    {
        if (!mVoices.TryGetValue(voiceId, out var node))
            return null;
        var model = (selectedModelId != null ? node.Models.FirstOrDefault(m => m.ModelId == selectedModelId) : null)
                    ?? node.Models[0];
        if (model.IsLegacy)
            return null;   // legacy 模型无版本概念 ⇒ 不出版本下拉
        var list = new List<(string, string)> { ("", $"{L.Tr("Latest")} ({model.Versions[0].Label})") };
        list.AddRange(model.Versions.Select(v => (v.Version.ToString(), v.Label)));
        return list;
    }

    // —— 节点类型 ——
    sealed class VoiceNode
    {
        public string VoiceId { get; }
        public VoiceSourceInfo Info { get; private set; }
        public List<ModelNode> Models { get; } = new();
        // 展示 model = 排序后首个（最新 released）；分组归属看它是不是多角色模型。Finish 后有效。
        public string DisplayModelId => Models[0].ModelId;
        readonly Dictionary<string, ModelNode> mByModel = new(StringComparer.Ordinal);
        PkgModel? mDisplayModel;
        ManifestVoice? mDisplayVoice;
        string mDisplayRoot = string.Empty;

        public VoiceNode(string voiceId) { VoiceId = voiceId; Info = new VoiceSourceInfo { Name = voiceId, Description = string.Empty }; }

        public void Add(PkgModel pm, ManifestVoice voice, string root, IReadOnlyList<ManifestVoice> packageVoices, string? hostLang, ILogger logger)
        {
            var model = mByModel.TryGetValue(pm.ModelId, out var m) ? m : mByModel[pm.ModelId] = AddModel(pm, hostLang);
            model.AddVersion(pm, voice, root, packageVoices, hostLang, logger);
        }

        ModelNode AddModel(PkgModel pm, string? hostLang)
        {
            var m = new ModelNode(pm, hostLang);
            Models.Add(m);
            return m;
        }

        public void Finish()
        {
            foreach (var m in Models)
                m.SortVersions();
            Models.Sort((a, b) =>
            {
                int c = string.CompareOrdinal(b.ReleasedKey, a.ReleasedKey);
                return c != 0 ? c : string.CompareOrdinal(a.ModelId, b.ModelId);
            });
            var defVer = Models[0].Versions[0];
            mDisplayModel = defVer.Model;
            mDisplayVoice = defVer.Voice;
            mDisplayRoot = defVer.RootPath;
        }

        public void Localize(string? hostLang)
        {
            if (mDisplayVoice is null || mDisplayModel is null)
                return;
            string name = I18n.Resolve(
                string.IsNullOrWhiteSpace(mDisplayVoice.Name) ? VoiceId : mDisplayVoice.Name!,
                mDisplayVoice.NameI18n, hostLang);
            string desc = I18n.Resolve(mDisplayModel.Name, mDisplayModel.NameI18n, hostLang);
            ImageResource? portrait = ResolvePortrait(mDisplayRoot, mDisplayVoice.Portrait);
            Info = new VoiceSourceInfo { Name = name, Description = desc, Portrait = portrait };
        }

        static ImageResource? ResolvePortrait(string root, string? portraitFile)
        {
            if (!string.IsNullOrWhiteSpace(portraitFile))
            {
                var p = Path.Combine(root, portraitFile);
                if (File.Exists(p)) return new FileImageResource(p);
            }
            var meta = CharacterMetadata.Read(root);
            if (!string.IsNullOrWhiteSpace(meta.PortraitOrImage))
            {
                var p = Path.Combine(root, meta.PortraitOrImage);
                if (File.Exists(p)) return new FileImageResource(p);
            }
            return null;
        }
    }

    sealed class ModelNode
    {
        public string ModelId { get; }
        public string ReleasedKey { get; }
        public string DisplayName { get; }
        public bool IsLegacy { get; }   // 无 manifest 背书 ⇒ 无版本概念（单版本、不出版本下拉）
        public List<VersionNode> Versions { get; } = new();

        public ModelNode(PkgModel pm, string? hostLang)
        {
            ModelId = pm.ModelId;
            ReleasedKey = pm.ReleasedKey;
            DisplayName = I18n.Resolve(pm.Name, pm.NameI18n, hostLang);
            IsLegacy = pm.Manifest is null;
        }

        public void AddVersion(PkgModel pm, ManifestVoice voice, string root, IReadOnlyList<ManifestVoice> packageVoices, string? hostLang, ILogger logger)
        {
            if (Versions.Any(v => v.Version == pm.Version))
            {
                logger.Warning($"DiffSinger：模型 {ModelId} 版本 {pm.Version} 重复（{root}），保留先出现者");
                return;
            }
            string label = !string.IsNullOrWhiteSpace(pm.VersionLabel)
                ? I18n.Resolve(pm.VersionLabel!, pm.VersionLabelI18n, hostLang)
                : $"v{pm.Version}";
            Versions.Add(new VersionNode(pm.Version, label, root, voice.Speaker, pm.Manifest, voice, packageVoices, pm));
        }

        public void SortVersions() => Versions.Sort((a, b) => b.Version.CompareTo(a.Version));
    }

    sealed record VersionNode(
        int Version, string Label, string RootPath, string Speaker, TunelabManifest? Manifest,
        ManifestVoice Voice, IReadOnlyList<ManifestVoice> PackageVoices, PkgModel Model);
}

// 包的模型层信息（manifest 取其字段；legacy 取文件夹名 + character）。
public sealed record PkgModel(
    string ModelId, string Name, IReadOnlyDictionary<string, string> NameI18n,
    int Version, string? VersionLabel, IReadOnlyDictionary<string, string> VersionLabelI18n,
    string ReleasedKey, TunelabManifest? Manifest);

// 解析输入：从 part 属性读出的 model/version 选择（空 = 用默认/最新）。
public readonly record struct ResolveProps(string? Model, string? Version);

// 解析结果：voiceId + 选择 → 具体物理包。
//   VoiceSpeaker = 此 voice 在该包的 dsconfig 后缀（单说话人模型为空串）；ExposedVoices = 同包暴露 voice（混音候选）。
//   Manifest 为 null = legacy/未适配包（retake 全关、语言取 dsconfig、无 i18n）。
public sealed record ResolvedVoice(
    string RootPath, TunelabManifest? Manifest, string? VoiceSpeaker,
    ManifestVoice? CurrentVoice, IReadOnlyList<ManifestVoice> ExposedVoices);

// 一个 part 的解析上下文：物理包能力集 + voice 语境 + 注册表（取 model/version 下拉选项）。
public sealed record PartContext(
    VoicebankConfig Config, ResolvedVoice Resolved, VoiceRegistry Registry, string VoiceId);

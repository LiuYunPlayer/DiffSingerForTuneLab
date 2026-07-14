using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TuneLab.SDK;
using YamlDotNet.Serialization;

namespace DiffSingerForTuneLab;

// tunelab.yaml —— 模型的 TuneLab 描述（可选叠加层）。缺失 → 纯走 OpenUtau 加载逻辑（= 今天行为）。
// 只承载“作者决策层”：模型/voice 身份、版本、retake 声明、语言显示、i18n、speaker 白名单；
// 不复制 dsconfig 的任何 DSP/模型事实（那些仍由 VoicebankConfig 从 dsconfig 读）。
// 解析失败或文件缺失 → 返回 null，由调用方降级到 OpenUtau 路径。设计见 docs/tunelab-voicebank-schema.md。
public sealed class TunelabManifest
{
    public const string FileName = "tunelab.yaml";

    public string ModelId { get; private init; } = string.Empty;
    public string Name { get; private init; } = string.Empty;
    public IReadOnlyDictionary<string, string> NameI18n { get; private init; } = EmptyMap;
    public int Version { get; private init; }
    public string? VersionLabel { get; private init; }
    public IReadOnlyDictionary<string, string> VersionLabelI18n { get; private init; } = EmptyMap;
    public string? Released { get; private init; }

    public bool RetakeAcoustic { get; private init; }
    public bool RetakePitch { get; private init; }
    public bool RetakeVariance { get; private init; }

    // 音素混合能力：模型（acoustic + pitch/variance role）已重导出、带 tokens_b/blend/encoder_out_b 输入。
    //   缺省 false ⇒ 不暴露音素混合 UI（老库/未重导出的库）。
    public bool PhonemeMix { get; private init; }

    public string? DefaultLanguage { get; private init; }
    // 语言显示叠加（id 必须匹配 dsconfig 语言表的键）；空 = 不限定，用 dsconfig 全部、显示裸 id。
    public IReadOnlyList<ManifestLanguage> Languages { get; private init; } = [];
    // 暴露的 voice 白名单；空 = 无白名单（退化为整模型 1 voice + 全 speaker 下拉）。
    public IReadOnlyList<ManifestVoice> Voices { get; private init; } = [];

    static readonly IReadOnlyDictionary<string, string> EmptyMap =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

    // 归一化 released 为可比较的完整日期串 YYYY-MM-DD（缺省段按“最早”补 -01）；无 released → ""（视作最旧）。
    //   定宽串字典序 == 时间序。
    public string ReleasedKey => NormalizeReleased(Released);

    // 解析 rootPath/tunelab.yaml；不存在或失败返回 null（降级到 OpenUtau 路径）。
    public static TunelabManifest? Load(string rootPath, ILogger logger)
    {
        var path = Path.Combine(rootPath, FileName);
        if (!File.Exists(path))
            return null;

        try
        {
            var dto = new DeserializerBuilder().IgnoreUnmatchedProperties().Build()
                .Deserialize<Dto>(File.ReadAllText(path));
            if (dto is null)
                return null;

            var manifest = new TunelabManifest
            {
                ModelId = dto.id?.Trim() ?? string.Empty,
                Name = dto.name?.Trim() ?? string.Empty,
                NameI18n = Map(dto.name_i18n),
                Version = dto.version,
                VersionLabel = NullIfBlank(dto.version_label),
                VersionLabelI18n = Map(dto.version_label_i18n),
                Released = NullIfBlank(dto.released),
                RetakeAcoustic = dto.retake?.acoustic ?? false,
                RetakePitch = dto.retake?.pitch ?? false,
                RetakeVariance = dto.retake?.variance ?? false,
                PhonemeMix = dto.phoneme_mix ?? false,
                DefaultLanguage = NullIfBlank(dto.languages?.@default),
                Languages = (dto.languages?.expose ?? new())
                    .Where(e => !string.IsNullOrWhiteSpace(e.id))
                    .Select(e => new ManifestLanguage(e.id!.Trim(), NullIfBlank(e.name), Map(e.name_i18n)))
                    .ToList(),
                Voices = (dto.voices ?? new())
                    .Where(v => !string.IsNullOrWhiteSpace(v.id) && !string.IsNullOrWhiteSpace(v.speaker))
                    .Select(v => new ManifestVoice(
                        v.id!.Trim(), v.speaker!.Trim(), NullIfBlank(v.name), Map(v.name_i18n),
                        NullIfBlank(v.default_language), NullIfBlank(v.portrait), NullIfBlank(v.color)))
                    .ToList(),
            };

            if (string.IsNullOrEmpty(manifest.ModelId))
            {
                logger.Warning($"DiffSinger：{path} 缺少 id，忽略该 tunelab.yaml");
                return null;
            }
            return manifest;
        }
        catch (Exception ex)
        {
            logger.Warning($"DiffSinger：解析 {path} 失败，降级到 OpenUtau 路径：{ex.Message}");
            return null;
        }
    }

    static string NormalizeReleased(string? r)
    {
        if (string.IsNullOrWhiteSpace(r))
            return string.Empty;
        var p = r.Trim().Split('-');
        string y = (p.Length > 0 ? p[0] : "0000").PadLeft(4, '0');
        string m = (p.Length > 1 ? p[1] : "01").PadLeft(2, '0');
        string d = (p.Length > 2 ? p[2] : "01").PadLeft(2, '0');
        return $"{y}-{m}-{d}";
    }

    static string? NullIfBlank(string? s) => string.IsNullOrWhiteSpace(s) ? null : s.Trim();

    static IReadOnlyDictionary<string, string> Map(Dictionary<string, string>? src)
    {
        if (src is null || src.Count == 0)
            return EmptyMap;
        var m = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var kv in src)
            if (!string.IsNullOrWhiteSpace(kv.Key) && kv.Value != null)
                m[kv.Key.Trim()] = kv.Value;
        return m;
    }

    // —— YamlDotNet DTO（snake_case 经 alias 映射；IgnoreUnmatchedProperties 容忍未来字段）——
    sealed class Dto
    {
        public string? id { get; set; }
        public string? name { get; set; }
        [YamlMember(Alias = "name_i18n")] public Dictionary<string, string>? name_i18n { get; set; }
        public int version { get; set; }
        [YamlMember(Alias = "version_label")] public string? version_label { get; set; }
        [YamlMember(Alias = "version_label_i18n")] public Dictionary<string, string>? version_label_i18n { get; set; }
        public string? released { get; set; }
        public RetakeDto? retake { get; set; }
        [YamlMember(Alias = "phoneme_mix")] public bool? phoneme_mix { get; set; }
        public LanguagesDto? languages { get; set; }
        public List<VoiceDto>? voices { get; set; }
    }

    sealed class RetakeDto
    {
        public bool acoustic { get; set; }
        public bool pitch { get; set; }
        public bool variance { get; set; }
    }

    sealed class LanguagesDto
    {
        [YamlMember(Alias = "default")] public string? @default { get; set; }
        public List<LangExposeDto>? expose { get; set; }
    }

    sealed class LangExposeDto
    {
        public string? id { get; set; }
        public string? name { get; set; }
        [YamlMember(Alias = "name_i18n")] public Dictionary<string, string>? name_i18n { get; set; }
    }

    sealed class VoiceDto
    {
        public string? id { get; set; }
        public string? speaker { get; set; }
        public string? name { get; set; }
        [YamlMember(Alias = "name_i18n")] public Dictionary<string, string>? name_i18n { get; set; }
        [YamlMember(Alias = "default_language")] public string? default_language { get; set; }
        public string? portrait { get; set; }
        public string? color { get; set; }
    }
}

// 一个暴露的 voice 在某模型里的声明（id = 全局 voice id；speaker = 本模型 dsconfig 后缀）。
public sealed record ManifestVoice(
    string Id, string Speaker, string? Name, IReadOnlyDictionary<string, string> NameI18n,
    string? DefaultLanguage, string? Portrait, string? Color);

// 语言显示叠加（id 引用 dsconfig 语言表键）。
public sealed record ManifestLanguage(
    string Id, string? Name, IReadOnlyDictionary<string, string> NameI18n);

// i18n 解析：按宿主 culture（en-US/zh-CN）取本地化串，回退顺序 精确 → 纯语言码 → 基准串。
//   大小写不敏感、容忍作者写成 en-us/en（见 docs §10.4）。
public static class I18n
{
    public static string Resolve(string baseStr, IReadOnlyDictionary<string, string> map, string? hostLang)
    {
        if (map.Count == 0 || string.IsNullOrEmpty(hostLang))
            return baseStr;

        if (map.TryGetValue(hostLang, out var exact))   // map 用 OrdinalIgnoreCase 比较器，已大小写不敏感
            return exact;

        var lang = LangPart(hostLang);
        foreach (var kv in map)
            if (string.Equals(LangPart(kv.Key), lang, StringComparison.OrdinalIgnoreCase))
                return kv.Value;

        return baseStr;
    }

    static string LangPart(string locale)
    {
        int dash = locale.IndexOf('-');
        return dash > 0 ? locale[..dash] : locale;
    }
}

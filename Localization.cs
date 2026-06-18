using System.Collections.Generic;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 插件侧自译：宿主不参与查表，插件读当前宿主语言后用自带词典出文案，未收录词回退原文（英文）。
// 面向最终用户的文案（设置项标题、下拉选项、声库名等）经此本地化。
internal static class L
{
    static readonly Dictionary<string, Dictionary<string, string>> Dict = new()
    {
        ["zh-CN"] = new()
        {
            ["Voicebank directories (separate with ;)"] = "声库目录（多个用 ; 分隔）",
            ["Execution device"] = "执行设备",
            ["GPU (DirectML)"] = "GPU（DirectML）",
            ["CPU"] = "CPU",
        },
    };

    public static string Tr(string text)
    {
        var lang = TuneLabContext.Global.Language;
        return Dict.TryGetValue(lang, out var m) && m.TryGetValue(text, out var v) ? v : text;
    }
}

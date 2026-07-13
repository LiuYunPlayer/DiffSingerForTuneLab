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
            ["Inference mode"] = "推理模式",
            ["Isolated process (recommended)"] = "隔离进程（推荐）",
            ["In-process"] = "进程内",
            ["Sampling steps"] = "采样步数",
            ["Tensor cache"] = "张量缓存",
            ["Cache size limit (MB, 0 = unlimited)"] = "缓存大小上限（MB，0 = 不限）",
            ["Speaker"] = "说话人",
            ["Speaker mix"] = "说话人混合",
            ["Language"] = "语言",
            ["Model"] = "模型",
            ["Version"] = "版本",
            ["Latest"] = "最新",
            ["(follow note)"] = "（跟随音符）",
            ["Mix phoneme"] = "混合音素",
            ["Mix language"] = "混合语言",
            ["(follow phoneme)"] = "（跟随音素）",
            ["Phoneme mix"] = "音素混合",
            ["Gender"] = "性别",
            ["Speed"] = "语速",
            ["Energy"] = "能量",
            ["Breathiness"] = "气声",
            ["Voicing"] = "发声",
            ["Tension"] = "张力",
            ["Pitch seed"] = "音高种子",
            ["Variance seed"] = "参数种子",
            ["Timbre seed"] = "音色种子",
            ["Synthesizing"] = "合成中",
        },
    };

    public static string Tr(string text)
    {
        var lang = TuneLabContext.Global.Language;
        return Dict.TryGetValue(lang, out var m) && m.TryGetValue(text, out var v) ? v : text;
    }
}

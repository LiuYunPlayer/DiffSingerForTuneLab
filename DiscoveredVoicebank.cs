using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 一个被扫描发现的 DiffSinger 声库。
// VoiceId 是 VoiceSourceInfos 的 key（宿主据此选库、传给 CreateSession）；
// RootPath 是声库根目录绝对路径（合成会话据此加载声学/预测器/声码器模型与发音词典）；
// Info 是暴露给宿主目录的展示元数据。
public sealed record DiscoveredVoicebank(string VoiceId, string RootPath, VoiceSourceInfo Info);

using System.IO;

namespace DiffSingerForTuneLab.G2p;

// 内置算法引擎的 g2p 资源包读取（嵌入本插件程序集，LogicalName = DiffSingerForTuneLab.G2p.Data.<fileName>）。
//   原挂在插件版 G2pPack 上；G2pPack 统一到 OpenUtau.Core 门面（忠实原版、无此插件私有件）后独立成类。
static class EmbeddedG2pPacks
{
    public static byte[] Load(string fileName)
    {
        var asm = typeof(EmbeddedG2pPacks).Assembly;
        var resName = $"DiffSingerForTuneLab.G2p.Data.{fileName}";
        using var s = asm.GetManifestResourceStream(resName)
            ?? throw new FileNotFoundException($"嵌入 g2p 资源未找到：{resName}");
        using var ms = new MemoryStream();
        s.CopyTo(ms);
        return ms.ToArray();
    }
}

namespace OpenUtau.Api {
    // 忠实移植自 OpenUtau（MIT）——OpenUtau.Core/Api/IG2p.cs。见仓库根 THIRD-PARTY-NOTICES.md。
    // G2P 引擎统一契约：词典层(G2pDictionary)、算法层(G2pPack)、重映射(G2pRemapper)、链式兜底(G2pFallbacks) 皆实现本接口。
    // 声库自带音素器 DLL 按 "OpenUtau.Core, 1.0.0.0" 绑定本类型——签名不可动。
    public interface IG2p {
        bool IsValidSymbol(string symbol);
        bool IsVowel(string symbol);

        // 半元音 / 流音（y w l r 之属）。
        bool IsGlide(string symbol);

        // grapheme → 音素串；查无返回 null（交由 fallback 链下一层）。
        string[] Query(string grapheme);

        // 提示串（用户手写音素）→ 过滤掉无效符号后的音素串。
        string[] UnpackHint(string hint, char separator = ' ');
    }
}

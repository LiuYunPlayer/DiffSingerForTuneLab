namespace OpenUtau.Api {
    // 忠实移植自 OpenUtau（MIT）——OpenUtau.Core/Api/G2pDictionaryData.cs。见仓库根 THIRD-PARTY-NOTICES.md。
    // yaml 词典反序列化数据类。
    public class G2pDictionaryData {
        public struct SymbolData {
            public string symbol;
            public string type;
        }

        public struct Entry {
            public string grapheme;
            public string[] phonemes;
        }

        public SymbolData[] symbols;
        public Entry[] entries;
    }
}

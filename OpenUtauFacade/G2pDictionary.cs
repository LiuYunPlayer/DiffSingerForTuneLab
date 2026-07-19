using System.Collections.Generic;
using System.Linq;

namespace OpenUtau.Api {
    // 忠实移植自 OpenUtau（MIT）——OpenUtau.Core/Api/G2pDictionary.cs。见仓库根 THIRD-PARTY-NOTICES.md。
    // 词典层：grapheme→音素，trie 存储（紧凑 + 快查）。符号类型(元音/滑音)随符号声明建立。
    //   与原版唯一差异：去掉依赖 OpenUtau.Core.Yaml 的 Load(string)/Load(TextReader) 重载——
    //   本插件由 DiffSingerPredictor 解析 dsdict 后喂 Builder，保留 Load(G2pDictionaryData)。
    public class G2pDictionary : IG2p {
        class TrieNode {
            public Dictionary<char, TrieNode> children = new Dictionary<char, TrieNode>();
            public string[] symbols;
        }

        TrieNode root;
        Dictionary<string, bool> phonemeSymbols; // (phoneme, isVowel)
        HashSet<string> glideSymbols;

        G2pDictionary(TrieNode root, Dictionary<string, bool> phonemeSymbols, HashSet<string> glideSymbols) {
            this.root = root;
            this.phonemeSymbols = phonemeSymbols;
            this.glideSymbols = glideSymbols;
        }

        public bool IsValidSymbol(string symbol) {
            return phonemeSymbols.ContainsKey(symbol);
        }

        public bool IsVowel(string symbol) {
            return phonemeSymbols.TryGetValue(symbol, out var isVowel) && isVowel;
        }

        public bool IsGlide(string symbol) {
            return glideSymbols.Contains(symbol);
        }

        public string[] Query(string grapheme) {
            return QueryTrie(root, grapheme, 0);
        }

        public string[] UnpackHint(string hint, char separator = ' ') {
            return hint.Split(separator)
                .Where(s => phonemeSymbols.ContainsKey(s))
                .ToArray();
        }

        string[] QueryTrie(TrieNode node, string word, int index) {
            if (index == word.Length) {
                if (node.symbols == null) {
                    return null;
                }
                return node.symbols.Clone() as string[];
            }
            if (node.children.TryGetValue(word[index], out var child)) {
                return QueryTrie(child, word, index + 1);
            }
            return null;
        }

        public class Builder {
            TrieNode root;
            Dictionary<string, bool> phonemeSymbols; // (phoneme, isVowel)
            HashSet<string> glideSymbols;

            internal Builder() {
                root = new TrieNode();
                phonemeSymbols = new Dictionary<string, bool>();
                glideSymbols = new HashSet<string>();
            }

            // 类型字符串：vowel / semivowel / liquid / 其它(辅音)。半元音 + 流音 = 滑音。
            public Builder AddSymbol(string symbol, string type) {
                phonemeSymbols[symbol] = type == "vowel";
                if (type == "semivowel" || type == "liquid") {
                    glideSymbols.Add(symbol);
                } else {
                    glideSymbols.Remove(symbol);
                }
                return this;
            }
            public Builder AddSymbol(string symbol, bool isVowel) {
                phonemeSymbols[symbol] = isVowel;
                return this;
            }
            public Builder AddSymbol(string symbol, bool isVowel, bool isGlide) {
                phonemeSymbols[symbol] = isVowel;
                if (isGlide && !isVowel) {
                    glideSymbols.Add(symbol);
                } else {
                    glideSymbols.Remove(symbol);
                }
                return this;
            }

            // 必须先加完符号再加词条，否则词条里未声明的符号会被丢弃。
            public Builder AddEntry(string grapheme, IEnumerable<string> symbols) {
                BuildTrie(root, grapheme, 0, symbols);
                return this;
            }

            void BuildTrie(TrieNode node, string grapheme, int index, IEnumerable<string> symbols) {
                if (index == grapheme.Length) {
                    node.symbols = symbols
                        .Where(symbol => phonemeSymbols.ContainsKey(symbol))
                        .ToArray();
                    return;
                }
                if (!node.children.TryGetValue(grapheme[index], out var child)) {
                    child = new TrieNode();
                    node.children[grapheme[index]] = child;
                }
                BuildTrie(child, grapheme, index + 1, symbols);
            }

            public Builder Load(G2pDictionaryData data) {
                if (data.symbols != null) {
                    foreach (var symbolData in data.symbols) {
                        AddSymbol(symbolData.symbol, symbolData.type);
                    }
                }
                if (data.entries != null) {
                    foreach (var entry in data.entries) {
                        AddEntry(entry.grapheme, entry.phonemes);
                    }
                }
                return this;
            }

            public G2pDictionary Build() {
                return new G2pDictionary(root, phonemeSymbols, glideSymbols);
            }
        }

        public static Builder NewBuilder() {
            return new Builder();
        }
    }
}

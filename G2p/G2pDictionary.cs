using System.Collections.Generic;
using System.Linq;

namespace DiffSingerForTuneLab.G2p;

// 移植自 OpenUtau（MIT）——OpenUtau.Core/Api/G2pDictionary.cs。见仓库根 THIRD-PARTY-NOTICES.md。
// 词典层：grapheme→音素，trie 存储（紧凑 + 快查）。符号类型(元音/滑音)随符号声明建立。
//   去掉了 OpenUtau 依赖其 Yaml 的 Load 重载——本插件由 DiffSingerPredictor 解析 dsdict 后喂 Builder.AddSymbol/AddEntry。
public sealed class G2pDictionary : IG2p
{
    sealed class TrieNode
    {
        public readonly Dictionary<char, TrieNode> children = new();
        public string[]? symbols;
    }

    readonly TrieNode root;
    readonly Dictionary<string, bool> phonemeSymbols; // (phoneme, isVowel)
    readonly HashSet<string> glideSymbols;

    G2pDictionary(TrieNode root, Dictionary<string, bool> phonemeSymbols, HashSet<string> glideSymbols)
    {
        this.root = root;
        this.phonemeSymbols = phonemeSymbols;
        this.glideSymbols = glideSymbols;
    }

    public bool IsValidSymbol(string symbol) => phonemeSymbols.ContainsKey(symbol);
    public bool IsVowel(string symbol) => phonemeSymbols.TryGetValue(symbol, out var isVowel) && isVowel;
    public bool IsGlide(string symbol) => glideSymbols.Contains(symbol);

    public string[]? Query(string grapheme) => QueryTrie(root, grapheme, 0);

    public string[]? UnpackHint(string hint, char separator = ' ')
        => hint.Split(separator).Where(s => phonemeSymbols.ContainsKey(s)).ToArray();

    static string[]? QueryTrie(TrieNode node, string word, int index)
    {
        if (index == word.Length)
            return node.symbols?.Clone() as string[];
        if (node.children.TryGetValue(word[index], out var child))
            return QueryTrie(child, word, index + 1);
        return null;
    }

    public sealed class Builder
    {
        readonly TrieNode root = new();
        readonly Dictionary<string, bool> phonemeSymbols = new(); // (phoneme, isVowel)
        readonly HashSet<string> glideSymbols = new();

        // 类型字符串：vowel / semivowel / liquid / 其它(辅音)。半元音 + 流音 = 滑音。
        public Builder AddSymbol(string symbol, string type)
        {
            phonemeSymbols[symbol] = type == "vowel";
            if (type == "semivowel" || type == "liquid")
                glideSymbols.Add(symbol);
            else
                glideSymbols.Remove(symbol);
            return this;
        }

        public Builder AddSymbol(string symbol, bool isVowel)
        {
            phonemeSymbols[symbol] = isVowel;
            return this;
        }

        // 必须先加完符号再加词条，否则词条里未声明的符号会被丢弃。
        public Builder AddEntry(string grapheme, IEnumerable<string> symbols)
        {
            BuildTrie(root, grapheme, 0, symbols);
            return this;
        }

        void BuildTrie(TrieNode node, string grapheme, int index, IEnumerable<string> symbols)
        {
            if (index == grapheme.Length)
            {
                node.symbols = symbols.Where(symbol => phonemeSymbols.ContainsKey(symbol)).ToArray();
                return;
            }
            if (!node.children.TryGetValue(grapheme[index], out var child))
            {
                child = new TrieNode();
                node.children[grapheme[index]] = child;
            }
            BuildTrie(child, grapheme, index + 1, symbols);
        }

        public G2pDictionary Build() => new(root, phonemeSymbols, glideSymbols);
    }

    public static Builder NewBuilder() => new();
}

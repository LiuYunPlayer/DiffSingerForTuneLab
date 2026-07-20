using OpenUtau.Api;

namespace OpenUtau.Core.DiffSinger {
    // OpenUtau DiffSingerBasePhonemizer / DiffSingerG2pPhonemizer 的空壳基类（虚方法签名忠实原版——
    //   OpenUtau.Core/DiffSinger/DiffSingerBasePhonemizer.cs 与 Phonemizers/DiffSingerG2pPhonemizer.cs）。
    // 声库自带音素器 DLL 的子类只 override 这些虚方法（G2P 配置数据），不触其余 OpenUtau 管线——
    //   歌词→音素的编排由本插件自己的 DiffSingerPhonemizer/DiffSingerPredictor 完成，故原版基类的
    //   加载/推理逻辑一概不需要；但类型全名与虚槽必须原样，外部 DLL 按此绑定。
    // 原版的 protected 虚方法经 public 只读桥（本门面自有扩展）暴露给插件读取。
    public abstract class DiffSingerBasePhonemizer {
        protected virtual string GetDictionaryName() => "dsdict.yaml";
        // 本音素器服务的语言码（多语言前缀声库的 <lang>/ 前缀；空 = 未声明）。
        // 注意声明为 protected：现行 OpenUtau 已改 public，但野外 DLL 多按旧版（protected virtual）编译，
        //   基类比覆写方法更宽会触发 CLR "cannot reduce access" 拒载；反向（旧基类 protected、新覆写 public）
        //   属放宽访问，CLR 允许——故取 protected 兼容新旧两代（实测 LYSE 六件套全为旧版）。
        protected virtual string GetLangCode() => string.Empty;

        // —— 门面扩展（非 OpenUtau API）：插件读取 protected 虚方法的桥 ——
        public string DictionaryName => GetDictionaryName();
        public string LangCode => GetLangCode();
    }

    public abstract class DiffSingerG2pPhonemizer : DiffSingerBasePhonemizer {
        protected virtual IG2p LoadBaseG2p() => null;
        // 算法引擎的规范元音/辅音清单（remap 到声库符号用）。
        protected virtual string[] GetBaseG2pVowels() => new string[0];
        protected virtual string[] GetBaseG2pConsonants() => new string[0];

        // —— 门面扩展（非 OpenUtau API）：插件读取 protected 虚方法的桥 ——
        // 每次调用可能新建引擎实例（子类内部通常有静态包缓存）；调用方自行缓存返回值。
        public IG2p CreateBaseG2p() => LoadBaseG2p();
        public string[] BaseG2pVowels => GetBaseG2pVowels();
        public string[] BaseG2pConsonants => GetBaseG2pConsonants();
    }
}

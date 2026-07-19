using System;

namespace OpenUtau.Api {
    // 忠实移植自 OpenUtau（MIT）——OpenUtau.Core/Api/Phonemizer.cs 的 PhonemizerAttribute。
    // 声库自带音素器 DLL 在类上标注本特性（名称 / tag / 作者 / 语言）；Language 是 GetLangCode() 为空时的语言回退依据。
    public class PhonemizerAttribute : Attribute {
        public string Name { get; private set; }
        public string Tag { get; private set; }
        public string Author { get; private set; }
        public string Language { get; private set; }

        public PhonemizerAttribute(string name, string tag, string author = null, string language = null) {
            Name = name;
            Tag = tag;
            Author = author;
            Language = language;
        }
    }
}

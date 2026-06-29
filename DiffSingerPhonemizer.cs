using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 一个音素的解析结果（绝对秒、归属 note、是否韵核、是否前置辅音）。时间可越界 note（前置辅音落在 note 起点之前）。
//   IsLead：音节核之前的引导辅音（onset），由 ProcessWord 的前置组判定；产物按时长模型回报时随之带出（§5.7）。
public readonly record struct PhonemeSpan(string Symbol, double StartTime, double EndTime, int NoteIndex, bool IsVowel, bool IsLead = false);

// DiffSinger phonemizer：歌词 → 音素时间线。G2P / 分组 / dur 预测忠实移植 OpenUtau DiffSingerBasePhonemizer：
//   · 短语首加前导 SP 组 + 500ms padding（给首辅音留空间），尾加哨兵；
//   · 音素按元音对齐到 note 起点（consonant-glide-vowel 特例：滑音起拍）；onset 辅音归前一组、定 IsLead；
//   · word_div=各组音素数、word_dur=相邻组（note 起点）间帧数 → linguistic(词模式) + dur → 每音素标称帧。
// 最终定位不再自己做对齐 / 钉死布局：把每发声 note 的标称音素（钉死值或 dur 预测）+ 几何（核起点 / FillEnd）
// 交宿主 SDK 的 PhonemeLayout.Resolve 统一派生 + 跨 note 去重叠——与宿主显示同源（WYSIWYG），钉死长辅音
// 越界自动压缩。在 TuneLab 绝对秒域工作（note 边界即秒）。延音符由宿主 VoiceSynthesisNoteSnapshot.IsContinuation 标志判定。
public static class DiffSingerPhonemizer
{
    const string Pause = "SP";
    const double PaddingSec = 0.5;   // 短语首辅音前导空间（OpenUtau padding=500ms）

    sealed class Group
    {
        public double Pos;            // 组起点（秒）；元音组=note 起点，首组=首 note 起点-padding，哨兵=末 note 终点
        public int Tone;
        public readonly List<string> Phonemes = new();
        public Group(double pos, int tone) { Pos = pos; Tone = tone; }
    }

    public static List<PhonemeSpan> Phonemize(
        DiffSingerPredictor dur, IReadOnlyList<VoiceSynthesisNoteSnapshot> notes,
        IReadOnlyList<string> noteLang, string speaker, int hop, int sampleRate, bool tensorCache)
    {
        if (notes.Count == 0)
            return new List<PhonemeSpan>();
        double frameSec = (double)hop / sampleRate;

        // —— 构建短语组：首组(前导 SP) + 各 note 的元音对齐分组 + 尾哨兵 ——
        var groups = new List<Group> { new(-1, notes[0].Pitch) { } };
        groups[0].Phonemes.Add(Pause);
        var notePhIndex = new List<int> { 1 };          // 各 note 在展平序列中的起始下标（含前导 SP 占 0）
        var noteSymbolCount = new int[notes.Count];     // 每 note 的音素数（onset+韵核）
        var leadCounts = new int[notes.Count];          // 每 note 的前置辅音数（ProcessWord 的前置组大小）：定 IsLead
        var pinned = new bool[notes.Count];

        for (int i = 0; i < notes.Count; i++)
        {
            var note = notes[i];

            // 延音符（宿主 IsContinuation 标志）：不产音素、不建组——前一发声 note 的元音经 FillEnd 铺过它。
            //   notePhIndex 记空区间。首 note 即延续则无前可沿，退化为常规 G2P。
            if (i > 0 && note.IsContinuation)
            {
                pinned[i] = false;
                noteSymbolCount[i] = 0;
                notePhIndex.Add(notePhIndex[^1]);
                continue;
            }

            string[] symbols = GetSymbols(dur, note, noteLang[i], out pinned[i]);
            noteSymbolCount[i] = symbols.Length;

            var wordGroups = ProcessWord(dur, note, symbols);
            leadCounts[i] = wordGroups[0].Phonemes.Count;           // 前置辅音（onset）数：归本 note，标 IsLead
            groups[^1].Phonemes.AddRange(wordGroups[0].Phonemes);   // 前置辅音并入前一组（侵入前一 note 尾）
            groups.AddRange(wordGroups.Skip(1));                    // 韵核组（起点=note 起点）
            notePhIndex.Add(notePhIndex[^1] + symbols.Length);
        }

        var last = notes[^1];
        groups.Add(new Group(last.EndTime, last.Pitch));            // 哨兵：标短语终点，无音素
        groups[0].Pos = notes[0].StartTime - PaddingSec;            // 前导 SP 起点

        // —— linguistic(词模式) + dur ——
        var flatSymbols = groups.SelectMany(g => g.Phonemes).ToArray();
        var flatTone = groups.SelectMany(g => Enumerable.Repeat(g.Tone, g.Phonemes.Count)).ToArray();
        int nTokens = flatSymbols.Length;

        var tokens = flatSymbols.Select(s => (long)dur.PhonemeToken(s)).ToArray();
        var langs = flatSymbols.Select(s => dur.LangId(PhonemeLanguage(s))).ToArray();
        var wordDiv = groups.Take(groups.Count - 1).Select(g => (long)g.Phonemes.Count).ToArray();
        var wordDur = groups.Zip(groups.Skip(1), (a, b) => (long)FramesBetween(a.Pos, b.Pos, frameSec)).ToArray();

        var durationFrames = RunDur(dur, tokens, langs, wordDiv, wordDur, flatTone, speaker, tensorCache);

        // —— 定位：每发声 note 物化为 PhonemeLayoutNote，交 SDK 的 PhonemeLayout.Resolve 统一派生 ——
        //   核起点 = 音符头；标称音素 = 钉死值（note.Phonemes）或 dur 模型预测（自由 note）；权重核 1 / 辅音 0；
        //   FillEnd = 下一发声 note 起点（跨延音符 / 空隙）或末 note 满末——令元音铺到下一发声 note、音素帧时间线连续。
        //   Resolve 内部「核起点=音符头、前置往左累积、核填到 FillEnd」+ 跨 note 两阶去重叠（元音先让、辅音簇等比压），
        //   故钉死长辅音越界自动压缩、自由 / 钉死混排边界协同一致，且与宿主显示同源（WYSIWYG）。
        var layoutNotes = new List<PhonemeLayoutNote>();
        var layoutNoteIndex = new List<int>();          // layoutNotes[ln] → snapshot note 下标
        for (int i = 0; i < notes.Count; i++)
        {
            int count = noteSymbolCount[i];
            if (count <= 0) continue;                   // 延音符 / 空 note：无音素，元音由前一发声 note 铺过
            int baseFlat = notePhIndex[i];
            var ph = notes[i].Phonemes;
            bool isPinned = pinned[i];
            var items = new SynthesizedPhoneme[count];
            for (int k = 0; k < count; k++)
            {
                string sym = flatSymbols[baseFlat + k];
                bool usePin = isPinned && k < ph.Count;
                items[k] = new SynthesizedPhoneme
                {
                    Symbol = sym,
                    Duration = usePin ? ph[k].Duration : durationFrames[baseFlat + k] * frameSec,
                    StretchWeight = usePin ? ph[k].StretchWeight : (dur.IsVowel(sym) ? 1 : 0),
                    IsLead = usePin ? ph[k].IsLead : k < leadCounts[i],
                };
            }
            layoutNotes.Add(new PhonemeLayoutNote
            {
                FillStart = notes[i].StartTime,
                FillEnd = NextSoundingStart(notes, noteSymbolCount, i),
                Phonemes = items,
            });
            layoutNoteIndex.Add(i);
        }

        var resolved = PhonemeLayout.Resolve(layoutNotes);

        var result = new List<PhonemeSpan>(nTokens);
        for (int ln = 0; ln < layoutNotes.Count; ln++)
        {
            int noteIndex = layoutNoteIndex[ln];
            var items = layoutNotes[ln].Phonemes;
            var times = resolved[ln];
            for (int k = 0; k < items.Count; k++)
            {
                var it = items[k];
                result.Add(new PhonemeSpan(it.Symbol, times[k].Start, times[k].End, noteIndex, dur.IsVowel(it.Symbol), it.IsLead));
            }
        }
        return result;
    }

    // 下一发声 note（noteSymbolCount>0）的起点；跨延音符 / 空隙；无则末 note 满末。作核(元音)填充终点 FillEnd。
    static double NextSoundingStart(IReadOnlyList<VoiceSynthesisNoteSnapshot> notes, int[] noteSymbolCount, int i)
    {
        for (int j = i + 1; j < notes.Count; j++)
            if (noteSymbolCount[j] > 0)
                return notes[j].StartTime;
        return notes[^1].EndTime;
    }

    // 单 note 的音素→组分配：元音起拍（consonant-glide-vowel：滑音起拍）；前置辅音落在 wordGroups[0]（归前组）。
    static List<Group> ProcessWord(DiffSingerPredictor dur, VoiceSynthesisNoteSnapshot note, string[] symbols)
    {
        var wordGroups = new List<Group> { new(-1, note.Pitch) };
        var isVowel = symbols.Select(dur.IsVowel).ToArray();
        var isGlide = symbols.Select(dur.IsGlide).ToArray();
        var isStart = new bool[symbols.Length];
        if (isVowel.All(b => !b) && symbols.Length > 0)
            isStart[0] = true;
        for (int i = 0; i < symbols.Length; i++)
        {
            if (!isVowel[i]) continue;
            if (i >= 2 && isGlide[i - 1] && !isVowel[i - 2])
                isStart[i - 1] = true;
            else
                isStart[i] = true;
        }
        bool started = false;
        for (int i = 0; i < symbols.Length; i++)
        {
            if (isStart[i] && !started) { wordGroups.Add(new Group(note.StartTime, note.Pitch)); started = true; }
            wordGroups[^1].Phonemes.Add(symbols[i]);
        }
        return wordGroups;
    }

    // 取音素符号串：钉死=用 note.Phonemes 符号；否则 G2P。过滤到「类型已定义 且 dur 表可 tokenize」；空则 [SP]。
    static string[] GetSymbols(DiffSingerPredictor dur, VoiceSynthesisNoteSnapshot note, string lang, out bool pinned)
    {
        pinned = note.Phonemes.Count > 0;
        IEnumerable<string> raw = pinned
            ? note.Phonemes.Select(p => ResolvePinnedSymbol(dur, p, lang))
            : dur.G2P(note.Lyric ?? string.Empty, lang);
        var symbols = raw.Where(s => dur.IsKnownSymbol(s) && dur.TryPhoneme(s, out _)).ToArray();
        return symbols.Length > 0 ? symbols : new[] { Pause };
    }

    // 钉死音素符号还原：音素符号保持干净（无 lang/ 前缀），语种来自 per-phoneme「language」属性（空 = 跟随 note）。
    //   忠实 OpenUtau ValidatePhoneme 次序：先试裸符号（命中语言无关符号如 SP/AP，或兼容历史带前缀数据），
    //   再试 <lang>/<符号> 还原嵌入表键；都不中交由上层 IsKnownSymbol 过滤 / [SP] 兜底。
    static string ResolvePinnedSymbol(DiffSingerPredictor dur, VoiceSynthesisPhonemeSnapshot ph, string noteLang)
    {
        var sym = ph.Symbol;
        if (dur.TryPhoneme(sym, out _)) return sym;
        var lang = ph.Properties.GetString(DiffSingerDeclarations.KeyLanguage, string.Empty);
        if (string.IsNullOrEmpty(lang)) lang = noteLang;
        var prefixed = $"{lang}/{sym}";
        return dur.TryPhoneme(prefixed, out _) ? prefixed : sym;
    }

    static int FramesBetween(double t1, double t2, double frameSec)
        => (int)(t2 / frameSec) - (int)(t1 / frameSec);

    static string PhonemeLanguage(string phoneme)
    {
        int slash = phoneme.IndexOf('/');
        return slash > 0 ? phoneme[..slash] : string.Empty;
    }

    // linguistic(词模式) + dur → 每音素预测帧（float）。
    static double[] RunDur(DiffSingerPredictor dur, long[] tokens, long[] langs,
        long[] wordDiv, long[] wordDur, int[] phMidi, string speaker, bool tensorCache)
    {
        int nTokens = tokens.Length, nWords = wordDiv.Length, hidden = dur.HiddenSize;
        var lingInputs = new List<NamedOnnxValue>
        {
            Nv("tokens", tokens, nTokens),
            Nv("word_div", wordDiv, nWords),
            Nv("word_dur", wordDur, nWords),
        };
        if (dur.Linguistic.InputMetadata.ContainsKey("languages"))
            lingInputs.Add(Nv("languages", langs, nTokens));

        var lingOut = DiffSingerTensorCache.Run(dur.Linguistic, dur.LinguisticHash, lingInputs, tensorCache);
        var enc = lingOut.First(v => v.Name == "encoder_out").AsTensor<float>();
        var mask = lingOut.First(v => v.Name == "x_masks").AsTensor<bool>();
        var encDense = new DenseTensor<float>(enc.ToArray(), enc.Dimensions.ToArray());
        var maskDense = new DenseTensor<bool>(mask.ToArray(), mask.Dimensions.ToArray());

        var durModel = dur.Model("dur");
        var durInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("encoder_out", encDense),
            NamedOnnxValue.CreateFromTensor("x_masks", maskDense),
            Nv("ph_midi", phMidi.Select(x => (long)x).ToArray(), nTokens),
        };
        // spk_embed 仅当 dur 模型声明该口时喂入（单说话人模型无此口）——
        //   与 DiffSingerPitch / DiffSingerVariance 一致；否则向 ORT 喂未声明输入会抛 Invalid Feed Input Name。
        if (durModel.InputMetadata.ContainsKey("spk_embed"))
        {
            float[] emb = dur.GetEmbedding(speaker);
            var spk = new float[nTokens * hidden];
            for (int i = 0; i < nTokens; i++) Array.Copy(emb, 0, spk, i * hidden, hidden);
            durInputs.Add(NamedOnnxValue.CreateFromTensor("spk_embed",
                new DenseTensor<float>(spk, new[] { 1, nTokens, hidden })));
        }
        var durOut = DiffSingerTensorCache.Run(durModel, dur.ModelHash("dur"), durInputs, tensorCache);
        return durOut.First(v => v.Name == "ph_dur_pred").AsTensor<float>().Select(x => (double)x).ToArray();
    }

    static NamedOnnxValue Nv(string name, long[] data, int n)
        => NamedOnnxValue.CreateFromTensor(name, new DenseTensor<long>(data, new[] { 1, n }));
}

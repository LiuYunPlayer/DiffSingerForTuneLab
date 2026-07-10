using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TuneLab.SDK;

namespace DiffSingerForTuneLab;

// 一个音素的解析结果（绝对秒、归属 note、是否韵核）。时间可越界 note（前置辅音落在 note 起点之前）。
//   前后归属（拍前 / 拍后）不落每音素标志——由 note 级前置量 Preutterance（拍前发声量）派生（见 PhonemeLayout / SynthesizedSyllable）。
public readonly record struct PhonemeSpan(string Symbol, double StartTime, double EndTime, int NoteIndex, bool IsVowel);

// DiffSinger phonemizer：歌词 → 音素时间线。G2P / 分组 / dur 预测忠实移植 OpenUtau DiffSingerBasePhonemizer：
//   · 短语首加前导 SP 组 + 500ms padding（给首辅音留空间），尾加哨兵；
//   · 音素按元音对齐到 note 起点（consonant-glide-vowel 特例：滑音起拍）；onset 辅音归前一组、计入 note 头前的前置量 Preutterance；
//   · word_div=各组音素数、word_dur=相邻组（note 起点）间帧数 → linguistic(词模式) + dur → 每音素标称帧。
// 最终定位不再自己做对齐 / 钉死布局：把每发声 note 的标称音素（钉死值或 dur 预测）+ 几何（核起点 / FillEnd）
// 交宿主 SDK 的 PhonemeLayout.Resolve 统一派生 + 跨 note 去重叠——与宿主显示同源（WYSIWYG），钉死长辅音
// 越界自动压缩。在 TuneLab 绝对秒域工作（note 边界即秒）。延音符由本插件自判（判定权归引擎的 SDK 契约）：
// live 域权威在 DiffSingerSynthesisSession.IsContinuation，快照域由 ComputeContinuation 以同语义自判（见彼处等价论证）。
public static class DiffSingerPhonemizer
{
    const string Pause = "SP";
    const double PaddingSec = 0.5;   // 短语首辅音前导空间（OpenUtau padding=500ms）
    // 插件约定：note 歌词写 "+" 表示「当前多音节词的下一个音节从本 note 起」（OpenUtau 同款记号）。
    //   "+" 是发声 note（自带音节、恒非延续）；纯延音 "-" 走延音判定（ComputeContinuation），两者互斥。
    const string SyllableAdvance = "+";

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

        // —— 构建短语组：首组(前导 SP) + 各「音节 note」(词首 + 每个 "+") 的元音对齐分组 + 尾哨兵 ——
        //   词模型：常规歌词 note 起一个新词，G2P 出全部音节；其后的 "+" 各取下一音节、"-" 为纯延音(melisma)。
        //   每个音节 note 与今天单 note 走完全相同的处理（onset 前移、韵核成组、FillEnd 铺元音）；
        //   音节与 "+" 数量不匹配的边界处理见下方分配逻辑。
        var groups = new List<Group> { new(-1, notes[0].Pitch) };
        groups[0].Phonemes.Add(Pause);
        var notePhIndex = new List<int> { 1 };          // 各 note 在展平序列中的起始下标（含前导 SP 占 0）
        var noteSymbolCount = new int[notes.Count];     // 每 note 的音素数（onset+韵核）；延音/空 note 为 0
        var leadCounts = new int[notes.Count];          // 每 note 的前置辅音数（ProcessWord 的前置组大小）：定自由 note 的前置量 Preutterance
        var pinned = new bool[notes.Count];

        // 把一个 note 的音素串并入 groups：onset 前移并入前组、韵核成组于本 note 起点；登记 count/lead/phIndex。
        void EmitNote(int idx, string[] symbols, bool isPinned)
        {
            pinned[idx] = isPinned;
            noteSymbolCount[idx] = symbols.Length;
            var wordGroups = ProcessWord(dur, notes[idx], symbols);
            leadCounts[idx] = wordGroups[0].Phonemes.Count;
            groups[^1].Phonemes.AddRange(wordGroups[0].Phonemes);
            groups.AddRange(wordGroups.Skip(1));
            notePhIndex.Add(notePhIndex[^1] + symbols.Length);
        }
        // 延音 note（count=0）：不产音素、不建组，前一发声 note 的元音经 FillEnd 铺过它。
        void EmitMelisma(int idx)
        {
            pinned[idx] = false;
            noteSymbolCount[idx] = 0;
            notePhIndex.Add(notePhIndex[^1]);
        }

        // 快照域延音判定（与 session live 判定同语义；孤儿 "-"——块首/断链——为 false，落词头路径发 SP）。
        var isCont = ComputeContinuation(notes);

        int wi = 0;
        while (wi < notes.Count)
        {
            if (isCont[wi]) { EmitMelisma(wi); wi++; continue; }

            // 词起点：取词首 note 的音素串（钉死=note.Phonemes 符号；否则 G2P 整词）。
            string[] leadSymbols = GetSymbols(dur, notes[wi], noteLang[wi], out bool isPinned);
            bool leadIsPlus = IsSyllableAdvance(notes[wi]);   // 句首孤儿 "+"（无前词）

            // 收集本词所有 note（词首 + 其后的 "+"/"-"，到下一个常规歌词为止）。音节槽 = 词首 + 各 "+"。
            int end = wi + 1;
            var slots = new List<int> { wi };
            while (end < notes.Count && (isCont[end] || IsSyllableAdvance(notes[end])))
            {
                if (IsSyllableAdvance(notes[end])) slots.Add(end);
                end++;
            }

            // 拆音节：仅常规 G2P 词（非钉死、非孤儿 "+"）才拆；否则整串作单音节。
            var segments = (isPinned || leadIsPlus)
                ? new List<string[]> { leadSymbols }
                : SplitSyllables(dur, leadSymbols);

            // 分配 segment → 音节槽：前 N-1 槽各取一段（不足留空 → 多余 "+"）；末槽收下余下全部（音节多于槽时堆末槽）。
            int n = slots.Count, sCount = segments.Count;
            var slotSymbols = new string[n][];
            for (int k = 0; k < n; k++)
                slotSymbols[k] = k < n - 1
                    ? (k < sCount ? segments[k] : Array.Empty<string>())
                    : (k < sCount ? segments.Skip(k).SelectMany(x => x).ToArray() : Array.Empty<string>());

            // 按 note-index 顺序发射词内每个 note：音节槽 → 该段；空段=多余 "+"（无音节可领）→ 当不认识的歌词发 SP；
            //   "-" → 延音(melisma)。多余 "+" 不走 melisma：判定权虽已归本插件，但 "+" 判延续须以「音节耗尽」
            //   为条件——那要在判定期跑 G2P 数音节（live 域同步可行但与本层词分配逻辑必须逐 case 一致），
            //   留待与判定联动的独立变更（化石翻案），当前保持 SP 与判定（"+" 恒非延续）严格一对一。
            int slotCursor = 0;
            for (int k = wi; k < end; k++)
            {
                if (slotCursor < n && k == slots[slotCursor])
                {
                    var syms = slotSymbols[slotCursor];
                    EmitNote(k, syms.Length > 0 ? syms : new[] { Pause }, isPinned && slotCursor == 0);
                    slotCursor++;
                }
                else EmitMelisma(k);
            }
            wi = end;
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
                };
            }
            // 前置量（拍前发声量，秒）：钉死 note 用宿主随快照下发的 note 头前音素占位量（用户可控）；
            //   自由 note 由音韵派生——前置辅音前缀（leadCounts）的标称时长之和（onset 落在 note 头之前）。
            double preutter;
            if (isPinned)
                preutter = notes[i].Preutterance;
            else
            {
                preutter = 0;
                for (int k = 0; k < leadCounts[i] && k < count; k++)
                    preutter += Math.Max(0, items[k].Duration);
            }
            layoutNotes.Add(new PhonemeLayoutNote
            {
                FillStart = notes[i].StartTime,
                FillEnd = NextSoundingStart(notes, noteSymbolCount, i),
                Preutterance = preutter,
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
                result.Add(new PhonemeSpan(it.Symbol, times[k].Start, times[k].End, noteIndex, dur.IsVowel(it.Symbol)));
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

    // note 是否拆音节推进记号 "+"（插件约定，见 SyllableAdvance）。与延音("-")记号天然互斥，无需守卫。
    static bool IsSyllableAdvance(VoiceSynthesisNoteSnapshot note)
        => note.Lyric?.Trim() == SyllableAdvance;

    // 快照域延音判定：与 DiffSingerSynthesisSession.IsContinuation 同语义（"-" ∧ 严格相接链回溯到内容 note；
    //   只看歌词与位置、不看钉死——钉死随 "-" 休眠，理由见彼处注释），改一处必改两处。
    //   等价论证：本会话按 note 间隙严格分块 ⇒ 链不跨块，块内自判与 live 判定逐 note 相同
    //   （块首 "-" 必因块边界空隙断链 → 孤儿，live 域同样为孤儿）。
    //   前向归纳：isCont[i] = 相接(i-1,i) ∧ 记号(i) ∧ (i-1 是内容 note ∨ isCont[i-1])。
    static bool[] ComputeContinuation(IReadOnlyList<VoiceSynthesisNoteSnapshot> notes)
    {
        var isCont = new bool[notes.Count];
        for (int i = 1; i < notes.Count; i++)
        {
            if (!DiffSingerSynthesisSession.IsContinuationMarker(notes[i].Lyric))
                continue;
            if (notes[i - 1].EndTime < notes[i].StartTime)
                continue;                                   // 空隙断链（严格比较，同 live 判定）→ 孤儿
            isCont[i] = isCont[i - 1] || !DiffSingerSynthesisSession.IsContinuationMarker(notes[i - 1].Lyric);
        }
        return isCont;
    }

    // 把一个词的音素串按元音起点切成音节段（与 ProcessWord 同款 isStart：含 consonant-glide-vowel 滑音起拍特例）。
    //   段 0 含词首前置辅音（[0, 第二个起点)）；其后每段 = [起点k, 起点k+1)。无元音 → 整串作一段。
    //   每段交回 EmitNote→ProcessWord 时会再各自做 onset/韵核切分，故段内前置辅音仍正确前移。
    static List<string[]> SplitSyllables(DiffSingerPredictor dur, string[] symbols)
    {
        if (symbols.Length == 0) return new List<string[]> { symbols };
        var isVowel = symbols.Select(dur.IsVowel).ToArray();
        var isGlide = symbols.Select(dur.IsGlide).ToArray();
        var isStart = new bool[symbols.Length];
        if (isVowel.All(b => !b)) isStart[0] = true;
        for (int i = 0; i < symbols.Length; i++)
        {
            if (!isVowel[i]) continue;
            if (i >= 2 && isGlide[i - 1] && !isVowel[i - 2]) isStart[i - 1] = true;
            else isStart[i] = true;
        }
        var starts = new List<int>();
        for (int i = 0; i < symbols.Length; i++) if (isStart[i]) starts.Add(i);
        if (starts.Count == 0) return new List<string[]> { symbols };

        var segs = new List<string[]>();
        for (int k = 0; k < starts.Count; k++)
        {
            int from = k == 0 ? 0 : starts[k];                              // 段 0 含词首 onset
            int to = k + 1 < starts.Count ? starts[k + 1] : symbols.Length;
            segs.Add(symbols[from..to]);
        }
        return segs;
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

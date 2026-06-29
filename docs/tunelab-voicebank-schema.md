# TuneLab 声库描述文件（`tunelab.yaml`）——初版设计方案

> 状态：草案 v0.1（待 DiffSinger 团队评审）
> 适用：DiffSingerForTuneLab 插件
> schema 版本：`tunelab-voicebank/1`

## 1. 背景与目标

本插件直接读取 **OpenUtau 形态**的 DiffSinger 声库（`dsconfig.yaml` + `character.yaml/.txt` + 预测器子目录 + `.emb` 等）。这套格式是社区既成事实，**不能推翻重做**——否则全社区声库都要重新打包。

因此本方案的定位是：在声库目录里**新增一个可选的、TuneLab 专属的叠加描述文件**，只用来解决 OpenUtau 格式**表达不了**的几件事，其余一律沿用现有推断逻辑。

要解决的核心问题：

1. **声库 ID 稳定性**——当前 voiceId 取声库文件夹名（见 [VoicebankScanner.cs](../VoicebankScanner.cs#L75-L77)），ID 与显示文本耦合、改名即失效。
2. **retake 能力声明**——本插件实现了 OpenUtau 不具备的 note 级 retake（pitch / variance / acoustic 三族各自独立），需要声库声明自己导出了哪几族的 retake 能力。
3. **speaker = voice**——DiffSinger 一个声库含多个 speaker，而 speaker 才对应别家引擎的"一个歌手/voice"。需要把暴露出的 speaker 各自呈现为 TuneLab 的一个 voice。
4. **声库版本演进**——同一声库会以独立包不断升级；同 ID 不同版本应自动合并，speaker 取并集，并给每个 voice 一个版本下拉。
5. **speaker 暴露白名单**——一个声学模型可能内含多个 speaker，但作者往往只想暴露其中一部分；未列出的应不可见、不可混入。

## 2. 设计原则

- **两个文件并存、职责不重叠、都读**：`tunelab.yaml` 不复制 dsconfig 的任何模型事实，只承载 TuneLab 的"作者决策层"。两边没有同名字段 → 无覆盖/合并规则 → 不会 desync。这与今天 `character.yaml` 和 `dsconfig.yaml` 各管一摊地并存是同一形态。
- **作者只手写一个小文件、不重打包**：作者只写下表左列那几项；DSP/模型事实继续由插件从 dsconfig 读，作者一个数都不用抄。
- **文件缺失 → 纯 OpenUtau 路径**：插件行为与今天完全一致（老声库一个不破），本文件新增的全部机制（retake / 多 voice / 版本 / i18n）退回默认关闭。
- **不参与声库发现**：发现逻辑继续认 `dsconfig.yaml` + `character.*`（见 [VoicebankScanner.cs](../VoicebankScanner.cs#L64-L71)）。`tunelab.yaml` 只在已识别为声库后被附加解析。
- **能力是构件的客观事实，不靠手写声明替代**——除一个例外：retake（理由见 §7）。
- **插件无权写用户工程数据**——任何"默认值"只能声明、不能回写 part 属性（这直接决定了版本"钉死/浮动"的形态，见 §8）。

### 2.1 dsconfig × tunelab.yaml 职责分工

经查证（对照 DiffSinger 配置 schema 文档、OpenUtau `DiffSingerConfig.cs`、导出脚本 `acoustic_exporter.py`/`variance_exporter.py` 与真实样例）：**dsconfig 的 43 个字段几乎全部由 DiffSinger 导出脚本生成、是模型自身契约，唯一例外是 `vocoder`**（导出不写、需手填、按名指向声码器目录，是宿主约定）。所以 dsconfig 本质是"对声库/模型的描述"，可放心直接读。

| 归 `tunelab.yaml`（作者手写、TuneLab 决策层） | 留在 `dsconfig` + 预测器子目录（导出生成、单一真相源，插件直接读） |
|---|---|
| `id` / `version` / `name` + i18n | `sample_rate` / `hop_size` / `win`·`fft_size` / `num_mel_bins` / `mel_*` / `hidden_size` |
| `voices`（id/speaker/name/i18n/portrait/color/默认语言 + 白名单） | `acoustic` / `phonemes` / `languages` 文件名（ONNX/表引用） |
| `retake.{acoustic,pitch,variance}` 声明 | `use_*_embed`（模型是否吃该输入——导出定死，决定哪些曲线可暴露） |
| `languages` 的显示名 + i18n + 白名单 | `speakers` / `languages` 原始表（tunelab 的 voices/languages 按 id **引用**，非复制） |
| | 预测器子目录 `predict_*` / `linguistic` / `dur`·`pitch`·`variance` 等 |

> **`vocoder` 字段**：它是 dsconfig 里唯一的宿主约定残留（手填、指向 `Vocoders/{name}/` 安装目录）。第一版**继续从 dsconfig 读**（方案 A，零改动；本插件已复用同款 Vocoders 目录约定）。将来若要与 OpenUtau 解耦，可把它提为 `tunelab.yaml` 的可选字段去覆盖 dsconfig（方案 B，见 §14）。`voices[].speaker`、`languages.expose[].id` 是对 dsconfig 表的**引用**而非复制，故也不构成重叠、不会 desync。

## 3. 选型结论

### 3.1 YAML 而非 JSON

| 维度 | YAML | JSON |
|---|---|---|
| 与声库内其它文件一致 | ✅ 同目录全是 `*.yaml` | ❌ 风格割裂 |
| 注释（可内联记录暴露约束的原因） | ✅ | ❌ |
| 受众（声库作者手改手打包） | ✅ 已习惯 | 一般 |
| 插件依赖 | ✅ 已依赖 YamlDotNet，零新增 | 需另走序列化 |

**结论：YAML。** 唯一偏向 JSON 的"TuneLab 自身用 JSON"指的是**插件清单** `description.json`，那是插件级文件、不在声库目录里；本文件住在声库目录、与一堆 `.yaml` 为邻，一致性 + 可注释 + 零新增依赖让 YAML 胜出。

### 3.2 文件名：`tunelab.yaml`

一眼可知归属、发现成本最低、不与现有文件冲突。放在**声库根目录**，与 `dsconfig.yaml` / `character.yaml` 同级。

## 4. 标识模型：三套标识严格分离

| 标识 | 取值 | 用途 | 稳定性要求 |
|---|---|---|---|
| **VoiceId** | `{声库id}.{voice id}` | TuneLab part 序列化锚点 | 跨版本、跨改名恒定 |
| **显示名** | `{声库name}·{voice name}` | UI 展示（voice 列表、面板） | 可随版本变化 |
| **dsconfig speaker** | dsconfig `speakers` 后缀 | 选 `.emb`、对接底层模型 | 可随上游变化 |

要点：

- **版本不进入身份**——它是合成时的解析维度（part 属性），见 §8。升级声库不会让任何老 part 的 VoiceId 失效。
- **speaker `id` 是跨版本契约**——版本合并完全依赖它稳定。要改名只改 `name`（显示）或 `speaker`（底层后缀），**绝不可改 `id`**。
- 不同声库可能含同名 speaker，故显示名必须带声库前缀 `{声库name}·{voice name}` 来消歧。

## 5. Schema

```yaml
# tunelab.yaml —— DiffSinger 声库的 TuneLab 叠加描述（可选）
# 缺失：插件从 dsconfig.yaml + character.yaml 推断一切（= 今天行为）
# 存在：此处字段覆盖推断值；未写的字段回退。本文件不参与声库发现。

format: tunelab-voicebank/1     # 必填：schema 标识 + 主版本

# ---- 声库身份 ----
id: example.choir               # 必填：稳定声库 id，与文件夹名/显示名解耦。建议反域名风格避免撞车
name: 示例声库                   # 必填：声库显示名（基准/兜底），用于 "声库·voice" 显示前缀
name_i18n:                      # 可选：声库名的本地化。key = 宿主 culture（en-US/zh-CN，区域大写）
  en-US: Example Voicebank
version: 3                      # 必填：单调递增整数。决定排序与"最新"；同 id 不同 version 触发合并
version_label: "2024 重训版"      # 可选：版本的人类可读标签，仅供下拉显示，不参与比较/排序
version_label_i18n: { en-US: "2024 Retrain" }   # 可选：标签的本地化

# ---- retake 能力（纯声明，三位独立，全默认 false）----
retake:
  acoustic: true                # 已导出 acoustic note 级 retake（软条件 mask）
  pitch: true                   # 已导出 pitch retake mask
  variance: false               # 未导出 → 不暴露 variance seed 轨

# ---- 语言（标识符来自 dsconfig，此处只叠加显示名 + i18n + 白名单）----
languages:                      # 可选。整块缺失 = 用 dsconfig 全部语言、显示=裸 id（今天行为）
  default: zh                   # 可选：声库级默认语言（覆盖 dsconfig languages 首项；voices[].default_language 再覆盖）
  expose:                       # 可选：白名单 + 显示名 + i18n。出现 = 只暴露列出的语言并按此顺序
    - id: zh                    # 必填：必须匹配 dsconfig 语言表的键（模型 lang_id 锚此，不可臆造）
      name: 中文                 # 下拉显示名（基准/兜底）；缺省回退裸 id
      name_i18n: { en-US: Chinese, ja-JP: Chinese }
    - { id: en, name: English, name_i18n: { zh-CN: 英语 } }

# ---- 暴露的 voice 清单（= speaker 白名单）----
voices:                         # 可选：出现 = 白名单；缺失 = 退化为"整库 1 voice + 全 speaker 下拉"
  - id: alpha                   # 必填：稳定 voice/speaker id → VoiceId = "example.choir.alpha"
    speaker: model.alpha        # 必填：对应 dsconfig speakers 条目（身份≠后缀，显式映射）
    name: 歌手甲                 # 必填：voice 显示名（基准）→ 显示为 "示例声库·歌手甲"
    name_i18n: { en-US: Singer A }  # 可选：voice 名本地化
    default_language: zh         # 可选：覆盖声库级默认语言
    portrait: spk/alpha.png      # 可选：该 voice 立绘（相对声库根）
    color: "#7AC1FF"             # 可选：UI 配色
  - { id: beta,  speaker: model.beta,  name: 歌手乙, name_i18n: { en-US: Singer B } }
  - { id: gamma, speaker: model.gamma, name: 歌手丙, name_i18n: { en-US: Singer C } }
  - { id: delta, speaker: model.delta, name: 歌手丁, name_i18n: { en-US: Singer D } }
  # dsconfig 里其余 speaker（不希望暴露的）：未列出 → 既不出现在 voice 列表，也不可作为混音目标

# ---- 待评审：是否纳入第一版 ----
# min_engine_version: "0.2.0"   # 低于此插件版本拒载并提示升级（与 retake 等新能力配对）
# license: ...                  # 授权/分发约束
# homepage: ...
```

### 字段表

| 字段 | 类型 | 必填 | 缺省回退 | 说明 |
|---|---|---|---|---|
| `format` | string | ✅ | — | `tunelab-voicebank/{major}`；major 变更 = 破坏性升级 |
| `id` | string | ✅ | 文件夹名 | 稳定声库 id |
| `name` | string | ✅ | character.yaml `name` / 文件夹名 | 声库显示名（基准/兜底） |
| `name_i18n` | map<locale,string> | ✗ | 回退 `name` | 声库名本地化，key = `en-US`/`zh-CN` |
| `version` | int | ✅ | `0` | 单调递增整数版本号，合并/排序依据（整数比较） |
| `version_label` | string | ✗ | 显示为 `v{version}` | 版本的人类可读标签，仅下拉显示、不参与比较 |
| `version_label_i18n` | map<locale,string> | ✗ | 回退 `version_label` | 版本标签本地化 |
| `retake.acoustic` | bool | ✗ | `false` | 是否暴露 acoustic seed 轨 |
| `retake.pitch` | bool | ✗ | `false` | 是否暴露 pitch seed 轨 |
| `retake.variance` | bool | ✗ | `false` | 是否暴露 variance seed 轨 |
| `languages` | map | ✗ | dsconfig 全部语言、显示=裸 id | 语言显示叠加层 |
| `languages.default` | string | ✗ | dsconfig languages 首项 | 声库级默认语言 |
| `languages.expose` | list | ✗ | dsconfig 全部语言 | 出现即白名单 + 排序 |
| `languages.expose[].id` | string | ✅* | — | 必须匹配 dsconfig 语言表的键（*若有 expose 块则必填） |
| `languages.expose[].name` | string | ✗ | 裸 id | 下拉显示名（基准） |
| `languages.expose[].name_i18n` | map<locale,string> | ✗ | 回退 `name` | 语言名本地化 |
| `voices` | list | ✗ | 整库 1 voice + 全 speaker 下拉 | 出现即白名单 |
| `voices[].id` | string | ✅* | — | voice 稳定 id（*若有 voices 块则必填） |
| `voices[].speaker` | string | ✅* | — | 对应 dsconfig speakers 条目 |
| `voices[].name` | string | ✅* | speaker 后缀 | voice 显示名（基准） |
| `voices[].name_i18n` | map<locale,string> | ✗ | 回退 `name` | voice 名本地化 |
| `voices[].default_language` | string | ✗ | `languages.default` | 覆盖声库级默认语言 |
| `voices[].portrait` | string | ✗ | character.yaml `image` | voice 立绘（相对路径） |
| `voices[].color` | string | ✗ | 自动轮转配色 | UI 配色 |

## 6. speaker = voice

TuneLab 的选择单位是 **voice**（`VoiceSourceInfos` 是 voiceId→{Name, Description, Portrait} 的扁平有序表，一个 part 选一个 voiceId）。DiffSinger 的 speaker 恰好对应别家引擎的"一个歌手"。

落地约束：

1. **后端模型按声库共享，N 个 voice 只加载一次。** 模型缓存继续按**物理声库路径/版本**做 key（见 [DiffSingerVoiceEngine.cs](../DiffSingerVoiceEngine.cs#L116-L126)），voice 只是其上的一层呈现。选歌手甲-voice 和歌手乙-voice 不会把模型 load 两遍。
2. **`voices` 块出现 = 白名单**：仅列出的 speaker 暴露为 voice，其余隐藏。
3. **白名单同时门控 speaker_mix 混入目标**（见 §9）——隐藏的 speaker 不能从混音后门泄露。
4. **`voices` 块缺失 = 退化为今天的行为**：整库 1 个 voice、speaker 作为 part 面板的下拉 + 混音容器（见 [DiffSingerDeclarations.cs](../DiffSingerDeclarations.cs#L140-L150)）。

## 7. retake 声明（唯一靠手写声明的能力）

retake 能力取决于声库的 ONNX 是否用 fork 版重导出（把初始噪声外置为 `noise` 输入）。acoustic / pitch / variance 是**三个独立模型族**，可以只导出其中一部分。

**为什么靠声明而非探测：**

- 自动探测需创建 `InferenceSession`（加载权重 + 初始化执行设备），这是合成阶段才付的重成本，无法在发现阶段对所有声库预先廉价判别。
- 探测出真相后若反向纠正并落盘，等于插件改用户/缓存数据——**越权**。

**因此：retake 三位纯由作者声明，默认 `false`。** 探测顶多作为合成时的**静默兜底**（声明 true 但模型实际无 `noise` 输入 → 该次合成按无 retake 处理，不写任何东西）。作者声明错了是作者责任。

声明位与 seed 自动化轨一一对应（见 [DiffSingerDeclarations.cs](../DiffSingerDeclarations.cs#L91-L97)）：

| 声明位 | 控制的 seed 轨 |
|---|---|
| `retake.pitch` | `seed_pitch`（Pitch seed） |
| `retake.variance` | `seed_variance`（Variance seed） |
| `retake.acoustic` | `seed_acoustic`（Timbre seed） |

> 当前实现里三条 seed 轨是**无条件**暴露的；接入本字段后，应改为按对应 `retake.*` 位 gating。

## 8. 版本合并与解析

### 8.1 合并

- **合并键 = `id`。** 扫描到多个物理包同 `id`、不同 `version` → 归为一个**逻辑声库**。
- **`version` 是单调递增整数**：排序、判等、"最新"都只需整数比较，无 semver 语义（本设计不用 major/minor 兼容规则），作者每次发布 +1 即可，也避开字符串比较的坑（"2.10" vs "2.9"）。
- **物理上各版本仍独立加载各自的 ONNX**（权重不同）。合并只发生在注册/呈现层：维护一张 `版本 → 物理目录` 映射；模型缓存仍按物理版本目录做 key，**不存在权重合并**。
- **voices = 跨版本 speaker `id` 并集**。同一 speaker 跨版本只出现一个 voice 条目。
- 同 `id` 同 `version` 撞包 → 判重，取其一 + 告警，不崩。
- 元数据（声库 `name`、voice `name`/`portrait` 等）跨版本不一致时 → **取最新版本的**为准。

### 8.2 版本下拉（part 属性）

每个 voice 在 part 面板暴露一个**版本下拉**：

```
[ 最新（跟随） ]   ← 默认选中。sentinel，永远解析到当前最新
  v3              ← 无 version_label 时显示 "v{整数}"；"v" 仅前缀，比较/排序按整数走
  v2
  ...
```

设了 `version_label` 则下拉显示该标签（如 `2024 重训版`，可按 locale 本地化），但底层仍是整数、比较不受影响。

- **下拉只列出"含此 speaker 的版本"**——某 speaker 可能 v2 才新增、v3 又移除，故各 voice 的可选版本是合并版本集的子集。
- **默认 = `最新（跟随）` sentinel**，解析为"含此 speaker 的最新版本"（非全局最新——全局最新可能没这人）。

### 8.3 钉死 vs 浮动（受平台能力约束）

插件**无权写 part 属性**，未被用户碰过的属性永远取声明时动态算出的默认值。因此：

- **默认浮动**：默认值 = 动态算出的最新。装了新版本包，没动过的 part 自动跟到新版——这也正是"打开旧工程效果自然变好"的轻度用户最优解。
- **钉死 = 显式 opt-in**：用户手选某个具体版本 → 这是用户**主动写**的数据（合法）→ 该 part 钉死在该版本，后续升级不再变。
- sentinel 的价值：用户**看得见**自己处于"跟随"状态，会有意识地去钉，不会误以为已锁定。

> **"调回去就能恢复"的前提**：旧版本包仍在。各版本是独立包，升级 = 并排多装；只要没删旧包，旧版本就还在下拉里、随时可调回，不丢任何东西。若是"删旧装新"式升级，那个版本从下拉消失、也调不回——这不是 bug，是包没了。

## 9. 混音门控

`speaker_mix` 是在**同一个模型的 embedding 空间**里混的。因此：

- **跨版本不能混**——不同版本是不同模型、不同 embedding 空间。
- **混入候选 = 当前选中版本里实际存在、且在白名单内的 speaker。** 双重门控：既不能混跨版本的人，也不能混白名单外（未暴露）的人。

这是对现有 `SelectedMixTracks` / `BuildSpeakerMixConfig`（见 [DiffSingerDeclarations.cs](../DiffSingerDeclarations.cs#L107-L175)）候选集的收窄。

## 10. i18n（本地化）

### 10.1 边界：只本地化"作者写的内容"

本文件的 i18n **只覆盖声库作者写的展示串**：声库 `name`、`voices[].name`、`languages.expose[].name`。

**插件自身的 UI 文案**（Speaker / Gender / Energy / 语言下拉的固定 label / 版本下拉 label 等）**不归本文件管**——那是插件 [Localization.cs](../Localization.cs) / `L.Tr` 的职责，声库作者既不该也不能本地化插件的壳。

### 10.2 形态：内联 per-field，而非集中 locale 块

```yaml
name: 示例声库
name_i18n: { en-US: Example Voicebank }
```

选内联（每个名字旁挂 `*_i18n`）而非宿主 `description.json` 那种集中 `localizations: { en-US: {...} }` 块的理由：

- 声库可本地化的串很少（声库名 + 几个 voice 名 + 几个语言名），集中块"一个翻译者编辑一整块"的优势用不上。
- 集中块要按 id 做 join，加一个 voice/语言就得记得去每个语言块补一条，漏了即 orphan；内联把译文贴在被译对象旁，**结构上不可能脱钩**，对手改手打包的社区作者更稳。

### 10.3 locale key 必须对齐宿主 culture

宿主语言来自 `TuneLabContext.Global.Language`，取值是 **`en-US` / `zh-CN`（BCP-47，区域大写）**（见 SDK [ITuneLabContext.cs](../../TuneLab/TuneLab.SDK/Environment/ITuneLabContext.cs#L7) 注释与本插件 [Localization.cs](../Localization.cs#L12) 的字典键）。因此 `*_i18n` 的 key **规范写法 = `en-US`/`zh-CN`**，与宿主传来的值逐字一致。

### 10.4 解析顺序

`name`（基准串）始终必填、是规范兜底。按宿主当前 locale：

```
精确匹配(zh-CN) → 退到语言码(zh，匹配任一 zh-*) → 退到基准 name
```

为容错作者手抖（写成 `en-us`、`en`），插件解析 `*_i18n` 时应**大小写不敏感** + 支持纯语言码兜底。注意这比插件现有 `L.Tr` 的大小写敏感直查更宽松——作者数据比插件自带词典更易写错，值得多一层容错。

## 11. 回退矩阵

| 情形 | 行为 |
|---|---|
| 无 `tunelab.yaml` | 完全等同今天：voiceId = 文件夹名，整库 1 voice，speaker 走 part 下拉 + 混音容器 |
| 有文件但无 `voices` 块 | 用文件的 `id`/`name`，但仍整库 1 voice（不拆 speaker） |
| 有 `voices` 块 | speaker 拆成多 voice、白名单生效、混音门控生效 |
| 无 `retake` 块 | 三位皆 false，不暴露任何 seed 轨 |
| 无 `languages` 块 | 用 dsconfig 全部语言，下拉显示裸 id（今天行为） |
| 同 `id` 多版本 | 合并为逻辑声库 + 版本下拉 |
| 仅单个版本 | 版本下拉只有一项（或直接隐藏下拉） |

## 12. 完整示例

```yaml
format: tunelab-voicebank/1
id: example.choir
name: 示例声库
name_i18n: { en-US: Example Voicebank }
version: 3
retake:
  acoustic: true
  pitch: true
  variance: false
languages:
  default: zh
  expose:
    - { id: zh, name: 中文, name_i18n: { en-US: Chinese } }
    - { id: en, name: English, name_i18n: { zh-CN: 英语 } }
voices:
  - id: alpha
    speaker: model.alpha
    name: 歌手甲
    name_i18n: { en-US: Singer A }
    default_language: zh
    portrait: spk/alpha.png
    color: "#7AC1FF"
  - { id: beta,  speaker: model.beta,  name: 歌手乙, name_i18n: { en-US: Singer B } }
  - { id: gamma, speaker: model.gamma, name: 歌手丙, name_i18n: { en-US: Singer C } }
  - { id: delta, speaker: model.delta, name: 歌手丁, name_i18n: { en-US: Singer D } }
```

> dsconfig 里的 `sample_rate`/`hop_size`/`use_*_embed`/`speakers` 等一概**不出现**在此文件——它们由插件直接读 dsconfig。

显示效果：voice 列表出现 `示例声库·歌手甲`、`示例声库·歌手乙`、`示例声库·歌手丙`、`示例声库·歌手丁` 四条；各带版本下拉（默认"最新（跟随）"）；底层四者共享同一套已加载模型；隐藏的其余 speaker 既不在列表、也不在任何 voice 的混音候选里。

## 13. 插件侧接入点改动清单

| 模块 | 现状 | 需改动 |
|---|---|---|
| [VoicebankScanner.cs](../VoicebankScanner.cs) | 每个声库文件夹 → 1 个 `DiscoveredVoicebank`，VoiceId = 文件夹名 | 解析 `tunelab.yaml`；按 `id` 跨包分组；按 `voices` 并集展开为多 voice；建"版本 → 物理目录"映射 |
| [DiscoveredVoicebank.cs](../DiscoveredVoicebank.cs) | `(VoiceId, RootPath, Info)` | 引入逻辑声库 + voice + 版本表的承载结构（RootPath 变为按版本解析） |
| [CharacterMetadata.cs](../CharacterMetadata.cs) | 读 character.yaml/.txt | 保留为回退源；`tunelab.yaml` 字段优先 |
| [DiffSingerVoiceEngine.cs](../DiffSingerVoiceEngine.cs) `CreateSession` | 按 voiceId 取单一 RootPath | 按 voiceId + 版本属性解析到具体物理目录；模型缓存 key 纳入版本 |
| [DiffSingerDeclarations.cs](../DiffSingerDeclarations.cs) `BuildPartConfig` | speaker 下拉 + 混音容器 + 语言 | 新增**版本下拉** part 属性；混音候选按 §9 收窄 |
| `BuildFixedAutomationConfigs` | 三条 seed 轨无条件暴露 | 按 `retake.*` 位 gating |
| `VoicebankConfig` / 新增 `TunelabManifest` | 仅解析 dsconfig | dsconfig 仍出 DSP/模型契约；新增类解析 tunelab.yaml 的作者层（id/name+i18n/version/retake/voices/languages），两者不重叠合并 |

## 14. 待定事项（评审决议）

- [ ] `min_engine_version` 是否纳入第一版（与 retake 等新能力的兼容兜底配对）。
- [ ] 授权/分发元数据（`license` / `homepage` / 署名要求）是否纳入。
- [ ] 版本下拉是否额外提供显式"跟随最新"选项（区别于默认 sentinel）的 UI 形态。
- [ ] `voices[].speaker` 是否允许一对多（一个 voice 默认混多个底层 speaker）——当前设计为一对一。
- [ ] 方案 B：是否把 `vocoder` 提为 `tunelab.yaml` 可选字段以与 OpenUtau 解耦（第一版默认走方案 A、仍读 dsconfig）。

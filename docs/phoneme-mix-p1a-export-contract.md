# P1-a：音素级混合（2 路）声学模型重导出契约

> 实验分支 `experiment/phoneme-mix-p1a`。本文是给**重导出 acoustic ONNX 的协作者**的接口规格，
> 目标是让重导出的模型接口与插件（TuneLab C# 端）将要喂入的张量完全对齐。
>
> 这是一次**未经训练的 phoneme embedding 线性插值**实验——落在训练分布之外，好不好听靠烘焙实测裁决。
> P1-a 只验证最便宜的一刀：**音素级、2 路（基准音素 + 混入一个音素，单一比例）**。
> 帧级"音内渐变"(P3) 和 N 路混合是后续、独立的事，本契约不涉及。

---

## 1. 要改什么：只动 phoneme/token embedding 查表

当前 acoustic 模型内部（FastSpeech2 风格 encoder 之前）大致是：

```python
x = txt_embed(tokens)              # [1, nTokens, hidden]  ← phoneme embedding 查表
x = x + lang_embed(languages)      # 多语言模型才有
# → encoder → 按 durations 展开到帧 → + f0/variance/spk_embed → diffusion decoder → mel
```

P1-a **只替换 `txt_embed(tokens)` 这一步的输出**为两个音素 embedding 的逐音素凸组合，其余全部不动：

```python
w = blend.unsqueeze(-1)                                   # [1, nTokens, 1]
x = (1.0 - w) * txt_embed(tokens) + w * txt_embed(tokens_b)
```

- `blend = 0` ⇒ `x = txt_embed(tokens)`，与原模型**逐位一致**。
- `blend = 1` ⇒ `x = txt_embed(tokens_b)`，等价于把该音素整个换成 `tokens_b`。
- `blend ∈ (0,1)` ⇒ 两音素 embedding 的插值（本实验的核心）。

`lang_embed` 仍按**主序列** `languages` 加一次（P1-a 约定主/次音素同语言，跨语言混合暂不支持——见 §4）。

---

## 2. 新增输入（ONNX 图的 input）

在现有 acoustic 输入基础上**新增两个**，形状/对齐与现有 `tokens` 完全一致：

| 名称 | dtype | 形状 | 含义 |
|---|---|---|---|
| `tokens_b` | int64 | `[1, nTokens]` | 次要（目标）音素 token 序列，与 `tokens` **同一音素表、同长度、逐槽对齐** |
| `blend` | float32 | `[1, nTokens]` | 逐音素混合权重 ∈ `[0,1]`；`0`=纯 `tokens`，`1`=纯 `tokens_b`，音素时长内恒定 |

现有输入（`tokens` / `languages` / `durations` / `f0` / `gender` / `velocity` / 各 variance 通道 /
`spk_embed` / `depth` / `steps` / `speedup` / retake 相关）**全部保持不变**。

### 必需 vs 可选

**不要求做成可选输入**——`tokens_b`/`blend` 作为**必需输入**即可。原因：插件按输入元数据 (`HasInput`) 条件喂入，
一旦模型声明了这两个输入，插件**总会**喂；用户没设混合时插件喂 `blend` 全 0（`tokens_b` 随便，取 `tokens` 副本）。
因此**唯一的硬性要求是：`blend` 全 0 时输出必须与"没有这两个输入的原模型"数值一致**（见 §5 验证 1）。

---

## 3. 插件端数据流（C# 会怎么喂，供你对齐命名）

插件在 [`DiffSingerSynthesisSession.cs:270-339`](../DiffSingerSynthesisSession.cs#L270) 构造 `tokens`，
用 `AddL`(int64) / `AddF`(float32) 两个 `HasInput` gating 的 helper 装配张量。P1-a 会新增（伪代码）：

```csharp
// nTokens = phones.Count + 2（前后各一个 SP）
var tokensB = new long[nTokens];
var blend   = new float[nTokens];              // 默认全 0 ⇒ 纯主音素
tokensB[0] = tokensB[nTokens - 1] = AcousticToken(models, "SP");  // padding 槽
// blend[0] = blend[nTokens-1] = 0（永不混合 SP padding）
for (int i = 0; i < phones.Count; i++)
{
    tokensB[i + 1] = AcousticToken(models, /* 该槽的次要音素，默认 = 主音素 */);
    blend[i + 1]   = /* 该槽的混合比例，默认 0 */;
}
AddL("tokens_b", tokensB, new[] { 1, nTokens });   // 仅当 ac.HasInput("tokens_b")
AddF("blend",    blend,   new[] { 1, nTokens });   // 仅当 ac.HasInput("blend")
```

要点：
- `tokens_b` 的 id 来自**与 `tokens` 相同的音素表**（`config.PhonemesFileName`，同一 `AcousticToken` 映射）。
- 长度含前后 2 个 SP padding；**padding 槽 `blend=0`**。
- `blend` 恒定于该音素的所有帧（音素级；这正是 P1-a 与 P3 帧级的区别）。

---

## 4. 边界与约定（P1-a 范围）

- **同音素表**：`tokens` 与 `tokens_b` 共用声学模型自己的 phoneme 字典，插值发生在同一张 embedding 表内。
- **同语言**：`lang_embed` 只按主 `languages` 加一次。若主/次音素分属不同语言，本期**不支持**（结果未定义），留待后续。
- **同一套 durations**：主/次音素共用主序列的时长（`durations` 不变），不存在"两音素自然时长不同"的对齐问题——那是 P3 帧级才需要面对的。
- **凸组合**：2 路权重天然 `(1-w, w)`，和为 1，embedding 幅度不失真。N 路推广（`Σwᵢ·emb`，和为 1）是后续 P1-b，本期不做。

---

## 5. 验证步骤（建议顺序）

1. **默认路径零回归**：重导出后，`blend` 全 0（`tokens_b` 任意）渲染，须与原模型输出**逐值一致**。
   这条过不了，说明 embedding 注入点接错了。
2. **`tokens_b` 接线正确**：某音素 `blend=1`、`tokens_b`=另一音素，渲染结果须**等同于直接把该音素换成那个音素**。
3. **核心实验**：`blend=0.5`（元音对元音，如 /a/↔/o/）——插出来的中间态**听起来是不是一个合理的过渡音素**。
   这是整个特性的生死线；先在同类元音上验，再看辅音/跨类（大概率不行，属预期）。

Python 侧可直接给 ONNX 喂 `tokens/tokens_b/blend` 验证 1–3，不必等插件 C# 端；
插件端 C# 装配（§3）在导出验证通过后再补，以驱动宿主内端到端试听。

---

## 6. 不在本契约内（后续，勿在 P1-a 混入）

- **P1-b**：N 路混合（vocab 软权重 matmul，复用 `DiffSingerSpeakerMix` 的凸归一，做成 `DiffSingerPhonemeMix`）。
- **P3**：帧级 `blend [1,nFrames]`，混合落在时长展开之后，实现"一个音内从 A 渐变到 B"的真·包络。
- 与 `spk_embed`（说话人混合）的交互：两者是正交轴、各自凸归一、图内不同节点注入、模型自然叠加，
  P1-a 无需为此做任何特殊处理（`spk_embed` 路径保持原样）。

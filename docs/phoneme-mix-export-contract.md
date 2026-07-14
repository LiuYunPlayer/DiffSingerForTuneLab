# 音素混合（帧级包络，N 槽）导出契约

> 分支 `experiment/phoneme-mix-p1a`。本文是**模型重导出方**与**插件**之间的接口契约（最终形态）。
> 已实测通过：单槽包络听感良好、N 槽正常、无混合零回归。历史上先做过 P1-a（逐音素 2 路），
> 现已升级为**帧级包络 + 任意 N 槽**——本文描述的是当前生效的契约。

音素混合 = 让一段发音在时间上从音素 A 渐变到 B（可多目标）。与说话人混合正交、可同用。
混合发生在**时长展开之后的帧级**（真·包络），并对 **acoustic + pitch + variance** 三域协调作用。

---

## 1. 模型侧新增输入（DiffSinger 重导出）

所有新输入的槽轴（`n_mix` / `S`）都是**动态 axis**，模型接受任意 N；**全 0 blend ⇒ 与无混合逐值一致**。

### 1.1 acoustic（单体，`FastSpeech2AcousticONNX.forward`）
| 输入 | dtype | 形状 | 说明 |
|---|---|---|---|
| `tokens_b` | int64 | `[S, n_tokens]` | S 条目标音素 token 流（与 `tokens` 同表、同长、逐位对齐） |
| `blend` | float32 | `[S, n_frames]` | 各槽逐帧混合权重 |

图内：`tokens`(base) 与 `tokens_b` 各自 embed→encoder→按 `durations` 展开到帧，
`condition = base_w·cond_base + Σ_s blend[s]·cond_tgt[s]`，`base_w = 1 - Σ_s blend[s]`（**不 clamp**，凸性由插件逐帧归一保证）。

### 1.2 pitch / variance role 模型（pre 段，`forward_pitch/variance_preprocess`）
| 输入 | dtype | 形状 | 说明 |
|---|---|---|---|
| `encoder_out_b` | float32 | `[S, n_tokens, H]` | S 条目标流的 encoder_out（由插件跑 linguistic 得到） |
| `blend` | float32 | `[S, n_frames]` | 各槽逐帧权重 |

图内：`forward_mel2x_gather` 把 base 与各目标 `encoder_out` 展开到帧，逐帧凸组合成单条 condition，
**去噪循环只跑一次**（混合在"拼 condition 之后、去噪之前"，见 `_blend_condition`）。melody/retake/pitch/spk 等共享项在混合后叠加一次。

### 1.3 linguistic（共享编码器）——遗留、可选清理
linguistic 仍带 P1-a 的逐 token `tokens_b [1,n_tokens]` + `blend [1,n_tokens]`（帧级方案里**插件喂 no-op**、不参与混合）。
保留无害；将来可从 linguistic 导出中移除以精简（插件已用 `HasInput` 兜底）。

### 关键要求
- **零回归**：`blend` 全 0（或 `tokens_b`/`encoder_out_b` = base）时，输出必须与无混合模型逐值一致。
- **必需输入**：上述输入一旦导出即为必需——插件会**无条件喂**（无混合时喂 no-op），故导出方无需做可选输入。
- **动态槽轴**：`tokens_b`/`encoder_out_b`/`blend` 的第 0 维标 `dynamic_axes` 为 `n_mix`。

---

## 2. 能力声明（`tunelab.yaml`，同 retake）

```yaml
phoneme_mix: true
```
缺省 `false`。**只有声明 `true` 的声库，插件才暴露音素混合 UI**（老库/未重导出的库完全不显示，零打扰）。

---

## 3. 插件侧（已实现，供导出方对照）

- part 属性「音素混合槽数」`phoneme_mix_slots`（0=关，N=槽 1..N）。
- 键按槽号：曲线 `phoneme_mix:k`（[0,1] 逐帧包络）+ 逐音素 `mix_phoneme:k`（目标音素）/`mix_phoneme_lang:k`（语言）。**号 = 张量槽行 k−1**。
- 逐帧**凸归一** Σ_s blend ≤ 1（多槽和>1 时按和缩放，保 base 权重≥0、不破音）。
- acoustic 喂 `tokens_b`/`blend`；pitch/variance 逐槽跑 linguistic 目标流得 `encoder_out_b`，喂 `encoder_out_b`/`blend`（无目标槽复用 base，等价 no-op）。
- 目标音素按各域**自己的音素表**解析（acoustic/predictor id 空间不共享，查不到即优雅降级不混）。

---

## 4. 导出方 checklist
1. acoustic + pitch + variance 三个模型按 §1 重导出（动态 `n_mix` 轴；blend=0 零回归）。
2. `tunelab.yaml` 加 `phoneme_mix: true`。
3. 同一 checkpoint 即可，**无需重训**（纯推理期 embedding/condition 插值）。
4. 验证：不设混合 → 与旧模型一致；设目标 + 画曲线 → 音素内 morph（三域协调）。

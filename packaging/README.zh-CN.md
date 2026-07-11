# DiffSinger for TuneLab

基于 [DiffSinger](https://github.com/openvpi/DiffSinger) 的 TuneLab 歌声合成引擎。直接读取**社区标准格式**的 DiffSinger 声库——即包含 `dsconfig.yaml` + 角色元数据 + 预测器子目录的模型文件夹——无需转换、无需重新打包。

> 仅支持 **Windows**。可选 GPU 加速（DirectML，兼容大多数独显/核显），无 GPU 时自动可切 CPU。

---

## 1. 安装模型（声库）

插件**不自带任何模型**，需要你自己放入。默认扫描目录：

```
%APPDATA%\DiffSingerForTuneLab\Voices
```

（即 `C:\Users\<你的用户名>\AppData\Roaming\DiffSingerForTuneLab\Voices`，插件首次启动会自动创建这个空目录。）

把**模型文件夹整个**放进去即可，例如：

```
Voices\
└─ 我的模型\
   ├─ dsconfig.yaml          ← 声学主配置（必需）
   ├─ character.yaml         ← 角色元数据（character.yaml 或 character.txt，二选一，必需）
   ├─ acoustic.onnx
   ├─ dsdur\ dspitch\ dsvariance\ …   ← 各预测器子目录
   └─ （可选）tunelab.yaml    ← TuneLab 专属描述，见 §4
```

判定规则：**同时含 `dsconfig.yaml` 与 `character.yaml`（或 `character.txt`）的目录**即被识别为一个模型。目录可以嵌套多层（会向下递归查找），识别到模型后不再深入其子目录。

### 使用其他目录

不想用默认目录、或模型散落在别处：打开 **设置 → 扩展 → DiffSinger**，在 **「声库目录」** 里逐行添加额外目录（每行一个）。默认目录始终生效，添加的目录是**追加**。改完设置即时重新扫描，无需重启。

---

## 2. 安装声码器（Vocoder）

DiffSinger 声学模型输出的是梅尔频谱，需要**声码器**才能还原成声音。默认声码器目录：

```
%APPDATA%\DiffSingerForTuneLab\Vocoders\<声码器名>\
   ├─ vocoder.yaml
   └─ <模型>.onnx
```

`<声码器名>` 必须与模型 `dsconfig.yaml` 里 `vocoder` 字段的值**一致**（大小写敏感）。一个声码器可被多个模型共用，装一次即可。

声码器也可放在别处：打开 **设置 → 扩展 → DiffSinger**，在 **「声码器目录」** 里逐行添加额外目录（每行一个）。默认目录始终生效，添加的目录是**追加**、按顺序查找。

> 若合成后**没有声音**，多半是声码器没装、或名字与 `dsconfig.yaml` 的 `vocoder` 对不上。

---

## 3. 设置项

**设置 → 扩展 → DiffSinger**：

| 设置 | 说明 | 默认 |
|---|---|---|
| **声库目录** | 追加的模型扫描目录（逐行添加） | 空（仅默认目录） |
| **声码器目录** | 追加的声码器扫描目录（逐行添加） | 空（仅默认目录） |
| **执行设备** | `GPU (DirectML)` 或 `CPU`。GPU 明显更快；驱动异常/无独显时改 CPU | GPU (DirectML) |
| **推理模式** | `隔离进程`在独立子进程跑 ONNX，原生崩溃不会拖垮 TuneLab（子进程起不来时——如被杀软拦截——自动退回进程内）；`进程内`直接在 TuneLab 内跑 | 隔离进程 |
| **采样步数** | 扩散采样步数。越大越精细也越慢，通常 20 足够 | 20 |
| **张量缓存** | 缓存推理中间结果，重复合成同段更快、且结果可复现 | 开 |
| **缓存大小上限 (MB)** | 张量缓存磁盘上限，`0` = 不限 | 4096 |

---

## 4. `tunelab.yaml`（可选）

模型目录里**可选**放一个 `tunelab.yaml`，承载基础声库格式表达不了、但 TuneLab 需要的「作者决策层」信息。**没有它，模型照样能用**——插件按默认方式加载（等同没有此文件时的行为：voice id = 文件夹名、整模型一个 voice、speaker 走下拉）。

它能解决的事：

- **稳定的模型 / voice 身份**——不再和文件夹名绑定，改名不失效；
- **多 speaker 拆成多个可选歌手** + 白名单（只暴露想暴露的）；
- **同一个人跨多个模型合并**为一个顶层条目（数据升级重训场景）；
- **版本管理**——同血统多版本，自动跟随最新或显式钉死；
- **retake 能力声明**——note 级 pitch / variance / 音色重摇要求模型用[外置噪声版 DiffSinger](https://github.com/LiuYunPlayer/DiffSinger)（把扩散噪声暴露为 `noise` 输入）构建；标准导出无法重摇。pitch / variance 重摇只需用它**重新导出**，而 **acoustic（音色）重摇还需用它重新训练**；仅按模型实际支持的项声明；
- **本地化**——模型名 / 歌手名 / 语言名按宿主语言显示。

最小示例（`模型目录/tunelab.yaml`）：

```yaml
format: tunelab-voicebank/1
id: myteam.my-model            # 稳定模型 id（合并键）
name: 我的模型                  # 显示名
name_i18n: { en-US: My Model }  # 可选：英文名
version: 1
released: 2026-01              # 可选：跨模型排序（决定“最新”）

retake:                        # 可选：仅当模型确实支持时才写 true
  pitch: true
  variance: false
  acoustic: false

voices:                        # 可选：出现即白名单；缺省=整模型一个 voice
  - { id: singer-a, speaker: spk_a, name: 歌手 A, name_i18n: { en-US: Singer A } }
  - { id: singer-b, speaker: spk_b, name: 歌手 B }
```

字段速览：

| 字段 | 必填 | 说明 |
|---|---|---|
| `format` | ✅ | 固定 `tunelab-voicebank/1` |
| `id` | ✅ | 稳定模型 id；不同模型用相同 `voices[].id` 即视为同一个人、自动合并 |
| `name` / `name_i18n` | `name` ✅ | 模型显示名 + 本地化（key 用 `en-US` / `zh-CN`） |
| `version` | | 同一模型的版本号（整数，越大越新） |
| `version_label` | | 版本的人类可读标签（仅显示） |
| `released` | | `YYYY` / `YYYY-MM` / `YYYY-MM-DD`，跨模型排序用 |
| `retake.{pitch,variance,acoustic}` | | 声明模型是否支持对应重摇；默认全 `false`（不暴露）。声明错了不会崩，合成时静默按不支持处理 |
| `voices[]` | | 暴露的歌手白名单：`id`=全局歌手 id、`speaker`=本模型 dsconfig 后缀、`name`/`name_i18n`/`default_language`/`portrait`/`color` |
| `languages` | | 语言显示名叠加与白名单（`id` 须匹配 dsconfig 语言表键） |

> 解析失败（写错格式）不会让模型消失——插件会告警并降级到默认加载方式。

---

## 5. 常见问题

- **模型没出现在歌手列表里** → 检查目录里是否**同时**有 `dsconfig.yaml` 和 `character.yaml`（或 `.txt`）；确认放对了目录（默认目录或设置里追加的目录）；改完设置会自动重扫。
- **合成没有声音** → 见 §2，多半是声码器缺失或名字对不上。
- **太慢** → 执行设备设为 GPU (DirectML)；或调低采样步数；保持张量缓存开启（重复合成会显著变快）。
- **想复用某次合成结果** → 保持张量缓存开启；同样的输入会命中缓存、结果可复现。

---

许可与第三方署名见随包的 `THIRD-PARTY-NOTICES.md`。

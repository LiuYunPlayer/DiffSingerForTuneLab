# MLRuntime 子进程方案——ONNX 推理进程隔离设计

> 状态：草案 v0.1（待评审）
> 适用：DiffSingerForTuneLab 插件
> 分支：`feat/ml-runtime-subprocess`

## 1. 背景与目标

### 1.1 起因

`onnxruntime-DirectML` 单包内，**同一进程一旦碰过 DML EP，再建 CPU 会话（或反之）会触发原生 `AccessViolation` 直接崩进程**——不可 catch。原 [LoadSession](../DiffSingerModels.cs) 的"每模型 try-DML、失败就地回退 CPU"恰在制造这种混用：某声学模型 DML 建会话失败 → 就地 `new` CPU 会话 → 与进程内已有 DML 会话混用 → 崩，日志停在回退 Warning。

已在 master（`c7e468e`）落地**止血补丁**：执行设备改为进程级同质承诺、不做进程内回退，DML 失败即抛可读异常提示"改 CPU 并重启"。止血版能不崩，但代价是：DML 模式下**任何一个模型**跑不了 DML，整条 part 就只能报错、要用户手动切 CPU 重启。

### 1.2 本方案要根治什么

把 ONNX 原生推理从插件（宿主）进程里**搬到独立子进程 `MLRuntime.exe`**，插件侧只做数据编排。收益：

1. **一整类 onnxruntime 原生崩溃不再拖垮 TuneLab**：DML/CPU 混用、并发 Run 设备崩、dispose use-after-free——子进程崩了就崩它自己，插件检测到重启即可。
2. **provider 回退终于安全**：DML 失败 → 杀掉子进程、重开一个 CPU 子进程（新进程从未碰过 DML、纯 CPU 干净），无 `AccessViolation`。
3. **GPU / CPU 共存**：当 DML 失败是"个别模型算子不支持"（见 §2）而非机器问题时，能跑 DML 的模型走 GPU 子进程、跑不了的路由到 CPU 子进程，而不是整条 part 降 CPU。
4. **退役进程内防御 hack**：现有 [`sRetired` 弱表 / `RetireAndDispose` / 全局 `sRunLock`](../DiffSingerTensorCache.cs) 那套为规避原生崩溃写的复杂度，可随隔离下放或退役。

### 1.3 非目标

- 不追求单块 GPU 上的并行推理（硬件做不到，见 §3.1）。
- 不改张量缓存的磁盘格式、不改推理数值语义（噪声仍插件侧生成、决定论不变）。
- 不改声库格式、发现逻辑、UI。

## 2. DML 失败的两类根因（决定路由策略）

DirectML 建会话失败有两类**互斥**根因，路由策略取决于是哪类：

- **A 机器/驱动问题（全局）**：机器无 DX12 能力 GPU / 驱动坏或太旧 / VM 无 GPU 直通 → **所有模型**都 DML 失败，与声库无关。特征：一个模型都没成功打过 `· DirectML`；错误指向 device/adapter 初始化。
- **B 声库/模型问题（单模型）**：DirectML EP 未实现某算子、或对某特定图 partition/compile 不了 → **有些模型 DML 正常、偏偏某个失败**。特征：同机器别的模型 DML 跑得好；完整错误通常点名某 node/算子（参见 smoke test 里 `/fs2/Sub_1`）。

判据（止血版已把日志改成 `ex.ToString()`，含 node 名/HResult）：
1. 同次会话里声码器 / linguistic 有没有成功 `· DirectML`？有 → B；一个都没成 → A。
2. 完整错误里有没有 node 名。

**对设计的影响**：
- 若是 **A**：所有模型走 CPU → 单 runtime 杀重开 CPU 即可，GPU 路由永不激活。
- 若是 **B**：需要 GPU + CPU 共存路由（§3）。

本设计**按支持共存来做**（成本低、面向未来），但共存是否真激活取决于运行时探测——A 情形下 CPU runtime 是唯一 runtime，不额外开销。

## 3. 进程模型

### 3.1 为什么不是"一个 session 一个进程"

宿主会并发给不同 session 派发合成，直觉上想"一个 session 一个子进程"。但这方向错，理由：

- **单块 GPU 本质不能并行推理**。设备级约束（现有注释：同一 GPU 上不同会话并发 Run 也会崩/出错）；跨进程虽未必崩（D3D12 命令队列按进程隔离、OS 时间片调度），但**一块物理 GPU 被时间片轮流跑，拿不到真并行**。N 个 GPU 进程 ≠ N 倍吞吐。
- **显存翻倍**。每进程各自把模型权重载进自己 VRAM，DirectML 不跨进程共享权重。N 个 session → 同一声学模型在显存存 N 份，极易 OOM。
- **砸掉现有共享**。[DiffSingerModelCache](../DiffSingerModels.cs) 现在多个宿主 session 共享同一声学/声码器会话——故意的省内存优化。按 session 开进程等于丢掉它。

### 3.2 采用：进程数挂钩「执行目标」，不挂钩 session

- **runtime 注册表按「执行目标」键**：`gpu:0`、`cpu`（预留 `gpu:1`… 供多物理 GPU）。
- **每目标默认 1 个共享 runtime**，可扩池（预留，不首发）。
- 所有要 GPU 的 session 路由到共享 GPU runtime，内部串行、权重复用（延续现有共享语义）。
- 宿主的并发派发由**插件侧一个请求队列**吸收成串行喂给 runtime——设备本就要求串行，队列是对的抽象，不需要多进程。

**何时才真需要多进程并行**（均为 profile 驱动的后续优化，注册表结构已预留、届时不重构）：
- 多块物理 GPU：路由到 `gpu:0` / `gpu:1`。
- CPU 榨并行：`cpu` 目标扩成小池（注意"单次 Run 已用满 intra-op 线程池"，收益有限，需实测）。

## 4. 边界与分层

### 4.1 两个天然接缝

1. **所有推理过同一咽喉**：全部 6 类模型（linguistic / dur / pitch / variance / acoustic / vocoder）无一例外走 [`DiffSingerTensorCache.Run`](../DiffSingerTensorCache.cs) → `RunSerialized(model, inputs)`。要外移的原生动作就这一个点。
2. **线格式已写好**：[`SerializeNamedOnnxValue` / `DeserializeNamedOnnxValue`](../DiffSingerTensorCache.cs) 已能把 `List<NamedOnnxValue>` 无损往返成字节流（现用于落盘缓存）。IPC 的请求/响应直接复用，**零新序列化代码**。

### 4.2 谁在哪一侧

```
插件进程（TuneLab 内）                     MLRuntime.exe（独立进程）
─────────────────────                     ──────────────────────
· 数据编排（音素/时长/f0/noise 生成）        · 持有 InferenceSession（模型加载 + provider 决策）
· 张量磁盘缓存（键哈希/Load/Save）           · 收 (modelPath, hash, inputs) → Run → 回 outputs
· 命中→本地返回，不过 IPC                    · 进程内串行锁（DML 设备级不可并发）
· 未命中→发请求给 runtime，收回落盘          · 崩了就崩自己，TuneLab 不受影响
· 持 IModelSession（元数据 + Run 委托）      · 加载时回传 InputMetadata/OutputMetadata
```

**缓存留插件侧是关键**：只有 cache miss 才走 IPC；反复合成/撤销重做那些命中场景完全不跨进程。而扩散推理本身是秒级的，一次 miss 多一趟编组可忽略。

### 4.3 元数据的接缝（不是纯"发输入收输出"）

预测器代码到处读 `session.InputMetadata.ContainsKey("languages")`、`Dimensions[2]` 来决定喂什么输入（如 [DiffSingerVariance.cs](../DiffSingerVariance.cs)）。故 `InferenceSession` 得抽象成 `IModelSession`（只含 `InputMetadata`/`OutputMetadata` + Run），加载时 runtime 把元数据回传插件；插件持一个轻量元数据镜像 + session 句柄。

### 4.4 三层解耦（传输可换）

```
逻辑层    RuntimeClient / IModelSession —— 只知"载模型 P、喂 inputs、拿 outputs"，不碰字节
  │
编解码层  请求/响应 ↔ 字节帧（复用 TensorCache 的 Serialize/Deserialize）—— 与传输无关
  │
传输层    IRuntimeTransport —— "发一个字节消息、收一个字节消息"
           ├─ PipeTransport（首发）
           └─ SharedMemoryTransport（将来：大帧写 MMF、管道只发 offset 控制帧）
```

传输层接口做成**消息帧粒度**（发 `ReadOnlyMemory<byte>`、收 `byte[]`）。管道实现流式收发；共享内存实现把大帧写 MemoryMappedFile、管道传句柄/偏移小控制帧——满足同一接口。**将来换传输只动传输层一个类，逻辑层/编解码层一行不改。**

## 5. 传输：首发命名管道

### 5.1 跨界数据量（只在 cache miss 时传）

按一条 ~10 秒乐句（hop 512 @ 44.1k ≈ 860 帧）估：

| 模型 | 进（inputs） | 出（outputs） | 量级 |
|---|---|---|---|
| linguistic | tokens/languages（int64，几十~几百） | encoder_out `[1,N,~256]` + mask | 几十~几百 KB |
| dur | encoder_out + spk_embed | ph_dur `[N]` | 小 |
| pitch | encoder_out + f0 + retake + noise | pitch `[F]` | 数十~数百 KB |
| variance | 同上 + noise | 数条曲线 `[F]` | 数十~数百 KB |
| **acoustic** | tokens/f0/retake/gt_mel`[1,F,128]`/depth/steps/**noise `[1,feats,outDims,F]`** | **mel `[1,F,128]`** | **noise 数百 KB~数 MB、mel ~440 KB** |
| **vocoder** | mel `[1,F,128]` + f0 | **waveform `[1,samples]`**（samples=F·hop） | **~1.8 MB / 10 秒** |

最大三块：声学 noise、mel、声码器 waveform，均**单 part 单位数 MB**内，且只在 miss 时传一次。

### 5.2 结论

**命名管道（`System.IO.Pipes`）绰绰有余**：2 MB 走管道 ~几毫秒，相对秒级推理可忽略；命中还不走 IPC。共享内存是过早优化，靠 §4.4 的接口留作将来。

## 6. 生命周期与健壮性

### 6.1 孤儿治理（宿主崩了子进程必须跟着死）

双保险：
- **主保险 Win32 Job Object + `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`**：启动子进程时塞进 Job；Job 句柄随宿主进程消亡而关闭 → OS 强制杀 Job 内所有子进程。哪怕宿主硬崩（不走清理），OS 兜底，不留孤儿。（插件侧 P/Invoke 创建 Job 并 `AssignProcessToJobObject`。）
- **副保险 管道 EOF 自杀**：子进程读命名管道，另一端一断即读到 EOF → 立刻退出。比 Job 更快反应正常关闭/崩溃。

### 6.2 崩溃重启

- 插件检测子进程退出 / 管道断（读写抛异常）→ 标记 runtime 失效。
- in-flight 请求：报失败（宿主按 §5.10 标合成失败）或对幂等请求重试一次。
- 下次请求触发重建：spawn 新子进程、重载所需模型（模型本就懒加载）。

### 6.3 provider 回退（安全版）

DML runtime 报某模型 DML 建会话失败 → 插件**杀掉该 runtime、按 `cpu` 目标重开**（或 §3 共存策略下把该模型路由到 CPU runtime）。新进程从未碰 DML，CPU 干净。

### 6.4 换执行设备 / 关闭

- provider 设置变更 → 弃旧 runtime（杀进程）、按新目标建。比现在"Dispose 原生会话"更干净（进程边界即释放边界）。
- 现有 [`RetireAndDispose` / `sRetired`](../DiffSingerTensorCache.cs) 的 use-after-free 防御随之可退役（跨进程无共享原生句柄）。

## 7. 取消与进度

- **取消**：宿主取消合成 → 插件发取消控制帧；runtime 端中断当前 Run（onnxruntime 的 `RunOptions.Terminate` 或最坏杀进程重启）。
- **进度**：vocoder 那步的 `progress?.Report` 现为进程内回调，改为 runtime 回传进度控制帧、插件侧转发给宿主。

## 8. 打包与部署

- `MLRuntime.exe` + `onnxruntime`（含 DirectML/CPU EP）原生库随 .tlx 分发。独立 exe 自带 `runtimes/` 加载原生库，比现在塞进插件 ALC **更简单**。
- 改 [`tools/deploy-dev.ps1`](../tools/deploy-dev.ps1)、[`tools/pack-tlx.ps1`](../tools/pack-tlx.ps1)：多构建/复制一个 exe 产物。

## 9. 分阶段实施

| 阶段 | 内容 | 落点 |
|---|---|---|
| **P0** | 本设计文档 | 设计记录（本文件） |
| **P1** | `IModelSession` 抽象 + `InProcessModelSession`（进程内直跑）。所有 `InferenceSession` 直接引用换成接口 | **编译+行为完全等价**，可单独提交、可回归 |
| **P2** | 线协议（LoadModel/Run/健康/关闭/取消/进度帧）+ 复用 TensorCache 序列化；先做进程内 loopback 跑通协议、验往返无损 | 不起子进程 |
| **P3** | `MLRuntime.exe` host：持会话、收帧跑、回帧、串行锁、provider 决策移入；接线 deploy/pack | 新产物 |
| **P4** | `RuntimeClient` + `RemoteModelSession`（IModelSession 第二实现，走 IPC）+ 生命周期（Job Object、EOF 自杀、崩溃重启、DML 失败→CPU） | 隔离生效点 |
| **P5** | 取消/进度跨进程；退役进程内防御 hack | |
| **P6** | 端到端验证、回归、退役无用代码、文档更新 | |

P1 落地后 master 就有"抽象已就位、仍进程内直跑"的稳定态，即便后续拖长也无半成品卡中间。

## 10. 待定 / 依赖外部输入

- **A/B 判定**：`0507aco.HaoYe_ZH` 的 DML 失败是 A（机器）还是 B（声库）——等重测把新 `Error` 日志贴出确认，据此定单/双 runtime 是否真激活 CPU 路由。
- **CPU 目标是否扩池**：profile 驱动，首发单进程。
- **取消粒度**：`RunOptions.Terminate` 是否够、还是靠杀进程——P5 实测定。

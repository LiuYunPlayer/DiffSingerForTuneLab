# DiffSingerForTuneLab

DiffSinger 歌声合成引擎，TuneLab 的 voice 插件（个人开发、独立公开仓库，产出 `.tlx`）。

## 开发环境

本仓库需与 [TuneLab](../TuneLab) 主仓**并排放置**：

```
GitHub/
  TuneLab/                # 宿主 + SDK 契约源 + 文档 + 范例夹具
  DiffSingerForTuneLab/   # 本仓
```

用根目录的 `DiffSingerForTuneLab.code-workspace` 打开，会同时挂上本仓与 TuneLab（只读参考：SDK 源、docs、tests/plugins 范例）。

### SDK 引用

开发期通过跨仓 `ProjectReference`（`Private=false`）引用 `TuneLab.Foundation` 与 `TuneLab.SDK`，
两者由宿主统一提供、**不打进包**。目标框架锁 `net8.0`（SDK ABI 地板）。

## 构建

```
dotnet build
```

## 参考

- 人类版开发文档：`../TuneLab/docs/plugin-development.md` §5（Voice）
- AI 事实清单：`../TuneLab/docs/plugin-development-llm.md`（Voice 段）
- SDK 真相源：`../TuneLab/TuneLab.SDK/Voice/`
- 范例夹具：`../TuneLab/tests/plugins/V1.Voice`（分块正弦）、`V1.I18N`（单块/本地化）

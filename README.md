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

## 发版

版本号存在**三处，必须同步**：`manifest.json` 的 `version`（宿主唯一读取处）、`DiffSingerForTuneLab.csproj` 的 `<Version>`（dll 元数据）、git tag `v<version>`。

```
pwsh tools/pack-tlx.ps1          # 本地出 tools/tlx/<id>-<version>.tlx
git tag v1.0.0 && git push origin v1.0.0
```

推 tag 触发 `.github/workflows/release.yml`：校验 tag 与 manifest 版本一致 → 跑单测 → 打包 → 建**草稿** Release 并附 `.tlx`，人工过目发行说明后手动发布。

## 许可

本仓库以 [MIT](LICENSE.txt) 发布。移植/捆绑的第三方代码与数据（OpenUtau，MIT）另见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)；两份许可文本均随 `.tlx` 落到安装目录。

## 参考

- 人类版开发文档：`../TuneLab/docs/plugin-development.md` §5（Voice）
- AI 事实清单：`../TuneLab/docs/plugin-development-llm.md`（Voice 段）
- SDK 真相源：`../TuneLab/TuneLab.SDK/Voice/`
- 范例夹具：`../TuneLab/tests/plugins/V1.Voice`（分块正弦）、`V1.I18N`（单块/本地化）

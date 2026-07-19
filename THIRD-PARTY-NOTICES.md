# Third-Party Notices

本插件包含从下列第三方项目移植或捆绑的代码与数据。

## OpenUtau

- 项目：OpenUtau — https://github.com/stakira/OpenUtau
- 许可证：MIT License
- 用途：
  - **OpenUtau.Core 门面程序集**（移植）：`OpenUtauFacade/` 下的 `IG2p.cs`、`G2pDictionary.cs`、
    `G2pDictionaryData.cs`、`G2pPack.cs`、`PhonemizerAttribute.cs` 忠实移植自 `OpenUtau.Core/Api/`
    （类型全名与成员签名保持原样——声库自带的 OpenUtau 自定义音素器 DLL 按
    `OpenUtau.Core, Version=1.0.0.0` 身份绑定这些类型）；`DiffSingerPhonemizerBase.cs` 是
    `DiffSingerBasePhonemizer` / `DiffSingerG2pPhonemizer` 的空壳基类（虚方法签名取自
    `OpenUtau.Core/DiffSinger/`）。
  - **G2P 框架代码**（移植）：`G2p/G2pRemapper.cs`、`G2p/G2pFallbacks.cs`，以及全部内置算法引擎类
    `G2p/{ArpabetPlus, Korean, FrenchMillefeuille, German, GermanMarzipan, Spanish, Italian,
    Portuguese, Russian, Filipino}G2p.cs`
    移植自 `OpenUtau.Core/Api/` 与 `OpenUtau.Core/G2p/`，并适配本插件命名空间与可空标注。
  - **托管程序集判定**（移植）：`ExternalPhonemizers.cs` 的 `IsManagedAssembly` 移植自
    `OpenUtau.Core/Util/LibraryLoader.cs`；声库音素器 DLL 的扫描/装载流程对齐
    `OpenUtau.Core/DocManager.SearchAllPlugins`。
  - **G2P 资源包**（捆绑）：`G2p/Data/g2p-{arpabet-plus, ko, fr-millefeuille, de, de-marzipan,
    es, it, pt, ru, fil}.zip` 取自 OpenUtau 仓 `OpenUtau.Core/G2p/Data/`，各含
    `dict.txt` + `phones.txt` + `g2p.onnx`。
    - 英语 ARPA+ 包由 OpenUtau 社区贡献者 Cadlaxa 提供；英语词表为 CMUdict 系。
    - 韩语包含规则式谚文→jamo 分解 + 神经 OOV 模型。
    - 法语 Millefeuille 与德语 Marzipan 方案由 UFR 提供（经上游合并）。

### MIT License（OpenUtau）

```
The MIT License (MIT)

Copyright (c) stakira and OpenUtau contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

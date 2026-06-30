# Third-Party Notices

本插件包含从下列第三方项目移植或捆绑的代码与数据。

## OpenUtau

- 项目：OpenUtau — https://github.com/stakira/OpenUtau
- 许可证：MIT License
- 用途：
  - **G2P 框架代码**（移植）：`G2p/IG2p.cs`、`G2p/G2pDictionary.cs`、`G2p/G2pRemapper.cs`、
    `G2p/G2pFallbacks.cs`、`G2p/G2pPack.cs`、`G2p/ArpabetPlusG2p.cs`、`G2p/KoreanG2p.cs`
    移植自 `OpenUtau.Core/Api/` 与 `OpenUtau.Core/G2p/`，并适配本插件命名空间与可空标注。
  - **G2P 资源包**（捆绑）：`G2p/Data/g2p-arpabet-plus.zip`、`G2p/Data/g2p-ko.zip`
    取自 OpenUtau 仓 `OpenUtau.Core/G2p/Data/`，各含 `dict.txt` + `phones.txt` + `g2p.onnx`。
    - 英语 ARPA+ 包由 OpenUtau 社区贡献者 Cadlaxa 提供；英语词表为 CMUdict 系。
    - 韩语包含规则式谚文→jamo 分解 + 神经 OOV 模型。

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

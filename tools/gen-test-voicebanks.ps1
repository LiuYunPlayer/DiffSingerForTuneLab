# 重新生成 _test_* 桩声库到插件声库目录（%AppData%/DiffSingerForTuneLab/Voices）。
# 这些桩只有 YAML、没有模型文件，用于测试 tunelab.yaml 清单解析：
#   A=无 tunelab.yaml 的 legacy 库  B=有清单但无 voices 块  C=单模型多 voice
#   D=同 id 双版本  E=两库共享同一 voice(合唱团)  F=同 voice 同时出现在两个模型(X 有双版本)
# 用法: pwsh tools/gen-test-voicebanks.ps1 [-VoicesRoot <目录>]
param(
    [string]$VoicesRoot = (Join-Path $env:APPDATA 'DiffSingerForTuneLab/Voices')
)

$ErrorActionPreference = 'Stop'

# 所有桩共用同一份 dsconfig（speakers/languages 每库不同）
function DsConfig([string[]]$Speakers, [string[]]$Languages) {
    $spk = ($Speakers -join ', ')
    $lang = ($Languages -join ', ')
    $yaml = @"
phonemes: phonemes.json
acoustic: acoustic.onnx
vocoder: testvocoder
hidden_size: 256
sample_rate: 44100
hop_size: 512
use_lang_id: true
use_key_shift_embed: true
use_speed_embed: true
use_energy_embed: true
speakers: [$spk]
languages: [$lang]
"@
    return $yaml + "`n"
}

$banks = @(
    @{
        Dir = '_test_A_legacy'
        Character = "name: 测A-Legacy声库`n"
        DsConfig = DsConfig @('spk_a', 'spk_b') @('zh')
    }
    @{
        Dir = '_test_B_novoices'
        Character = "name: 旧名应被覆盖`n"
        DsConfig = DsConfig @('spk_only') @('zh')
        Tunelab = @"
format: tunelab-voicebank/1
id: test.novoices
name: 测B-无voices块
name_i18n: { en-US: Test-B NoVoices }
"@
    }
    @{
        Dir = '_test_C_single'
        Character = "name: 测C-单模型`n"
        DsConfig = DsConfig @('spk_alpha', 'spk_beta', 'spk_gamma') @('zh', 'en')
        Tunelab = @"
format: tunelab-voicebank/1
id: test.single
name: 测C-单模型
name_i18n: { en-US: Test-C Single }
version: 1
retake:
  acoustic: true
  pitch: true
  variance: false
languages:
  default: zh
  expose:
    - { id: zh, name: 中文, name_i18n: { en-US: Chinese } }
    - { id: en, name: 英语, name_i18n: { en-US: English } }
voices:
  - { id: c_alpha, speaker: spk_alpha, name: 阿尔法, name_i18n: { en-US: Alpha } }
  - { id: c_beta,  speaker: spk_beta,  name: 贝塔,   name_i18n: { en-US: Beta } }
"@
    }
    @{
        Dir = '_test_D_dual_v1'
        Character = "name: 测D-双版本v1`n"
        DsConfig = DsConfig @('spk_x') @('zh')
        Tunelab = @"
format: tunelab-voicebank/1
id: test.dual
name: 测D-双版本模型
version: 1
released: 2023-01-01
retake: { pitch: true }
voices:
  - { id: d_singer, speaker: spk_x, name: 双版本歌手, name_i18n: { en-US: Dual Singer } }
"@
    }
    @{
        Dir = '_test_D_dual_v2'
        Character = "name: 测D-双版本v2`n"
        DsConfig = DsConfig @('spk_x') @('zh')
        Tunelab = @"
format: tunelab-voicebank/1
id: test.dual
name: 测D-双版本模型
version: 2
version_label: 2024版
version_label_i18n: { en-US: 2024 Edition }
released: 2024-06
retake: { pitch: true }
voices:
  - { id: d_singer, speaker: spk_x, name: 双版本歌手, name_i18n: { en-US: Dual Singer } }
"@
    }
    @{
        Dir = '_test_E_choirA'
        Character = "name: 测E-合唱团A`n"
        DsConfig = DsConfig @('spk_c', 'spk_a2') @('zh')
        Tunelab = @"
format: tunelab-voicebank/1
id: test.choirA
name: 测E-合唱团A
name_i18n: { en-US: Test-E Choir A }
version: 1
released: 2023-06
voices:
  - { id: e_common, speaker: spk_c,  name: 共同歌手, name_i18n: { en-US: Common Singer } }
  - { id: e_onlya,  speaker: spk_a2, name: 仅A歌手,  name_i18n: { en-US: Only-A Singer } }
"@
    }
    @{
        Dir = '_test_E_choirB'
        Character = "name: 测E-合唱团B`n"
        DsConfig = DsConfig @('spk_c', 'spk_b2') @('zh')
        Tunelab = @"
format: tunelab-voicebank/1
id: test.choirB
name: 测E-合唱团B
name_i18n: { en-US: Test-E Choir B }
version: 1
released: 2024-06
voices:
  - { id: e_common, speaker: spk_c,  name: 共同歌手, name_i18n: { en-US: Common Singer } }
  - { id: e_onlyb,  speaker: spk_b2, name: 仅B歌手,  name_i18n: { en-US: Only-B Singer } }
"@
    }
    @{
        Dir = '_test_F_bothX_v1'
        Character = "name: 测F-模型X-v1`n"
        DsConfig = DsConfig @('spk_f') @('zh')
        Tunelab = @"
format: tunelab-voicebank/1
id: test.bothX
name: 测F-模型X
name_i18n: { en-US: Test-F Model X }
version: 1
released: 2024-01
voices:
  - { id: f_both, speaker: spk_f, name: 双选歌手, name_i18n: { en-US: Both Selector } }
"@
    }
    @{
        Dir = '_test_F_bothX_v2'
        Character = "name: 测F-模型X-v2`n"
        DsConfig = DsConfig @('spk_f') @('zh')
        Tunelab = @"
format: tunelab-voicebank/1
id: test.bothX
name: 测F-模型X
name_i18n: { en-US: Test-F Model X }
version: 2
released: 2024-06
voices:
  - { id: f_both, speaker: spk_f, name: 双选歌手, name_i18n: { en-US: Both Selector } }
"@
    }
    @{
        Dir = '_test_F_bothY_v1'
        Character = "name: 测F-模型Y`n"
        DsConfig = DsConfig @('spk_f') @('zh')
        Tunelab = @"
format: tunelab-voicebank/1
id: test.bothY
name: 测F-模型Y
name_i18n: { en-US: Test-F Model Y }
version: 1
released: 2023-06
voices:
  - { id: f_both, speaker: spk_f, name: 双选歌手, name_i18n: { en-US: Both Selector } }
"@
    }
)

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
foreach ($bank in $banks) {
    $dir = Join-Path $VoicesRoot $bank.Dir
    New-Item -ItemType Directory -Force $dir | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $dir 'character.yaml'), $bank.Character, $utf8NoBom)
    [System.IO.File]::WriteAllText((Join-Path $dir 'dsconfig.yaml'), $bank.DsConfig, $utf8NoBom)
    if ($bank.Tunelab) {
        [System.IO.File]::WriteAllText((Join-Path $dir 'tunelab.yaml'), $bank.Tunelab, $utf8NoBom)
    }
    Write-Host "生成 $dir"
}
Write-Host "完成：$($banks.Count) 个测试桩声库已写入 $VoicesRoot"

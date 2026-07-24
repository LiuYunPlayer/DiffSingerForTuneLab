# 打 .tlx 分发包：构建插件并把输出目录【内容】压成 zip（改名 .tlx）。
# .tlx = zip，根目录直接含 manifest.json + dll + runtimes/ 子树（勿套外层文件夹）。
# ONNX 原生库在 runtimes/win-x64/native/，靠整树递归打入、勿扁平化。
# 只有 .tlx 文件能被 TuneLab 安装（拖文件夹不装，见 Editor.OnDrop / InstallExtensions）。
# 用法: pwsh tools/pack-tlx.ps1 [-Configuration Release]
param([string]$Configuration = "Release")

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression.FileSystem

$repo = Split-Path $PSScriptRoot -Parent
$source = Join-Path $repo "bin/$Configuration/net8.0"
$out = Join-Path $PSScriptRoot "tlx"

dotnet build (Join-Path $repo "DiffSingerForTuneLab.csproj") -c $Configuration
dotnet build (Join-Path $repo "MLRuntime/MLRuntime.csproj") -c $Configuration

# MLRuntime.exe 子进程：暂存进插件输出的 mlruntime/ 子目录（自带 onnxruntime + runtimes/），随后一并打进 .tlx。
$mlSource = Join-Path $repo "MLRuntime/bin/$Configuration/net8.0"
$mlStage = Join-Path $source "mlruntime"
if (Test-Path $mlStage) { Remove-Item $mlStage -Recurse -Force }
New-Item -ItemType Directory -Force -Path $mlStage | Out-Null
Copy-Item -Path (Join-Path $mlSource "*") -Destination $mlStage -Recurse -Force

# 去重 onnxruntime：MLRuntime 子进程经 OnnxNativeResolver 改从父插件目录 runtimes/ 加载原生库，
# 不再自带一份（省 ~15MB×平台）。删掉暂存里的 mlruntime/runtimes/（父目录整树 runtimes/ 仍在）。
$mlRuntimes = Join-Path $mlStage "runtimes"
if (Test-Path $mlRuntimes) { Remove-Item $mlRuntimes -Recurse -Force }

# 剪除输出【根目录】冗余的原生库副本 + DirectML 调试符号：规范副本在 runtimes/<rid>/native/，
# 宿主经 deps.json（AssemblyDependencyResolver）从那里解析、根部这几份用不上。某些 SDK 版本
# （如 CI 的 10.x）会把它们额外拷到输出根、白占 ~17MB，某些（本机 8.x）不拷——显式剪除以保证
# 跨环境产物一致、精简。只删根部，runtimes/ 整树不动。
foreach ($f in 'DirectML.dll','DirectML.pdb','DirectML.Debug.dll','DirectML.Debug.pdb','onnxruntime.dll','onnxruntime.lib') {
    $p = Join-Path $source $f
    if (Test-Path $p) { Remove-Item $p -Force }
}

# 从 manifest.json 取 id + version 命名产物
$desc = Get-Content (Join-Path $source "manifest.json") -Raw | ConvertFrom-Json
$tlx = Join-Path $out ("$($desc.id)-$($desc.version).tlx")

New-Item -ItemType Directory -Force -Path $out | Out-Null
if (Test-Path $tlx) { Remove-Item $tlx -Force }
[System.IO.Compression.ZipFile]::CreateFromDirectory($source, $tlx)

Write-Host "已打包 $tlx"

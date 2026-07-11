# 打 .tlx 分发包：构建插件并把输出目录【内容】压成 zip（改名 .tlx）。
# .tlx = zip，根目录直接含 description.json + dll + runtimes/ 子树（勿套外层文件夹）。
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

# 从 description.json 取 id + version 命名产物
$desc = Get-Content (Join-Path $source "description.json") -Raw | ConvertFrom-Json
$tlx = Join-Path $out ("$($desc.id)-$($desc.version).tlx")

New-Item -ItemType Directory -Force -Path $out | Out-Null
if (Test-Path $tlx) { Remove-Item $tlx -Force }
[System.IO.Compression.ZipFile]::CreateFromDirectory($source, $tlx)

Write-Host "已打包 $tlx"

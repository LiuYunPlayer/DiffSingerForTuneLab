# dev 部署：构建插件并镜像输出到 TuneLab 扩展目录。
# 因 ONNX 原生库在 runtimes/win-x64/native/（宿主 ALC 经 deps.json 解析），必须连子目录树一起拷、勿扁平化。
# 用法: pwsh tools/deploy-dev.ps1 [-Configuration Debug]
param([string]$Configuration = "Debug")

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$source = Join-Path $repo "bin/$Configuration/net8.0"
$dest = Join-Path $env:APPDATA "TuneLab/Extensions/diffsingerfortunelab"

dotnet build (Join-Path $repo "DiffSingerForTuneLab.csproj") -c $Configuration

# 清旧 + 整树镜像（保留 runtimes/ 等子目录）。SDK 程序集因 Private=false 不在输出，无需排除。
if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item -Path (Join-Path $source "*") -Destination $dest -Recurse -Force

Write-Host "已部署到 $dest"

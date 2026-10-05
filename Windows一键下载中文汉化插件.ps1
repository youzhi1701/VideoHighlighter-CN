$ErrorActionPreference = "Stop"

$RepoZip = "https://github.com/youzhi1701/VideoHighlighter-CN/archive/refs/heads/main.zip"
$ZipFile = Join-Path $env:TEMP "VideoHighlighter-CN-main.zip"
$ExtractRoot = Join-Path $env:TEMP "VideoHighlighter-CN-Plugin-Extract"
$DownloadRoot = Join-Path $env:USERPROFILE "Downloads"
$TargetDir = Join-Path $DownloadRoot "VideoHighlighter-中文汉化插件"

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

Write-Host "VideoHighlighter 中文汉化插件 一键下载" -ForegroundColor Green

Write-Step "下载最新版汉化插件"
if (Test-Path $ZipFile) {
    Remove-Item $ZipFile -Force
}
if (Test-Path $ExtractRoot) {
    Remove-Item $ExtractRoot -Recurse -Force
}

Invoke-WebRequest -Uri $RepoZip -OutFile $ZipFile -UseBasicParsing

Write-Step "解压插件"
Expand-Archive -Path $ZipFile -DestinationPath $ExtractRoot -Force

$SourceDir = Join-Path $ExtractRoot "VideoHighlighter-CN-main\中文汉化插件"
if (-not (Test-Path $SourceDir)) {
    throw "压缩包中未找到“中文汉化插件”目录。"
}

if (Test-Path $TargetDir) {
    $BackupDir = "$TargetDir-旧版"
    if (Test-Path $BackupDir) {
        Remove-Item $BackupDir -Recurse -Force
    }
    Rename-Item -Path $TargetDir -NewName (Split-Path $BackupDir -Leaf)
}

Copy-Item -Path $SourceDir -Destination $TargetDir -Recurse -Force

Write-Step "清理临时文件"
Remove-Item $ZipFile -Force -ErrorAction SilentlyContinue
Remove-Item $ExtractRoot -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "中文汉化插件下载完成。" -ForegroundColor Green
Write-Host "保存位置：" -ForegroundColor Yellow
Write-Host $TargetDir
Write-Host ""
Write-Host "已为你打开插件目录。"
Start-Process explorer.exe $TargetDir

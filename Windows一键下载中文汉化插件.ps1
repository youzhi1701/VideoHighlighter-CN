$ErrorActionPreference = "Stop"

$RepoZip = "https://github.com/youzhi1701/VideoHighlighter-CN/archive/refs/heads/main.zip"
$ZipFile = Join-Path $env:TEMP "VideoHighlighter-CN-main.zip"
$ExtractRoot = Join-Path $env:TEMP "VideoHighlighter-CN-Plugin-Extract"
$DownloadRoot = Join-Path $env:USERPROFILE "Downloads"

# Build the Chinese folder name without embedding non-ASCII source text.
$ChineseName = -join @(
    [char]0x4E2D,  # Zhong
    [char]0x6587,  # Wen
    [char]0x6C49,  # Han
    [char]0x5316,  # Hua
    [char]0x63D2,  # Cha
    [char]0x4EF6   # Jian
)
$TargetDir = Join-Path $DownloadRoot ("VideoHighlighter-" + $ChineseName)
$BackupDir = $TargetDir + "-old"

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

Write-Host "VideoHighlighter CN localization plugin downloader" -ForegroundColor Green

Write-Step "Downloading latest repository package..."
if (Test-Path $ZipFile) {
    Remove-Item $ZipFile -Force
}
if (Test-Path $ExtractRoot) {
    Remove-Item $ExtractRoot -Recurse -Force
}

Invoke-WebRequest -Uri $RepoZip -OutFile $ZipFile -UseBasicParsing

Write-Step "Extracting package..."
Expand-Archive -Path $ZipFile -DestinationPath $ExtractRoot -Force

# Avoid relying on a Chinese path literal. Locate the plugin by a unique asset.
$Marker = Get-ChildItem -Path $ExtractRoot -Filter "ChineseSimplified.isl" -File -Recurse |
    Where-Object { $_.FullName -match "[\\/]resources?[\\/]|[\\/][^\\/]+[\\/]ChineseSimplified\.isl$" } |
    Select-Object -First 1

if (-not $Marker) {
    # Fallback: any ChineseSimplified.isl in the extracted repository.
    $Marker = Get-ChildItem -Path $ExtractRoot -Filter "ChineseSimplified.isl" -File -Recurse |
        Select-Object -First 1
}

if (-not $Marker) {
    throw "Localization plugin marker file was not found in the downloaded archive."
}

# Marker is expected at: plugin/resources/ChineseSimplified.isl
$SourceDir = $Marker.Directory.Parent.FullName

if (-not (Test-Path (Join-Path $SourceDir "README.md"))) {
    throw "Localization plugin directory could not be identified safely."
}

Write-Step "Copying plugin to Downloads..."
if (Test-Path $BackupDir) {
    Remove-Item $BackupDir -Recurse -Force
}
if (Test-Path $TargetDir) {
    Move-Item -Path $TargetDir -Destination $BackupDir -Force
}

Copy-Item -Path $SourceDir -Destination $TargetDir -Recurse -Force

Write-Step "Cleaning temporary files..."
Remove-Item $ZipFile -Force -ErrorAction SilentlyContinue
Remove-Item $ExtractRoot -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Localization plugin download completed." -ForegroundColor Green
Write-Host "Saved to:" -ForegroundColor Yellow
Write-Host $TargetDir
Write-Host ""
Write-Host "Opening plugin folder..."
Start-Process explorer.exe $TargetDir

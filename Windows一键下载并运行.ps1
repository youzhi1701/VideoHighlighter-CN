$ErrorActionPreference = "Stop"

$RepoZip = "https://github.com/youzhi1701/VideoHighlighter-CN/archive/refs/heads/main.zip"
$InstallRoot = Join-Path $env:LOCALAPPDATA "VideoHighlighter-CN"
$ZipFile = Join-Path $env:TEMP "VideoHighlighter-CN-main.zip"
$ExtractRoot = Join-Path $env:TEMP "VideoHighlighter-CN-Extract"
$ProjectDir = Join-Path $InstallRoot "VideoHighlighter-CN-main"
$VenvDir = Join-Path $ProjectDir ".venv"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

function Write-Step([string]$Text) {
    Write-Host ""
    Write-Host "==> $Text" -ForegroundColor Cyan
}

Write-Host "VideoHighlighter-CN Windows 一键下载并运行" -ForegroundColor Green
Write-Host "安装目录：$InstallRoot"

Write-Step "检查 Python 3.12"
$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if (-not $winget) {
            throw "未检测到 Python，也未检测到 winget。请先安装 Python 3.12 后重新运行。"
        }
        Write-Host "未检测到 Python，正在通过 winget 安装 Python 3.12..."
        winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) {
            throw "Python 3.12 安装失败。"
        }
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        $py = Get-Command py -ErrorAction SilentlyContinue
        $python = Get-Command python -ErrorAction SilentlyContinue
    }
}

Write-Step "下载最新版 VideoHighlighter-CN"
if (Test-Path $ZipFile) { Remove-Item $ZipFile -Force }
if (Test-Path $ExtractRoot) { Remove-Item $ExtractRoot -Recurse -Force }

Invoke-WebRequest -Uri $RepoZip -OutFile $ZipFile -UseBasicParsing

Write-Step "解压程序"
Expand-Archive -Path $ZipFile -DestinationPath $ExtractRoot -Force

New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
if (Test-Path $ProjectDir) {
    $backup = "$ProjectDir.backup"
    if (Test-Path $backup) { Remove-Item $backup -Recurse -Force }
    Rename-Item $ProjectDir $backup
}
Move-Item (Join-Path $ExtractRoot "VideoHighlighter-CN-main") $ProjectDir -Force

Write-Step "创建独立 Python 环境"
if ($py) {
    & py -3.12 -m venv $VenvDir
} elseif ($python) {
    & python -m venv $VenvDir
} else {
    throw "Python 安装完成后仍无法找到命令，请重新打开 PowerShell 后再运行。"
}

Write-Step "升级 pip"
& $PythonExe -m pip install --upgrade pip setuptools wheel

Write-Step "安装基础依赖（首次运行时间较长）"
& $PythonExe -m pip install numpy==2.2.6 mkl mkl-include
& $PythonExe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
& $PythonExe -m pip install openvino==2026.4.0
& $PythonExe -m pip install -r (Join-Path $ProjectDir "requirements.txt")
& $PythonExe -m pip install yolox==0.3.0 --no-deps

Write-Step "启动 VideoHighlighter-CN"
Set-Location $ProjectDir
Start-Process -FilePath $PythonExe -ArgumentList @("main.py") -WorkingDirectory $ProjectDir

Write-Host ""
Write-Host "已启动 VideoHighlighter-CN。" -ForegroundColor Green
Write-Host "程序目录：$ProjectDir"
Write-Host "以后可直接运行：" -ForegroundColor Yellow
Write-Host "$PythonExe $ProjectDir\main.py"

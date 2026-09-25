<#
.SYNOPSIS
    Set up reproducible Python venvs for the Flower and SecretFlow local VFL runners.

.DESCRIPTION
    For each framework this script:
      1. Creates <framework>\.venv if it doesn't already exist.
      2. Upgrades pip/setuptools/wheel inside that venv.
      3. Installs the pinned requirements.txt (CPU torch from the PyTorch index).
      4. Verifies `import torch, torchvision, pandas, sklearn, PIL` succeeds.

    Re-running is safe - existing venvs are reused unless -Recreate is passed.

.PARAMETER Framework
    Which venv to build. Default: 'both'. One of: 'flower', 'secretflow', 'both'.

.PARAMETER Recreate
    Delete and rebuild the venv(s) from scratch.

.PARAMETER Python
    Python executable to bootstrap from. Default: 'python'. Use this when you have
    multiple Pythons installed, e.g. -Python 'py -3.10'.

.PARAMETER CudaVersion
    Pull a CUDA-enabled torch build instead of CPU, e.g. 'cu126' or 'cu128'
    (check pytorch.org for the tags available for the pinned torch version).
    Default: 'cpu' (works everywhere, no GPU required).

.EXAMPLE
    .\bootstrap.ps1

.EXAMPLE
    .\bootstrap.ps1 -Framework flower -Recreate

.EXAMPLE
    .\bootstrap.ps1 -CudaVersion cu128    # use your RTX GPU during training
#>
param(
    [ValidateSet('flower','secretflow','both')]
    [string]$Framework = 'both',
    [switch]$Recreate,
    [string]$Python = 'python',
    [string]$CudaVersion = 'cpu'
)

$ErrorActionPreference = 'Stop'
$ModelsRoot = $PSScriptRoot

$frameworks = if ($Framework -eq 'both') { @('Flower','SecretFlow') } else { @((Get-Culture).TextInfo.ToTitleCase($Framework)) }
$torchIndex = "https://download.pytorch.org/whl/$CudaVersion"

function Invoke-VenvPython {
    param([string]$Venv, [string[]]$PyArgs)
    $exe = Join-Path $Venv 'Scripts\python.exe'
    if (-not (Test-Path $exe)) { throw "venv python not found at $exe" }
    & $exe @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "python $($PyArgs -join ' ') exited with $LASTEXITCODE" }
}

foreach ($fw in $frameworks) {
    $fwDir  = Join-Path $ModelsRoot $fw
    $venv   = Join-Path $fwDir '.venv'
    $req    = Join-Path $fwDir 'requirements.txt'

    if (-not (Test-Path $req)) {
        Write-Warning "[$fw] no requirements.txt at $req - skipping."
        continue
    }

    Write-Host "`n=== $fw ===" -ForegroundColor Cyan

    if ($Recreate -and (Test-Path $venv)) {
        Write-Host "[$fw] removing existing venv..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venv
    }

    if (-not (Test-Path $venv)) {
        Write-Host "[$fw] creating venv at $venv" -ForegroundColor Green
        & $Python -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw "venv creation failed for $fw" }
    } else {
        Write-Host "[$fw] reusing existing venv at $venv" -ForegroundColor DarkGray
    }

    Write-Host "[$fw] upgrading pip / setuptools / wheel" -ForegroundColor Green
    Invoke-VenvPython $venv @('-m','pip','install','--upgrade','pip','setuptools','wheel')

    Write-Host "[$fw] installing pinned deps (torch from $torchIndex)" -ForegroundColor Green
    Invoke-VenvPython $venv @(
        '-m','pip','install',
        '--extra-index-url', $torchIndex,
        '-r', $req
    )

    Write-Host "[$fw] verifying imports" -ForegroundColor Green
    Invoke-VenvPython $venv @('-c', "import torch, torchvision, pandas, sklearn, PIL, numpy; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| pandas', pandas.__version__, '| sklearn', sklearn.__version__, '| numpy', numpy.__version__, '| cuda', torch.cuda.is_available())")

    Write-Host "[$fw] verifying vfl_paths resolver" -ForegroundColor Green
    Invoke-VenvPython $venv @((Join-Path $fwDir 'vfl_paths.py'))
}

Write-Host "`nBootstrap complete." -ForegroundColor Cyan
Write-Host "Next: run .\smoke_test.ps1 to verify end-to-end training + unlearning." -ForegroundColor Cyan

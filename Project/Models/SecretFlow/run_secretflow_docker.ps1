param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("mnist", "nih", "celeba", "launcher")]
    [string]$Target,

    [string]$Dataset,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$containerWorkspace = "/workspace"
$containerSecretFlow = "$containerWorkspace/SecretFlow"
$containerFlower = "$containerWorkspace/Flower"
$hostHome = [Environment]::GetFolderPath("UserProfile")
$hostNihCache = Join-Path $hostHome ".cache\kagglehub\datasets\nih-chest-xrays\data\versions\3"
$hostCelebaCache = Join-Path $hostHome ".cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2"
$containerNih = "/datasets/nih"
$containerCelebaRoot = "/datasets"
$image = "secretflow/secretflow-anolis8:latest"

# Host data folder (same one fetch_data.py writes to): VFL_DATA_ROOT env var,
# else data_root from ..\config.yaml, else C:\691\data. Mounted at /data.
$hostDataRoot = $env:VFL_DATA_ROOT
if (-not $hostDataRoot) {
    $cfg = Join-Path $workspaceRoot "config.yaml"
    if (Test-Path $cfg) {
        $line = Select-String -Path $cfg -Pattern '^\s*data_root:\s*(.+?)\s*$' | Select-Object -First 1
        if ($line) { $hostDataRoot = $line.Matches[0].Groups[1].Value.Trim('"', "'") }
    }
}
if (-not $hostDataRoot) { $hostDataRoot = "C:\691\data" }
$containerData = "/data"

switch ($Target) {
    "mnist" {
        $script = "run_vfl_mnist_local.py"
        $defaultArgs = @(
            "--data-root", $containerData
        )
    }
    "nih" {
        $script = "run_vfl_nih_chest_xray14_local.py"
        $defaultArgs = @(
            "--data-root", "$(if (Test-Path $hostNihCache) { $containerNih } else { "$containerData/nih_chest_xray14" })"
        )
    }
    "celeba" {
        $script = "run_vfl_celeba_local.py"
        $defaultArgs = @(
            "--data-root", "$(if (Test-Path $hostCelebaCache) { $containerCelebaRoot } else { $containerData })"
        )
    }
    "launcher" {
        $script = "run_vfl.py"
        $defaultArgs = @()
    }
}

$allArgs = @()
if ($script -eq "run_vfl.py") {
    $allArgs += "--dataset"
    if ($Dataset) {
        $allArgs += $Dataset
        if (-not ($ExtraArgs -contains "--data-root")) {
            switch ($Dataset) {
                "mnist" { $allArgs += @("--data-root", $containerData) }
                "celeba" {
                    if (Test-Path $hostCelebaCache) {
                        $allArgs += @("--data-root", $containerCelebaRoot)
                    } else {
                        $allArgs += @("--data-root", $containerData)
                    }
                }
                "nih" {
                    if (Test-Path $hostNihCache) {
                        $allArgs += @("--data-root", $containerNih)
                    } else {
                        $allArgs += @("--data-root", "$containerData/nih_chest_xray14")
                    }
                }
            }
        }
        if ($ExtraArgs) {
            $allArgs += $ExtraArgs
        }
    } else {
        throw "For Target=launcher, pass -Dataset mnist, celeba, or nih."
    }
} else {
    $allArgs += $defaultArgs
    if ($ExtraArgs) {
        $allArgs += $ExtraArgs
    }
}

$dockerArgs = @(
    "run",
    "--rm",
    "--entrypoint",
    "python",
    "-v",
    "${workspaceRoot}:${containerWorkspace}",
    "-w",
    $containerSecretFlow
)

if (Test-Path $hostDataRoot) {
    $dockerArgs += @("-v", "${hostDataRoot}:${containerData}")
}

if (Test-Path $hostNihCache) {
    $dockerArgs += @("-v", "${hostNihCache}:${containerNih}")
}

if (Test-Path $hostCelebaCache) {
    $dockerArgs += @("-v", "${hostCelebaCache}:/datasets/celeba")
}

$dockerArgs += @($image, $script)
$dockerArgs += $allArgs

docker @dockerArgs

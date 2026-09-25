<#
.SYNOPSIS
    End-to-end smoke test of the VFL runners after bootstrap.ps1 has succeeded.

.DESCRIPTION
    For each selected framework (Flower, SecretFlow) and dataset (mnist, celeba, nih):
      1. Run a tiny baseline + unlearning pass (1 epoch, few batches, CPU).
      2. Verify metrics.csv exists with rows for phase=baseline AND phase=unlearn,
         each including attack_accuracy (ASR) and attack_auc columns.
      3. Verify run_summary.csv has a row containing baseline_*, final_*, and
         delta_* columns for the 14 tracked metrics.

    MNIST auto-downloads via torchvision. CelebA and NIH require real data at
    data_root (run `python fetch_data.py celeba nih` first). Datasets without data are skipped.

.PARAMETER Framework
    Which framework(s) to smoke-test. Default: 'both'.

.PARAMETER Dataset
    Which dataset(s) to smoke-test. Default: 'all'. Any of: mnist,celeba,nih,all.

.PARAMETER TrainBatches
    Cap batches per epoch. Default 3. Lower = faster, less signal.

.PARAMETER TestBatches
    Cap eval batches. Default 2.

.EXAMPLE
    .\smoke_test.ps1                         # smoke-test everything

.EXAMPLE
    .\smoke_test.ps1 -Dataset mnist          # fastest path, no local data required

.EXAMPLE
    .\smoke_test.ps1 -Framework flower -Dataset mnist,celeba
#>
param(
    [ValidateSet('flower','secretflow','both')]
    [string]$Framework = 'both',
    [string[]]$Dataset = @('all'),
    [int]$TrainBatches = 3,
    [int]$TestBatches  = 2
)

$ErrorActionPreference = 'Stop'
$ModelsRoot = $PSScriptRoot

# ---- normalise args ----------------------------------------------------------
$frameworks = if ($Framework -eq 'both') { @('Flower','SecretFlow') } else { @((Get-Culture).TextInfo.ToTitleCase($Framework)) }
$datasets = if ($Dataset -contains 'all') { @('mnist','celeba','nih') } else { $Dataset | ForEach-Object { $_.ToLower() } }

# ---- resolve data + logs roots from config.yaml via vfl_paths ----------------
$paths = & (Join-Path $ModelsRoot 'Flower\.venv\Scripts\python.exe') (Join-Path $ModelsRoot 'Flower\vfl_paths.py') |
    ForEach-Object { $_ }
$dataRoot = ($paths | Where-Object { $_ -match '^data_root:' }) -replace '^data_root:\s*',''
$logsRoot = ($paths | Where-Object { $_ -match '^logs_root:' }) -replace '^logs_root:\s*',''
Write-Host "data_root = $dataRoot"
Write-Host "logs_root = $logsRoot"

# ---- helpers -----------------------------------------------------------------
function Test-DatasetPresent {
    param([string]$Dataset)
    switch ($Dataset) {
        'mnist'  { return $true }  # auto-downloads
        'celeba' {
            return (Test-Path (Join-Path $dataRoot 'celeba\list_attr_celeba.txt')) -and
                   (Test-Path (Join-Path $dataRoot 'celeba\img_align_celeba'))
        }
        'nih'    {
            return (Test-Path (Join-Path $dataRoot 'nih_chest_xray14\Data_Entry_2017.csv'))
        }
        default  { return $false }
    }
}

function Invoke-RunnerSmoke {
    param([string]$Framework, [string]$Dataset)
    $fwDir  = Join-Path $ModelsRoot $Framework
    $py     = Join-Path $fwDir '.venv\Scripts\python.exe'
    $runner = switch ($Dataset) {
        'mnist'  { 'run_vfl_mnist_local.py' }
        'celeba' { 'run_vfl_celeba_local.py' }
        'nih'    { 'run_vfl_nih_chest_xray14_local.py' }
    }
    $script = Join-Path $fwDir $runner

    $extra = @()
    if ($Dataset -eq 'mnist') { $extra += '--download' }
    if ($Framework -eq 'SecretFlow') { $extra += '--skip-secretflow-check' }

    Write-Host "`n  [$Framework/$Dataset] running $runner" -ForegroundColor Cyan
    & $py $script `
        --epochs 1 `
        --unlearn-epochs 1 `
        --repair-epochs 1 `
        --max-train-batches $TrainBatches `
        --max-test-batches  $TestBatches `
        @extra
    if ($LASTEXITCODE -ne 0) { throw "[$Framework/$Dataset] runner exit code $LASTEXITCODE" }
}

function Test-RunOutput {
    param([string]$Framework, [string]$Dataset)
    $fwTag  = $Framework.ToLower()
    $dsDir  = Join-Path $logsRoot "$fwTag\$Dataset"
    if (-not (Test-Path $dsDir)) { throw "no output dir $dsDir" }

    # find newest run
    $newest = Get-ChildItem $dsDir -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $metrics = Join-Path $newest.FullName 'metrics.csv'
    $summary = Join-Path (Join-Path $logsRoot $fwTag) 'run_summary.csv'   # one summary per framework

    if (-not (Test-Path $metrics)) { throw "missing metrics.csv in $($newest.FullName)" }
    if (-not (Test-Path $summary)) { throw "missing $summary" }

    # validate metrics.csv
    $rows = Import-Csv $metrics
    $hasBaseline = $rows | Where-Object { $_.phase -eq 'baseline' } | Measure-Object | Select-Object -Expand Count
    $hasUnlearn  = $rows | Where-Object { $_.phase -eq 'unlearn'  } | Measure-Object | Select-Object -Expand Count
    if ($hasBaseline -lt 1) { throw "no phase=baseline row in $metrics" }
    if ($hasUnlearn  -lt 1) { throw "no phase=unlearn row in $metrics" }

    $asrMissing = $rows | Where-Object { -not $_.attack_accuracy -or -not $_.attack_auc }
    if ($asrMissing) { throw "ASR columns empty for some rows in $metrics" }

    # validate run_summary.csv - must have at least one row with baseline_*/final_*/delta_*
    $summ = Import-Csv $summary | Select-Object -Last 1
    foreach ($req in 'baseline_attack_accuracy','final_attack_accuracy','delta_attack_accuracy',
                     'baseline_forget_acc','final_forget_acc','delta_forget_acc',
                     'unlearn_epochs','repair_epochs','forget_fraction') {
        if (-not $summ.PSObject.Properties[$req]) { throw "summary missing column $req" }
        if ([string]::IsNullOrEmpty($summ.$req))  { throw "summary column $req is empty"   }
    }

    Write-Host "  [$Framework/$Dataset] OK" -ForegroundColor Green
    Write-Host "      metrics.csv : $metrics"
    Write-Host "      baseline ASR -> final ASR : $($summ.baseline_attack_accuracy) -> $($summ.final_attack_accuracy)  (delta $($summ.delta_attack_accuracy))"
}

# ---- drive the matrix --------------------------------------------------------
$skipped = @()
$failed  = @()
$passed  = @()

foreach ($fw in $frameworks) {
    $py = Join-Path $ModelsRoot "$fw\.venv\Scripts\python.exe"
    if (-not (Test-Path $py)) {
        Write-Warning "[$fw] venv not found at $py - run .\bootstrap.ps1 first."
        $skipped += "$fw/(all)"
        continue
    }
    foreach ($ds in $datasets) {
        if (-not (Test-DatasetPresent $ds)) {
            Write-Warning "[$fw/$ds] dataset not present at $dataRoot - skipping."
            $skipped += "$fw/$ds"
            continue
        }
        try {
            Invoke-RunnerSmoke   -Framework $fw -Dataset $ds
            Test-RunOutput       -Framework $fw -Dataset $ds
            $passed += "$fw/$ds"
        } catch {
            Write-Host "  [$fw/$ds] FAILED: $_" -ForegroundColor Red
            $failed += "$fw/$ds"
        }
    }
}

Write-Host "`n=== Smoke test summary ==="
Write-Host ("  passed  : {0}" -f ($passed  -join ', '))
Write-Host ("  skipped : {0}" -f ($skipped -join ', '))
Write-Host ("  failed  : {0}" -f ($failed  -join ', '))

if ($failed.Count -gt 0) { exit 1 } else { exit 0 }

# Running VFL locally

Datasets and training logs live **outside** the repository (and outside
OneDrive): code is in git, data is downloaded on demand, logs go to a local
folder. Both Flower and SecretFlow read their paths through a shared
resolver, `vfl_paths.py`, so both frameworks share one copy of the data.

## Reproducible setup

From a PowerShell prompt at the repo root:

```powershell
.\bootstrap.ps1                         # create .venv per framework + install pinned deps
.\Flower\.venv\Scripts\Activate.ps1
python fetch_data.py mnist celeba nih --limit 10000 --nih-size 256
.\smoke_test.ps1 -Dataset mnist         # 1-epoch end-to-end proof
```

Each step is idempotent — re-running does the right thing.

`bootstrap.ps1` reads `Flower\requirements.txt` and `SecretFlow\requirements.txt`
(both pin `torch==2.11.0`, `torchvision==0.26.0`, `numpy==2.4.4`,
`pandas==3.0.2`, `scikit-learn==1.8.0`, `Pillow==12.1.1`). It pulls torch
from the official PyTorch CPU index. Pass `-CudaVersion cu128` (or another
tag listed on pytorch.org) to train on your GPU, `-Python 'py -3.13'` to pick
an interpreter, and `-Recreate` to rebuild the venvs from scratch.

`smoke_test.ps1` runs a truncated baseline + unlearning pass (1 epoch,
3 train batches, 2 eval batches) for each dataset that's present on disk,
then asserts that `metrics.csv` contains both `phase=baseline` and
`phase=unlearn` rows with ASR populated, and that `run_summary.csv`
contains the `baseline_*`, `final_*`, and `delta_*` columns. Datasets
without local data are skipped. MNIST auto-downloads via torchvision.

## Getting the data: `fetch_data.py`

```powershell
python fetch_data.py --status                           # what's on disk, and how big
python fetch_data.py mnist                              # torchvision, ~60 MB
python fetch_data.py celeba --limit 20000               # Hugging Face flwrlabs/celeba
python fetch_data.py nih --limit 10000 --nih-size 256   # Hugging Face timm/nih-chest-xray-14
python fetch_data.py all                                # everything, full size
python fetch_data.py --clean nih                        # delete a local copy to free space
```

| Flag | What it does |
| --- | --- |
| `--limit N` | at most N images per split (train / test), shuffled with `--seed` so a subset is representative |
| `--nih-size N` | resize X-rays to NxN on the way in (runners resize to 224 anyway; 256 cuts ~45 GB to a few GB) |
| `--data-root PATH` | write somewhere other than `data_root` from `config.yaml` |
| `--force` | re-download even if the dataset is already there |

CelebA and NIH are streamed from Hugging Face, so only the rows you keep
are downloaded and there's no second copy in a cache. The script writes the
layout below, which is what `common_*_vfl.py` already reads — the training
code is unchanged.

Note: the Hugging Face copies are repackaged versions of the originals
(CelebA with its official train/test split; NIH with the timm train/test
split, which the NIH loader re-splits 80/20 as before). Numbers from a
`--limit` subset are not directly comparable to full-dataset runs, so record
the flags you used alongside your results.

## Layout

```
C:\691\
  data\                          <- data_root
    MNIST\raw\
    celeba\
      list_attr_celeba.csv       (or the original .txt files)
      list_eval_partition.csv
      img_align_celeba\
    nih_chest_xray14\
      Data_Entry_2017.csv
      images\
  logs\                          <- logs_root
    flower\
    secretflow\
```

If you already have the original datasets somewhere, skip `fetch_data.py`
and point `data_root` (or the per-dataset overrides) at them.

## Configuration

All path choices live in `config.yaml` at the repo root:

```yaml
data_root: C:\691\data
logs_root: C:\691\logs
# celeba_root: D:\datasets\celeba       # optional per-dataset overrides
# mnist_root:  D:\datasets\MNIST
# nih_root:    D:\datasets\nih_chest_xray14
```

Resolution order inside the Python code:

1. Explicit `--data-root` passed on the command line.
2. Environment variable — `VFL_DATA_ROOT`, `VFL_LOGS_ROOT`,
   `VFL_CELEBA_ROOT`, `VFL_MNIST_ROOT`, `VFL_NIH_ROOT`.
3. The values in `config.yaml` (walked upward from the run script).
4. Hard-coded defaults (`C:\691\data`, `C:\691\logs`).

For someone else cloning the repo, setting `VFL_DATA_ROOT` / `VFL_LOGS_ROOT`
(or editing `config.yaml`) is all that's needed to use their own folders.

Sanity-check the resolver with:

```powershell
.\Flower\.venv\Scripts\python.exe .\Flower\vfl_paths.py
```

It prints the exact paths each runner will use.

## Phase 1 — baseline VFL on all three datasets

Each runner trains the baseline model and logs per-epoch metrics to
`C:\691\logs\<framework>\<dataset>\<run_id>\metrics.csv`, plus a one-line
entry in `run_summary.csv`.

Flower:

```powershell
cd Models\Flower
.\.venv\Scripts\python.exe .\run_vfl.py --dataset mnist
.\.venv\Scripts\python.exe .\run_vfl.py --dataset celeba
.\.venv\Scripts\python.exe .\run_vfl.py --dataset nih
```

SecretFlow (uses the same three datasets; Docker wrappers in
`run_secretflow_*_docker.cmd` still work):

```powershell
cd Models\SecretFlow
python .\run_vfl.py --dataset mnist
python .\run_vfl.py --dataset celeba
python .\run_vfl.py --dataset nih
```

Quick smoke tests:

```powershell
.\.venv\Scripts\python.exe .\run_vfl_mnist_local.py --epochs 1 --max-train-batches 3 --max-test-batches 2
```

## Phase 2 — unlearning + ASR

The unlearning + membership-inference (ASR) evaluation is built into the
same runners. `vfl_unlearning.py` splits the training set into retain /
forget loaders and `run_vfl_<dataset>_local.py` runs it automatically
after baseline training. Relevant flags:

| Flag | What it controls |
| --- | --- |
| `--forget-fraction 0.1` | Fraction of training data treated as "forget" set |
| `--unlearn-epochs 1`    | Gradient-ascent epochs on the forget set |
| `--repair-epochs 1`     | Retain-set repair epochs after each unlearn epoch |
| `--seed 42`             | RNG seed for the retain/forget split |

Example: run a phase-1 baseline then a heavier phase-2 unlearning pass on
CelebA:

```powershell
.\.venv\Scripts\python.exe .\run_vfl_celeba_local.py `
    --epochs 3 `
    --forget-fraction 0.15 `
    --unlearn-epochs 3 `
    --repair-epochs 2
```

Per-epoch output CSV captures both phases:

- `phase=baseline` rows for phase 1 (rows 1..`--epochs`).
- `phase=unlearn` rows for phase 2 (rows 1..`--unlearn-epochs`).

Each row includes train/retain/forget/test loss and accuracy plus the
membership-inference attack metrics (`attack_accuracy`, `attack_auc`,
`attack_f1`, etc.). ASR is `attack_accuracy` — you want that to drop
from phase 1 to phase 2 without destroying `retain_acc` or `test_acc`.

### Side-by-side comparison in `run_summary.csv`

For each run, one row is appended to
`C:\691\logs\<framework>\run_summary.csv` (shared across datasets) with pre/post
unlearning columns so you don't need to re-parse `metrics.csv` to eyeball
the change. The 14 tracked metrics are:

```
train_loss, train_acc, retain_loss, retain_acc,
forget_loss, forget_acc, test_loss, test_acc,
attack_accuracy, attack_precision, attack_recall, attack_f1,
attack_auc, attack_threshold
```

Each one is emitted in three columns:

- `baseline_<metric>` — value captured at the end of phase 1 (last
  baseline epoch, before any unlearning).
- `final_<metric>`    — value after the last unlearn + repair pass.
- `delta_<metric>`    — signed difference `final - baseline` (e.g.
  `delta_attack_accuracy = -0.12` means ASR dropped by 12 points).

The summary row also records `unlearn_epochs`, `repair_epochs`, and
`forget_fraction` alongside the existing `run_id`, `dataset`, `epochs`,
`batch_size`, `device`, `metrics_csv`, and `notes` columns.

## Why this changes nothing about the experiments

The model code, loss function, split logic, and attack evaluator are all
untouched — only the *paths* they read from changed. So baseline numbers
from runs done against the original local copy remain directly comparable,
as long as the same data (full set vs. `--limit` subset) is used.

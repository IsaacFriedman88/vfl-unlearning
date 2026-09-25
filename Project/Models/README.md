# Federated Unlearning: Flower vs. SecretFlow

Two-party Vertical Federated Learning (VFL) experiments comparing the
Flower and SecretFlow frameworks across three datasets. Each dataset runs
in two phases:

1. **Baseline** — train a two-party VFL model with full metrics logged.
2. **Unlearning** — apply gradient-ascent-based unlearning on a forget
   set, then repair on the retain set, and re-measure accuracy plus the
   membership-inference Attack Success Rate (ASR).

| Dataset | Party A | Party B | Task |
| --- | --- | --- | --- |
| MNIST | left half of the digit | right half of the digit | 10-class digit |
| CelebA | face image | `Eyeglasses` attribute | predict `Smiling` |
| NIH Chest X-ray14 | left half of the X-ray | right half of the X-ray | binary finding (default `Infiltration`) |

The per-epoch `metrics.csv` captures train/retain/forget/test loss and
accuracy alongside the six attack metrics. The per-run `run_summary.csv`
appends a side-by-side view with `baseline_*`, `final_*`, and `delta_*`
columns so pre/post-unlearning changes can be read at a glance.

## Quick start (Windows, PowerShell)

```powershell
# 1. Build the per-framework venvs with pinned dependencies
.\bootstrap.ps1

# 2. Download the data you need (goes to data_root in config.yaml, NOT the repo)
.\Flower\.venv\Scripts\Activate.ps1
python fetch_data.py mnist
python fetch_data.py celeba --limit 20000              # max 20k images per split; omit for all
python fetch_data.py nih --limit 10000 --nih-size 256  # full NIH is ~45 GB

# 3. Prove the pipeline end-to-end
.\smoke_test.ps1 -Dataset mnist
```

See [LOCAL_SETUP.md](LOCAL_SETUP.md) for the full walkthrough: runner
flags, the unlearning phase, CUDA options, and how paths are resolved
through `config.yaml` + `vfl_paths.py`.

## Data

No datasets are stored in this repository. `fetch_data.py` downloads them
on demand into `data_root` (default `C:\691\data`, set in `config.yaml` or
the `VFL_DATA_ROOT` environment variable) in exactly the layout the
loaders expect:

| Dataset | Source | Full size |
| --- | --- | --- |
| MNIST | torchvision downloader | ~60 MB |
| CelebA | Hugging Face [`flwrlabs/celeba`](https://huggingface.co/datasets/flwrlabs/celeba) | ~1–2 GB as JPEG |
| NIH Chest X-ray14 | Hugging Face [`timm/nih-chest-xray-14`](https://huggingface.co/datasets/timm/nih-chest-xray-14) | ~45 GB (much less with `--nih-size 256`) |

CelebA and NIH are streamed, so only the rows you ask for are downloaded
and nothing is duplicated in a hidden cache. Check what's on disk with
`python fetch_data.py --status`, and free the space again with
`python fetch_data.py --clean nih` (or `--clean all`) whenever you like.
If you already have the original datasets, point `data_root` at them
instead; the loaders read the original file layout as well.

## Repository layout

```
Flower/                      Flower-based local runners
  run_vfl.py                 unified launcher (--dataset mnist|celeba|nih)
  run_vfl_<dataset>_local.py
  common_*_vfl.py            dataset loaders + models
  vfl_unlearning.py          retain/forget split, unlearning, MIA metrics
  run_logging.py, vfl_paths.py
  Client-/Server-Isaac_Laptop.py   original networked Flower client/server
  tests/
SecretFlow/                  SecretFlow-mirror runners (same layout)
  secretflow_vfl_shared.py
  run_secretflow_*_docker.*  Docker wrappers for the official SecretFlow image
  tests/
paper/                       LaTeX draft + bibliography
fetch_data.py                download datasets on demand
config.yaml                  data_root / logs_root for both frameworks
bootstrap.ps1                create venvs + install pinned deps
smoke_test.ps1               end-to-end truncated run + asserts
LOCAL_SETUP.md               full documentation
```

Training logs are written to `logs_root` (default `C:\691\logs`) and are
not committed.

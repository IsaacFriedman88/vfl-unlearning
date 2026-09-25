"""Download the experiment datasets on demand, outside the git repo.

Nothing in this repo ships data. Run this once on a new machine (or any time
you've deleted the data to free space) and it writes each dataset into the
folder the runners already read from (``data_root`` in config.yaml, which
defaults to C:\\691\\data). The layout it produces is exactly what
common_*_vfl.py expects, so none of the training code changes.

Sources
-------
    mnist   torchvision's built-in downloader                         (~60 MB)
    celeba  Hugging Face  flwrlabs/celeba   (train + test splits)     (~1-2 GB as JPEG)
    nih     Hugging Face  timm/nih-chest-xray-14                      (~45 GB full size)

CelebA and NIH are *streamed* from Hugging Face, so only the rows you ask for
are downloaded and nothing is duplicated in a hidden cache. Use ``--limit`` to
pull a subset and ``--nih-size`` to shrink the X-rays on the way in.

Examples (from the repo root, with a venv active)
-------------------------------------------------
    python fetch_data.py --status
    python fetch_data.py mnist
    python fetch_data.py celeba --limit 20000
    python fetch_data.py nih --limit 10000 --nih-size 256
    python fetch_data.py all
    python fetch_data.py --clean nih          # delete the local copy again
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# vfl_paths lives next to the runners; reuse it so data lands where they look.
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / "Flower"))
import vfl_paths  # noqa: E402

CELEBA_HF_REPO = "flwrlabs/celeba"
NIH_HF_REPO = "timm/nih-chest-xray-14"

CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
]

# The CelebA loader only reads partition 0 (train) and 2 (test).
CELEBA_SPLITS = {"train": 0, "test": 2}
NIH_SPLITS = ("train", "test")


# --------------------------------------------------------------------- paths


def dataset_dirs(data_root: Path) -> dict[str, Path]:
    """Where each dataset lives. Matches what the loaders read."""
    paths = vfl_paths.resolve()
    return {
        # common_mnist_vfl: datasets.MNIST(root=data_root) -> data_root/MNIST
        "mnist": data_root / "MNIST",
        # common_celeba_vfl: data_root/celeba if it exists
        "celeba": data_root / "celeba",
        # common_nih_chest_xray14_vfl: nih_root (defaults to data_root/nih_chest_xray14)
        "nih": paths.nih_root if data_root == paths.data_root else data_root / "nih_chest_xray14",
    }


def is_complete(name: str, folder: Path) -> bool:
    if name == "mnist":
        return (folder / "raw" / "train-images-idx3-ubyte").exists()
    if name == "celeba":
        return (folder / "list_attr_celeba.csv").exists() or (folder / "list_attr_celeba.txt").exists()
    if name == "nih":
        return (folder / "Data_Entry_2017.csv").exists()
    return False


def folder_size(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ------------------------------------------------------------------- helpers


def _hf_stream(repo: str, split: str, limit: int | None, seed: int):
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit(
            "The Hugging Face 'datasets' package is required for CelebA / NIH.\n"
            "Install it into your venv:  python -m pip install datasets"
        )
    ds = load_dataset(repo, split=split, streaming=True)
    if limit is not None:
        # Shuffle before truncating so a subset isn't just the first few
        # shards (which are often sorted by identity / patient).
        ds = ds.shuffle(seed=seed, buffer_size=5_000).take(limit)
    return ds


class _Progress:
    def __init__(self, label: str, total: int | None):
        self.label, self.total, self.n = label, total, 0
        self.t0 = time.time()

    def tick(self) -> None:
        self.n += 1
        if self.n % 500 == 0:
            rate = self.n / max(time.time() - self.t0, 1e-6)
            of = f"/{self.total}" if self.total else ""
            print(f"  {self.label}: {self.n}{of} images ({rate:.0f}/s)", flush=True)


# -------------------------------------------------------------------- MNIST


def fetch_mnist(data_root: Path, **_: object) -> None:
    from torchvision import datasets

    for train in (True, False):
        datasets.MNIST(root=str(data_root), train=train, download=True)


# ------------------------------------------------------------------- CelebA


def fetch_celeba(data_root: Path, limit: int | None, seed: int, **_: object) -> None:
    import pandas as pd

    out = dataset_dirs(data_root)["celeba"]
    img_dir = out / "img_align_celeba"
    img_dir.mkdir(parents=True, exist_ok=True)

    attr_rows, part_rows = [], []
    idx = 0
    for split, partition in CELEBA_SPLITS.items():
        progress = _Progress(f"celeba/{split}", limit)
        for row in _hf_stream(CELEBA_HF_REPO, split, limit, seed):
            idx += 1
            name = f"{idx:06d}.jpg"
            row["image"].convert("RGB").save(img_dir / name, quality=95)
            attr_rows.append({"image_id": name, **{a: 1 if row[a] else -1 for a in CELEBA_ATTRS}})
            part_rows.append({"image_id": name, "partition": partition})
            progress.tick()

    # Written last: their presence marks the download as complete.
    pd.DataFrame(part_rows).to_csv(out / "list_eval_partition.csv", index=False)
    pd.DataFrame(attr_rows, columns=["image_id", *CELEBA_ATTRS]).to_csv(out / "list_attr_celeba.csv", index=False)


# ---------------------------------------------------------------------- NIH


def fetch_nih(data_root: Path, limit: int | None, seed: int, nih_size: int | None, **_: object) -> None:
    import pandas as pd

    out = dataset_dirs(data_root)["nih"]
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in NIH_SPLITS:
        progress = _Progress(f"nih/{split}", limit)
        for row in _hf_stream(NIH_HF_REPO, split, limit, seed):
            name = str(row["image_id"])
            if not name.lower().endswith(".png"):
                name += ".png"
            image = row["image"].convert("L")
            if nih_size:
                image = image.resize((nih_size, nih_size))
            image.save(img_dir / name)
            labels = [str(x) for x in (row.get("label_names") or [])]
            rows.append(
                {
                    "Image Index": name,
                    "Finding Labels": "|".join(labels) if labels else "No Finding",
                    "Patient ID": row.get("patient_id"),
                    "Patient Age": row.get("patient_age"),
                    "Patient Gender": row.get("patient_sex"),
                    "View Position": row.get("view_position"),
                    "HF Split": split,
                }
            )
            progress.tick()

    # Written last: its presence marks the download as complete.
    pd.DataFrame(rows).to_csv(out / "Data_Entry_2017.csv", index=False)


FETCHERS = {"mnist": fetch_mnist, "celeba": fetch_celeba, "nih": fetch_nih}


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download VFL datasets into data_root (outside the repo).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples", 1)[-1],
    )
    parser.add_argument("datasets", nargs="*", metavar="DATASET", help="mnist, celeba, nih, or all")
    parser.add_argument("--data-root", help="override data_root from config.yaml")
    parser.add_argument("--limit", type=int, help="max images per split (CelebA / NIH). Omit for the full set.")
    parser.add_argument("--nih-size", type=int, help="resize NIH X-rays to NxN on download (e.g. 256)")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed used when --limit is set")
    parser.add_argument("--force", action="store_true", help="re-download even if the dataset is already present")
    parser.add_argument("--status", action="store_true", help="show what is on disk and exit")
    parser.add_argument("--clean", nargs="+", choices=[*FETCHERS, "all"], help="delete local copies and exit")
    args = parser.parse_args(argv)
    unknown = set(args.datasets) - {*FETCHERS, "all"}
    if unknown:
        parser.error(f"unknown dataset(s): {', '.join(sorted(unknown))} (choose from mnist, celeba, nih, all)")

    data_root = Path(args.data_root) if args.data_root else vfl_paths.resolve().data_root
    dirs = dataset_dirs(data_root)
    print(f"data_root: {data_root}")

    if args.status or not (args.datasets or args.clean):
        for name, folder in dirs.items():
            state = "ready" if is_complete(name, folder) else ("partial" if folder.exists() else "missing")
            print(f"  {name:7s} {state:8s} {human(folder_size(folder)):>10s}  {folder}")
        if not (args.status or args.clean):
            print("\nNothing to fetch. Try:  python fetch_data.py mnist   (see --help)")
        return

    if args.clean:
        names = list(FETCHERS) if "all" in args.clean else args.clean
        for name in names:
            if dirs[name].exists():
                print(f"deleting {dirs[name]} ({human(folder_size(dirs[name]))})")
                shutil.rmtree(dirs[name])
        return

    names = list(FETCHERS) if "all" in args.datasets else args.datasets
    data_root.mkdir(parents=True, exist_ok=True)
    for name in names:
        folder = dirs[name]
        if is_complete(name, folder) and not args.force:
            print(f"[{name}] already present at {folder} (use --force to re-download)")
            continue
        if folder.exists() and name != "mnist":
            shutil.rmtree(folder)  # clear a partial / forced download
        print(f"[{name}] downloading into {folder} ...")
        FETCHERS[name](data_root, limit=args.limit, seed=args.seed, nih_size=args.nih_size)
        print(f"[{name}] done - {human(folder_size(folder))} on disk")


if __name__ == "__main__":
    main()

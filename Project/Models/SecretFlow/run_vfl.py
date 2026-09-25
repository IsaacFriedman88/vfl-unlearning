from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RUNNERS = {
    "celeba": "run_vfl_celeba_local.py",
    "mnist": "run_vfl_mnist_local.py",
    "nih": "run_vfl_nih_chest_xray14_local.py",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Unified local VFL launcher for CelebA, MNIST, and NIH Chest X-ray14 in the SecretFlow folder."
    )
    parser.add_argument("--dataset", choices=sorted(RUNNERS), required=True)
    args, extra = parser.parse_known_args()
    return args, extra


def main() -> None:
    args, extra = parse_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    script_path = Path(__file__).with_name(RUNNERS[args.dataset])
    command = [sys.executable, str(script_path), *extra]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()

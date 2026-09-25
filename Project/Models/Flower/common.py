"""Backward-compatible CelebA VFL exports.

This project now uses dataset-specific files like `common_celeba_vfl.py` and
local runners such as `run_vfl.py --dataset celeba`.
"""

from common_celeba_vfl import *  # noqa: F401,F403

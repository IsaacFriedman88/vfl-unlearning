"""Resolve local dataset and log paths for the VFL experiment.

Shared between Flower/ and SecretFlow/. Lives as a sibling of the run
scripts so a plain ``import vfl_paths`` works from either folder.

Resolution order (first match wins):
    1. Explicit argument passed into ``resolve_data_root(...)`` etc.
    2. Environment variables:
           VFL_DATA_ROOT, VFL_LOGS_ROOT,
           VFL_CELEBA_ROOT, VFL_MNIST_ROOT, VFL_NIH_ROOT
    3. Entries in ``config.yaml`` found by walking up from this file.
    4. ``DEFAULT_DATA_ROOT`` / ``DEFAULT_LOGS_ROOT`` below.

The YAML parser intentionally only understands flat ``key: value`` files
so we don't need PyYAML as a dependency.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _looks_absolute(raw: str) -> bool:
    """True if *raw* is absolute on either POSIX or Windows.

    ``Path("C:\\\\691\\\\data").is_absolute()`` returns False on Linux
    because backslashes aren't path separators there, so we recognize
    Windows-style drive prefixes explicitly and skip the relative-path
    fixup for them.
    """
    if _WINDOWS_ABS_RE.match(raw):
        return True
    if raw.startswith(("\\\\", "//")):  # UNC / network share
        return True
    return Path(raw).is_absolute()


DEFAULT_DATA_ROOT = r"C:\691\data"
DEFAULT_LOGS_ROOT = r"C:\691\logs"

# Subfolder names we expect under data_root.
CELEBA_SUBDIR = "celeba"
MNIST_SUBDIR = "MNIST"
NIH_SUBDIR = "nih_chest_xray14"

_CONFIG_FILENAME = "config.yaml"


@dataclass(frozen=True)
class ResolvedPaths:
    data_root: Path
    logs_root: Path
    celeba_root: Path
    mnist_root: Path
    nih_root: Path
    config_path: Path | None


# --------------------------------------------------------------------- cache

_cached: ResolvedPaths | None = None


def resolve() -> ResolvedPaths:
    """Return the resolved paths. Cached for cheap repeat access."""
    global _cached
    if _cached is None:
        _cached = _resolve_uncached()
    return _cached


def _resolve_uncached() -> ResolvedPaths:
    config_path, cfg = _load_config()

    data_root = _pick(
        env="VFL_DATA_ROOT",
        cfg_value=cfg.get("data_root"),
        default=DEFAULT_DATA_ROOT,
        base=config_path,
    )
    logs_root = _pick(
        env="VFL_LOGS_ROOT",
        cfg_value=cfg.get("logs_root"),
        default=DEFAULT_LOGS_ROOT,
        base=config_path,
    )
    celeba_root = _pick(
        env="VFL_CELEBA_ROOT",
        cfg_value=cfg.get("celeba_root"),
        default=str(data_root / CELEBA_SUBDIR),
        base=config_path,
    )
    mnist_root = _pick(
        env="VFL_MNIST_ROOT",
        cfg_value=cfg.get("mnist_root"),
        default=str(data_root / MNIST_SUBDIR),
        base=config_path,
    )
    nih_root = _pick(
        env="VFL_NIH_ROOT",
        cfg_value=cfg.get("nih_root"),
        default=str(data_root / NIH_SUBDIR),
        base=config_path,
    )

    return ResolvedPaths(
        data_root=data_root,
        logs_root=logs_root,
        celeba_root=celeba_root,
        mnist_root=mnist_root,
        nih_root=nih_root,
        config_path=config_path,
    )


# ---------------------------------------------------------- convenience API


def resolve_data_root(explicit: str | os.PathLike | None = None) -> Path:
    return Path(explicit) if explicit else resolve().data_root


def resolve_logs_root(explicit: str | os.PathLike | None = None) -> Path:
    return Path(explicit) if explicit else resolve().logs_root


def resolve_celeba_root(explicit: str | os.PathLike | None = None) -> Path:
    return Path(explicit) if explicit else resolve().celeba_root


def resolve_mnist_root(explicit: str | os.PathLike | None = None) -> Path:
    return Path(explicit) if explicit else resolve().mnist_root


def resolve_nih_root(explicit: str | os.PathLike | None = None) -> Path:
    return Path(explicit) if explicit else resolve().nih_root


# ------------------------------------------------------------ internals ----


def _pick(
    *,
    env: str,
    cfg_value: str | None,
    default: str,
    base: Path | None,
) -> Path:
    """Pick env > config > default and return a Path.

    Absolute paths (POSIX or Windows-style) are returned as-is.
    Relative paths are resolved against the directory containing
    ``config.yaml`` so ``data_root: ./data`` Does The Right Thing from
    either the Flower/ or SecretFlow/ subfolder.
    """
    raw = str(os.environ.get(env) or cfg_value or default).strip().strip('"').strip("'")
    if _looks_absolute(raw):
        return Path(raw).expanduser()
    if base is not None:
        return (base.parent / raw).expanduser().resolve()
    return Path(raw).expanduser().resolve()


def _load_config() -> tuple[Path | None, Mapping[str, str]]:
    """Walk up from this file looking for config.yaml.

    Returns (path_or_None, parsed_dict). If no config file is found we
    return an empty dict and callers fall back to defaults.
    """
    start = Path(__file__).resolve().parent
    for candidate in (start, *start.parents):
        cfg_path = candidate / _CONFIG_FILENAME
        if cfg_path.is_file():
            return cfg_path, _parse_flat_yaml(cfg_path)
    return None, {}


def _parse_flat_yaml(path: Path) -> dict[str, str]:
    """Minimal YAML-ish parser: ``key: value`` one per line, ``#`` comments.

    Good enough for our config file and keeps us free of a PyYAML dep.
    Supports quoted values (single or double), bare values, and Windows
    paths with backslashes.
    """
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        # Strip inline comments, but only when unquoted.
        value = _strip_inline_comment(value).strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if value:
            result[key] = value
    return result


def _strip_inline_comment(value: str) -> str:
    # Don't split on '#' if it appears inside quotes.
    in_single = False
    in_double = False
    for i, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return value[:i]
    return value


# ---------------------------------------------------------------- manual test

if __name__ == "__main__":
    paths = resolve()
    print(f"config_path: {paths.config_path}")
    print(f"data_root:   {paths.data_root}")
    print(f"logs_root:   {paths.logs_root}")
    print(f"celeba:      {paths.celeba_root}")
    print(f"mnist:       {paths.mnist_root}")
    print(f"nih:         {paths.nih_root}")

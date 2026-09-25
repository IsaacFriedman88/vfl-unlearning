from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import vfl_paths

# Logs land wherever vfl_paths resolves logs_root to. This is configurable
# via config.yaml or VFL_LOGS_ROOT; default is C:\691\logs so training
# artifacts don't churn through OneDrive sync on every epoch. We keep a
# framework-scoped subfolder (flower/ or secretflow/) so runs from the two
# stacks don't collide in a single run_summary.csv.
_FRAMEWORK_TAG = Path(__file__).resolve().parent.name.lower()
LOGS_DIR = vfl_paths.resolve().logs_root / _FRAMEWORK_TAG


@dataclass(frozen=True)
class RunLogPaths:
    run_id: str
    run_dir: Path
    metrics_csv: Path
    summary_csv: Path
    latest_csv: Path


def create_run_log_paths(dataset: str) -> RunLogPaths:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_dir = LOGS_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{dataset}_{timestamp}"
    run_dir = dataset_dir / run_id
    run_dir.mkdir(exist_ok=True)

    return RunLogPaths(
        run_id=run_id,
        run_dir=run_dir,
        metrics_csv=run_dir / "metrics.csv",
        summary_csv=LOGS_DIR / "run_summary.csv",
        latest_csv=dataset_dir / "latest.csv",
    )


def init_metrics_log(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_latest_pointer(latest_csv: Path, run_id: str, metrics_csv: Path) -> None:
    fieldnames = ["run_id", "metrics_csv", "updated_at"]
    row = {
        "run_id": run_id,
        "metrics_csv": str(metrics_csv),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with latest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


# Fields copied out of a (snapshot ∪ attack) dict into the summary row,
# under ``baseline_``/``final_`` prefixes and as ``delta_`` rows.
_SUMMARY_SCALAR_FIELDS = (
    "train_loss",
    "train_acc",
    "retain_loss",
    "retain_acc",
    "forget_loss",
    "forget_acc",
    "test_loss",
    "test_acc",
    "attack_accuracy",
    "attack_auc",
    "attack_f1",
    "attack_precision",
    "attack_recall",
    "attack_threshold",
)


def _summary_fieldnames() -> list[str]:
    """Column order for run_summary.csv.

    ``final_train_loss``, ``final_train_acc``, ``final_test_loss`` and
    ``final_test_acc`` are the same columns older callers set directly;
    they're emitted through the ``final_`` prefix over
    ``_SUMMARY_SCALAR_FIELDS``, so any script that read those four
    columns before will keep working.
    """
    baseline_cols = [f"baseline_{f}" for f in _SUMMARY_SCALAR_FIELDS]
    final_cols = [f"final_{f}" for f in _SUMMARY_SCALAR_FIELDS]
    delta_cols = [f"delta_{f}" for f in _SUMMARY_SCALAR_FIELDS]
    return [
        "timestamp",
        "run_id",
        "dataset",
        "epochs",
        "unlearn_epochs",
        "repair_epochs",
        "forget_fraction",
        "batch_size",
        "device",
        *baseline_cols,
        *final_cols,
        *delta_cols,
        "metrics_csv",
        "notes",
    ]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _prefixed_metrics(
    metrics: Mapping[str, float] | None, prefix: str
) -> dict[str, str]:
    row: dict[str, str] = {}
    for field in _SUMMARY_SCALAR_FIELDS:
        row[f"{prefix}{field}"] = _fmt(metrics.get(field)) if metrics else ""
    return row


def _deltas(
    baseline: Mapping[str, float] | None, final: Mapping[str, float] | None
) -> dict[str, str]:
    row: dict[str, str] = {}
    for field in _SUMMARY_SCALAR_FIELDS:
        if baseline is None or final is None:
            row[f"delta_{field}"] = ""
            continue
        b = baseline.get(field)
        f = final.get(field)
        if b is None or f is None:
            row[f"delta_{field}"] = ""
            continue
        try:
            row[f"delta_{field}"] = f"{float(f) - float(b):+.6f}"
        except (TypeError, ValueError):
            row[f"delta_{field}"] = ""
    return row


def append_run_summary(
    summary_csv: Path,
    *,
    run_id: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    device: str,
    final_train_loss: float,
    final_train_acc: float,
    final_test_loss: float,
    final_test_acc: float,
    metrics_csv: Path,
    notes: str = "",
    baseline_metrics: Mapping[str, float] | None = None,
    unlearn_metrics: Mapping[str, float] | None = None,
    unlearn_epochs: int | None = None,
    repair_epochs: int | None = None,
    forget_fraction: float | None = None,
) -> None:
    """Append a one-line summary of a VFL run.

    Callers pass:
      * ``baseline_metrics`` — a dict combining the last baseline-epoch
        snapshot (train/retain/forget/test loss+acc) with that epoch's
        attack metrics (accuracy=ASR, auc, f1, ...). Typically built via
        ``{**snapshot.as_dict(), **attack.as_dict()}``.
      * ``unlearn_metrics`` — same shape, for the last unlearn epoch.

    When both are given, the summary row includes ``baseline_*``,
    ``final_*``, and ``delta_*`` columns so you can eyeball the
    pre/post-unlearning delta (in particular ``delta_attack_accuracy``
    for ASR change) without re-opening metrics.csv.

    The legacy positional-ish kwargs (``final_train_loss`` etc.) are kept
    so older call sites keep working; if ``unlearn_metrics`` is provided
    it takes precedence.
    """
    row: dict[str, str] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "dataset": dataset,
        "epochs": str(epochs),
        "unlearn_epochs": "" if unlearn_epochs is None else str(unlearn_epochs),
        "repair_epochs": "" if repair_epochs is None else str(repair_epochs),
        "forget_fraction": "" if forget_fraction is None else f"{forget_fraction:.6f}",
        "batch_size": str(batch_size),
        "device": device,
        "metrics_csv": str(metrics_csv),
        "notes": notes,
    }

    # Legacy columns: prefer values from unlearn_metrics if present so
    # new callers stay consistent, else fall back to the positional args.
    leg_train_loss = (unlearn_metrics or {}).get("train_loss", final_train_loss)
    leg_train_acc = (unlearn_metrics or {}).get("train_acc", final_train_acc)
    leg_test_loss = (unlearn_metrics or {}).get("test_loss", final_test_loss)
    leg_test_acc = (unlearn_metrics or {}).get("test_acc", final_test_acc)
    row["final_train_loss"] = _fmt(leg_train_loss)
    row["final_train_acc"] = _fmt(leg_train_acc)
    row["final_test_loss"] = _fmt(leg_test_loss)
    row["final_test_acc"] = _fmt(leg_test_acc)

    row.update(_prefixed_metrics(baseline_metrics, "baseline_"))
    row.update(_prefixed_metrics(unlearn_metrics, "final_"))
    row.update(_deltas(baseline_metrics, unlearn_metrics))

    append_csv_row(summary_csv, _summary_fieldnames(), row)

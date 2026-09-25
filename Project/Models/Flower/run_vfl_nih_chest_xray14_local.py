from __future__ import annotations

import argparse

import torch

from common_nih_chest_xray14_vfl import (
    CFG,
    PartialImageEncoder,
    ServerBinaryClassifier,
    get_nih_loaders,
)
from run_logging import append_csv_row, append_run_summary, create_run_log_paths, init_metrics_log, write_latest_pointer
from vfl_unlearning import (
    build_unlearning_loaders,
    collect_attack_metrics,
    collect_metric_snapshot,
    create_optimizer,
    seed_everything,
    train_epoch_with_loss_scale,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local two-party VFL experiment on NIH Chest X-ray14.")
    parser.add_argument("--data-root", default=CFG.DATA_ROOT)
    parser.add_argument("--task-label", default=CFG.TASK_LABEL)
    parser.add_argument("--epochs", type=int, default=CFG.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=CFG.BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=CFG.RANDOM_STATE)
    parser.add_argument("--forget-fraction", type=float, default=0.1)
    parser.add_argument("--unlearn-epochs", type=int, default=1)
    parser.add_argument("--repair-epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_nih_loaders(
        root=args.data_root,
        task_label=args.task_label,
        batch_size=args.batch_size,
        test_batch_size=CFG.TEST_BATCH_SIZE,
    )
    loaders = build_unlearning_loaders(
        train_loader.dataset,
        batch_size=args.batch_size,
        forget_fraction=args.forget_fraction,
        seed=args.seed,
    )
    log_paths = create_run_log_paths("nih")
    metric_fields = [
        "phase",
        "epoch",
        "active_loss",
        "active_acc",
        "train_loss",
        "train_acc",
        "retain_loss",
        "retain_acc",
        "forget_loss",
        "forget_acc",
        "test_loss",
        "test_acc",
        "attack_accuracy",
        "attack_precision",
        "attack_recall",
        "attack_f1",
        "attack_auc",
        "attack_threshold",
        "member_mean_loss",
        "nonmember_mean_loss",
        "member_mean_confidence",
        "nonmember_mean_confidence",
    ]
    init_metrics_log(log_paths.metrics_csv, metric_fields)

    left_party = PartialImageEncoder().to(device)
    right_party = PartialImageEncoder().to(device)
    server_head = ServerBinaryClassifier().to(device)

    optimizer = create_optimizer(left_party, right_party, server_head, lr=CFG.LR)

    print(f"Running NIH Chest X-ray14 VFL locally on {device} with task '{args.task_label}'.")
    print(
        f"Seed={args.seed} | forget_fraction={args.forget_fraction:.3f} | "
        f"retain_samples={loaders.retain_count} | forget_samples={loaders.forget_count}"
    )
    print(f"Logging run to {log_paths.metrics_csv}")
    final_snapshot = None
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch_with_loss_scale(
            left_party=left_party,
            right_party=right_party,
            server_head=server_head,
            loader=loaders.train_loader,
            optimizer=optimizer,
            device=device,
            task_type="binary",
            max_batches=args.max_train_batches,
        )
        snapshot = collect_metric_snapshot(
            left_party=left_party,
            right_party=right_party,
            server_head=server_head,
            loaders=loaders,
            test_loader=test_loader,
            device=device,
            task_type="binary",
            max_eval_batches=args.max_test_batches,
        )
        attack_metrics = collect_attack_metrics(
            left_party=left_party,
            right_party=right_party,
            server_head=server_head,
            member_loader=loaders.forget_eval_loader,
            nonmember_loader=test_loader,
            device=device,
            task_type="binary",
            max_eval_batches=args.max_test_batches,
        )
        print(
            f"Baseline {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"forget_acc={snapshot.forget_acc:.4f} | "
            f"retain_acc={snapshot.retain_acc:.4f} | "
            f"test_acc={snapshot.test_acc:.4f} | "
            f"attack_acc={attack_metrics.attack_accuracy:.4f} attack_auc={attack_metrics.attack_auc:.4f}"
        )
        row = {
            "phase": "baseline",
            "epoch": epoch,
            "active_loss": f"{train_loss:.6f}",
            "active_acc": f"{train_acc:.6f}",
            "train_loss": f"{snapshot.train_loss:.6f}",
            "train_acc": f"{snapshot.train_acc:.6f}",
            "retain_loss": f"{snapshot.retain_loss:.6f}",
            "retain_acc": f"{snapshot.retain_acc:.6f}",
            "forget_loss": f"{snapshot.forget_loss:.6f}",
            "forget_acc": f"{snapshot.forget_acc:.6f}",
            "test_loss": f"{snapshot.test_loss:.6f}",
            "test_acc": f"{snapshot.test_acc:.6f}",
        }
        row.update({k: f"{v:.6f}" for k, v in attack_metrics.as_dict().items()})
        append_csv_row(
            log_paths.metrics_csv,
            metric_fields,
            row,
        )
        final_snapshot = snapshot
        final_attack_metrics = attack_metrics

    if final_snapshot is None:
        raise RuntimeError("Baseline training did not run any epochs.")

    # Freeze the pre-unlearn state so the summary row can show the ASR /
    # accuracy before AND after the unlearning pass.
    baseline_final_snapshot = final_snapshot
    baseline_final_attack_metrics = final_attack_metrics

    print(
        f"Pre-unlearning snapshot | forget_loss={baseline_final_snapshot.forget_loss:.4f} "
        f"forget_acc={baseline_final_snapshot.forget_acc:.4f} | "
        f"retain_acc={baseline_final_snapshot.retain_acc:.4f} | "
        f"test_acc={baseline_final_snapshot.test_acc:.4f} | "
        f"attack_acc={baseline_final_attack_metrics.attack_accuracy:.4f} "
        f"attack_auc={baseline_final_attack_metrics.attack_auc:.4f}"
    )

    for epoch in range(1, args.unlearn_epochs + 1):
        forget_loss, forget_acc = train_epoch_with_loss_scale(
            left_party=left_party,
            right_party=right_party,
            server_head=server_head,
            loader=loaders.forget_train_loader,
            optimizer=optimizer,
            device=device,
            task_type="binary",
            loss_scale=-1.0,
            max_batches=args.max_train_batches,
        )
        for _ in range(args.repair_epochs):
            train_epoch_with_loss_scale(
                left_party=left_party,
                right_party=right_party,
                server_head=server_head,
                loader=loaders.retain_train_loader,
                optimizer=optimizer,
                device=device,
                task_type="binary",
                loss_scale=1.0,
                max_batches=args.max_train_batches,
            )

        snapshot = collect_metric_snapshot(
            left_party=left_party,
            right_party=right_party,
            server_head=server_head,
            loaders=loaders,
            test_loader=test_loader,
            device=device,
            task_type="binary",
            max_eval_batches=args.max_test_batches,
        )
        attack_metrics = collect_attack_metrics(
            left_party=left_party,
            right_party=right_party,
            server_head=server_head,
            member_loader=loaders.forget_eval_loader,
            nonmember_loader=test_loader,
            device=device,
            task_type="binary",
            max_eval_batches=args.max_test_batches,
        )
        print(
            f"Unlearn {epoch}/{args.unlearn_epochs} | forget_loss={forget_loss:.4f} "
            f"forget_acc={forget_acc:.4f} | "
            f"retain_acc={snapshot.retain_acc:.4f} | "
            f"test_acc={snapshot.test_acc:.4f} | "
            f"attack_acc={attack_metrics.attack_accuracy:.4f} attack_auc={attack_metrics.attack_auc:.4f}"
        )
        row = {
            "phase": "unlearn",
            "epoch": epoch,
            "active_loss": f"{forget_loss:.6f}",
            "active_acc": f"{forget_acc:.6f}",
            "train_loss": f"{snapshot.train_loss:.6f}",
            "train_acc": f"{snapshot.train_acc:.6f}",
            "retain_loss": f"{snapshot.retain_loss:.6f}",
            "retain_acc": f"{snapshot.retain_acc:.6f}",
            "forget_loss": f"{snapshot.forget_loss:.6f}",
            "forget_acc": f"{snapshot.forget_acc:.6f}",
            "test_loss": f"{snapshot.test_loss:.6f}",
            "test_acc": f"{snapshot.test_acc:.6f}",
        }
        row.update({k: f"{v:.6f}" for k, v in attack_metrics.as_dict().items()})
        append_csv_row(
            log_paths.metrics_csv,
            metric_fields,
            row,
        )
        final_snapshot = snapshot
        final_attack_metrics = attack_metrics

    append_run_summary(
        log_paths.summary_csv,
        run_id=log_paths.run_id,
        dataset="nih",
        epochs=args.epochs,
        unlearn_epochs=args.unlearn_epochs,
        repair_epochs=args.repair_epochs,
        forget_fraction=args.forget_fraction,
        batch_size=args.batch_size,
        device=str(device),
        final_train_loss=final_snapshot.train_loss,
        final_train_acc=final_snapshot.train_acc,
        final_test_loss=final_snapshot.test_loss,
        final_test_acc=final_snapshot.test_acc,
        baseline_metrics={
            **baseline_final_snapshot.as_dict(),
            **baseline_final_attack_metrics.as_dict(),
        },
        unlearn_metrics={
            **final_snapshot.as_dict(),
            **final_attack_metrics.as_dict(),
        },
        metrics_csv=log_paths.metrics_csv,
        notes=(
            f"Task={args.task_label}, two-party vertical split: left_half vs right_half; "
            f"seed={args.seed}"
        ),
    )
    write_latest_pointer(log_paths.latest_csv, log_paths.run_id, log_paths.metrics_csv)
    print(f"Saved metrics to {log_paths.metrics_csv}")


if __name__ == "__main__":
    main()

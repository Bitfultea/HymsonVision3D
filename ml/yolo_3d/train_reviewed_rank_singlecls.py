#!/usr/bin/env python3
"""Train reviewed rank dataset with the fixed single-class baseline recipe."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from controlled_experiments import metric, yolov114_config


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "ml" / "yolo_3d"
DEFAULT_DATA = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/dataset_rank_reviewed_20260623/data.yaml"
)
RESULTS_ROOT = (
    SCRIPT_DIR / "experiment_results" / "rank_reviewed_singlecls_20260623"
)
YOLO11S = SCRIPT_DIR / "yolo11s-seg.pt"
BASELINE_GROUP = (
    SCRIPT_DIR
    / "experiment_results"
    / "singlecls_seed_repeat_20260622"
    / "group_summary.csv"
)


def read_baseline_mean() -> float | None:
    if not BASELINE_GROUP.exists():
        return None
    with BASELINE_GROUP.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or not rows[0].get("mean"):
        return None
    return float(rows[0]["mean"])


def check_inputs(data: Path) -> None:
    if not data.exists():
        raise FileNotFoundError(data)
    if not YOLO11S.exists():
        raise FileNotFoundError(YOLO11S)


def train_one(
    data: Path,
    runs_dir: Path,
    name_prefix: str,
    seed: int,
    epochs: int,
    device: str,
) -> Path:
    check_inputs(data)
    name = f"{name_prefix}_seed{seed}"
    run_dir = runs_dir / name
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        print(f"skip existing train: {name}", flush=True)
        return run_dir

    config = yolov114_config(epochs, single_cls=True)
    print(f"training {name}", flush=True)
    model = YOLO(str(YOLO11S))
    model.train(
        data=str(data),
        device=device,
        project=str(runs_dir),
        name=name,
        exist_ok=True,
        deterministic=True,
        seed=seed,
        augment=True,
        **config,
    )
    return run_dir


def validate_one(
    data: Path,
    val_dir: Path,
    runs_dir: Path,
    name_prefix: str,
    seed: int,
    epochs: int,
    device: str,
) -> dict[str, Any]:
    name = f"{name_prefix}_seed{seed}"
    run_dir = runs_dir / name
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)

    config = yolov114_config(epochs, single_cls=True)
    print(f"validating {name}", flush=True)
    model = YOLO(str(weights))
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(data),
        imgsz=int(config["imgsz"]),
        batch=int(config["batch"]),
        device=device,
        workers=int(config["workers"]),
        single_cls=True,
        plots=False,
        save_json=False,
        project=str(val_dir),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return {
        "seed": seed,
        "name": name,
        "mask_map50_95": metric(metrics.seg, "map"),
        "mask_map50": metric(metrics.seg, "map50"),
        "mask_precision": metric(metrics.seg, "mp"),
        "mask_recall": metric(metrics.seg, "mr"),
        "box_map50_95": metric(metrics.box, "map"),
        "box_map50": metric(metrics.box, "map50"),
        "box_precision": metric(metrics.box, "mp"),
        "box_recall": metric(metrics.box, "mr"),
        "single_cls": True,
        "model": str(YOLO11S),
        "data": str(data),
        "epochs": epochs,
        "elapsed_s": round(elapsed, 3),
        "run_dir": str(run_dir),
        "weights": str(weights),
    }


def load_existing_rows(summary_csv: Path) -> dict[int, dict[str, Any]]:
    if not summary_csv.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with summary_csv.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("seed"):
                rows[int(row["seed"])] = row
    return rows


def write_summary(summary_csv: Path, group_csv: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: float(row["mask_map50_95"]), reverse=True)
    fields = [
        "rank",
        "seed",
        "name",
        "mask_map50_95",
        "mask_map50",
        "mask_precision",
        "mask_recall",
        "box_map50_95",
        "box_map50",
        "box_precision",
        "box_recall",
        "single_cls",
        "model",
        "data",
        "epochs",
        "elapsed_s",
        "run_dir",
        "weights",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    values = [float(row["mask_map50_95"]) for row in rows]
    with group_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["n", "mean", "std", "best", "worst"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "n": len(values),
                "mean": statistics.mean(values) if values else "",
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "best": max(values) if values else "",
                "worst": min(values) if values else "",
            }
        )


def write_analysis(results_root: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row["mask_map50_95"]), reverse=True)
    values = [float(row["mask_map50_95"]) for row in rows]
    mean_value = statistics.mean(values)
    std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
    baseline_mean = read_baseline_mean()

    lines = [
        "# Reviewed Rank Single-Class Training",
        "",
        "Recipe: `ia_rank_residual_intensity_gradient` images + reviewed labels + "
        "`yolo11s-seg` + yolov114-style augmentation + `single_cls=True`.",
        "",
        "| Rank | Seed | Mask mAP50-95 | Mask mAP50 | Box mAP50-95 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {seed} | {mask:.6f} | {mask50:.6f} | {box:.6f} |".format(
                rank=rank,
                seed=row["seed"],
                mask=float(row["mask_map50_95"]),
                mask50=float(row["mask_map50"]),
                box=float(row["box_map50_95"]),
            )
        )

    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- n: {len(values)}",
            f"- mean mask mAP50-95: {mean_value:.6f}",
            f"- std mask mAP50-95: {std_value:.6f}",
            f"- best mask mAP50-95: {max(values):.6f}",
            f"- worst mask mAP50-95: {min(values):.6f}",
        ]
    )
    if baseline_mean is not None:
        lines.extend(
            [
                "",
                "## Label-Cleaning Delta",
                "",
                f"- previous rank single-class mean: {baseline_mean:.6f}",
                f"- reviewed-label mean delta: {mean_value - baseline_mean:+.6f}",
            ]
        )
    lines.append("")
    (results_root / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--name-prefix", default="rank_reviewed_yolov114_y11s_singlecls")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_inputs(args.data)
    runs_dir = args.results_root / "runs"
    val_dir = args.results_root / "val_runs"
    summary_csv = args.results_root / "summary.csv"
    group_csv = args.results_root / "group_summary.csv"
    args.results_root.mkdir(parents=True, exist_ok=True)

    rows_by_seed = load_existing_rows(summary_csv)
    for seed in args.seeds:
        if not args.validate_only:
            train_one(args.data, runs_dir, args.name_prefix, seed, args.epochs, args.device)
        if not args.train_only:
            rows_by_seed[seed] = validate_one(
                args.data,
                val_dir,
                runs_dir,
                args.name_prefix,
                seed,
                args.epochs,
                args.device,
            )
            rows = list(rows_by_seed.values())
            write_summary(summary_csv, group_csv, rows)
            write_analysis(args.results_root, rows)

    if not args.train_only:
        rows = list(rows_by_seed.values())
        write_summary(summary_csv, group_csv, rows)
        write_analysis(args.results_root, rows)
        print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Repeat the historical yolov114 recipe across seeds."""

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
RESULTS_ROOT = (
    SCRIPT_DIR / "experiment_results" / "historical_yolov114_seed_repeat_20260622"
)
RUNS_DIR = RESULTS_ROOT / "runs"
VAL_DIR = RESULTS_ROOT / "val_runs"
SUMMARY_CSV = RESULTS_ROOT / "summary.csv"
GROUP_CSV = RESULTS_ROOT / "group_summary.csv"

OLD_DATA = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/dataset/data.yaml"
)
YOLO11S = SCRIPT_DIR / "yolo11s-seg.pt"


def check_inputs() -> None:
    if not OLD_DATA.exists():
        raise FileNotFoundError(OLD_DATA)
    if not YOLO11S.exists():
        raise FileNotFoundError(YOLO11S)


def train_one(seed: int, epochs: int, device: str) -> Path:
    check_inputs()
    name = f"historical_yolov114_seed{seed}"
    run_dir = RUNS_DIR / name
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        print(f"skip existing train: {name}", flush=True)
        return run_dir

    config = yolov114_config(epochs, single_cls=True)
    print(f"training {name}", flush=True)
    model = YOLO(str(YOLO11S))
    model.train(
        data=str(OLD_DATA),
        device=device,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=True,
        deterministic=True,
        seed=seed,
        augment=True,
        **config,
    )
    return run_dir


def validate_one(seed: int, epochs: int, device: str) -> dict[str, Any]:
    name = f"historical_yolov114_seed{seed}"
    run_dir = RUNS_DIR / name
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)

    config = yolov114_config(epochs, single_cls=True)
    print(f"validating {name}", flush=True)
    model = YOLO(str(weights))
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(OLD_DATA),
        imgsz=int(config["imgsz"]),
        batch=int(config["batch"]),
        device=device,
        workers=int(config["workers"]),
        single_cls=True,
        plots=False,
        save_json=False,
        project=str(VAL_DIR),
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
        "data": str(OLD_DATA),
        "epochs": epochs,
        "elapsed_s": round(elapsed, 3),
        "run_dir": str(RUNS_DIR / name),
        "weights": str(weights),
    }


def load_existing_rows() -> dict[int, dict[str, Any]]:
    if not SUMMARY_CSV.exists():
        return {}
    rows: dict[int, dict[str, Any]] = {}
    with SUMMARY_CSV.open() as file:
        for row in csv.DictReader(file):
            if row.get("seed"):
                rows[int(row["seed"])] = row
    return rows


def write_summary(rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: float(row["mask_map50_95"]), reverse=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
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
    with SUMMARY_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    values = [float(row["mask_map50_95"]) for row in rows]
    with GROUP_CSV.open("w", newline="") as file:
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


def write_analysis(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row["mask_map50_95"]), reverse=True)
    values = [float(row["mask_map50_95"]) for row in rows]
    mean_value = statistics.mean(values)
    std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
    best_value = max(values)
    worst_value = min(values)
    historical_existing = 0.532118
    rank_single_mean = 0.520733
    rank_single_best = 0.536776
    strict_rank_best = 0.521098

    text = [
        "# Historical YOLOv114 Seed Repeat Analysis",
        "",
        "Date: 2026-06-22",
        "",
        "Recipe: original generated dataset + yolo11s-seg + historical "
        "`yolov114` augmentation + `single_cls=True`.",
        "",
        "| Rank | Seed | Mask mAP50-95 | Mask mAP50 | Box mAP50-95 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        text.append(
            "| {rank} | {seed} | {mask:.6f} | {mask50:.6f} | {box:.6f} |".format(
                rank=rank,
                seed=row["seed"],
                mask=float(row["mask_map50_95"]),
                mask50=float(row["mask_map50"]),
                box=float(row["box_map50_95"]),
            )
        )

    text.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- n: {len(values)}",
            f"- mean mask mAP50-95: {mean_value:.6f}",
            f"- std mask mAP50-95: {std_value:.6f}",
            f"- best mask mAP50-95: {best_value:.6f}",
            f"- worst mask mAP50-95: {worst_value:.6f}",
            "",
            "## Baselines",
            "",
            f"- existing historical `yolov114`: {historical_existing:.6f}",
            f"- rank single-class seed-repeat mean: {rank_single_mean:.6f}",
            f"- rank single-class seed-repeat best: {rank_single_best:.6f}",
            f"- best multi-class strict rank result: {strict_rank_best:.6f}",
            "",
            "## Deltas",
            "",
            f"- mean vs. existing historical `yolov114`: {mean_value - historical_existing:+.6f}",
            f"- mean vs. rank single-class mean: {mean_value - rank_single_mean:+.6f}",
            f"- best vs. rank single-class best: {best_value - rank_single_best:+.6f}",
            f"- mean vs. best multi-class strict rank: {mean_value - strict_rank_best:+.6f}",
            "",
            "## Conclusion",
            "",
            "Use the seed-repeat mean and variance here to decide whether the "
            "historical original-preprocessing recipe is actually more stable "
            "than the rank-preprocessing single-class candidate.",
            "",
        ]
    )
    (RESULTS_ROOT / "analysis.md").write_text("\n".join(text), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_inputs()
    torch.backends.cudnn.benchmark = True

    rows = load_existing_rows()
    for seed in args.seeds:
        train_one(seed, args.epochs, args.device)
        rows[seed] = validate_one(seed, args.epochs, args.device)
        current_rows = [rows[seed_value] for seed_value in sorted(rows)]
        write_summary(current_rows)
        write_analysis(current_rows)
        print(f"wrote {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Controlled training experiments for YOLO-3D preprocessing comparisons."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "ml" / "yolo_3d"
RESULTS_ROOT = (
    SCRIPT_DIR / "experiment_results" / "controlled_preprocess_train_20260617"
)
RUNS_DIR = RESULTS_ROOT / "runs"
VAL_DIR = RESULTS_ROOT / "val_runs"
SUMMARY_CSV = RESULTS_ROOT / "summary.csv"
GROUP_CSV = RESULTS_ROOT / "group_summary.csv"

OLD_DATA = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/dataset/data.yaml"
)
RANK_DATA = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/preprocess_v2_20260612/datasets/"
    "ia_rank_residual_intensity_gradient/data.yaml"
)

YOLO11M = SCRIPT_DIR / "yolo11m-seg.pt"
YOLO11S = SCRIPT_DIR / "yolo11s-seg.pt"
YOLOV8M = SCRIPT_DIR / "yolov8m-seg.pt"


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    data: Path
    model: Path
    config: dict[str, Any]


def strict_config(epochs: int) -> dict[str, Any]:
    return {
        "epochs": epochs,
        "imgsz": 960,
        "batch": 16,
        "workers": 4,
        "patience": 0,
        "cache": True,
        "single_cls": False,
        "degrees": 90.0,
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "close_mosaic": 20,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "dropout": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    }


def v14_config(epochs: int) -> dict[str, Any]:
    return {
        "epochs": epochs,
        "imgsz": 960,
        "batch": 16,
        "workers": 4,
        "patience": 0,
        "cache": False,
        "single_cls": False,
        "degrees": 90.0,
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "close_mosaic": 10,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "dropout": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
    }


def trainpy_config(epochs: int) -> dict[str, Any]:
    return {
        "epochs": epochs,
        "imgsz": 960,
        "batch": 16,
        "workers": 4,
        "patience": 0,
        "cache": True,
        "single_cls": False,
        "degrees": 180.0,
        "translate": 0.1,
        "scale": 0.8,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "close_mosaic": 50,
        "mixup": 0.0,
        "cutmix": 0.30,
        "copy_paste": 0.8,
        "dropout": 0.4,
    }


def yolov114_config(epochs: int, single_cls: bool) -> dict[str, Any]:
    return {
        "epochs": epochs,
        "imgsz": 960,
        "batch": 8,
        "workers": 4,
        "patience": 50,
        "cache": True,
        "single_cls": single_cls,
        "degrees": 180.0,
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.5,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "close_mosaic": 50,
        "mixup": 0.3,
        "copy_paste": 0.8,
        "dropout": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
    }


def experiments(epochs: int) -> list[Experiment]:
    return [
        Experiment(
            name="old_dataset_new_strict_y11m",
            group="old_data_new_strict",
            data=OLD_DATA,
            model=YOLO11M,
            config=strict_config(epochs),
        ),
        Experiment(
            name="rank_dataset_v14_yolov8m",
            group="rank_data_v14_recipe",
            data=RANK_DATA,
            model=YOLOV8M,
            config=v14_config(epochs),
        ),
        Experiment(
            name="old_dataset_trainpy_y11m",
            group="old_data_trainpy_recipe",
            data=OLD_DATA,
            model=YOLO11M,
            config=trainpy_config(epochs),
        ),
        Experiment(
            name="rank_dataset_yolov114_y11s_multiclass",
            group="rank_data_yolov114_recipe_multiclass",
            data=RANK_DATA,
            model=YOLO11S,
            config=yolov114_config(epochs, single_cls=False),
        ),
        Experiment(
            name="rank_dataset_yolov114_y11s_singlecls",
            group="rank_data_yolov114_recipe_singlecls",
            data=RANK_DATA,
            model=YOLO11S,
            config=yolov114_config(epochs, single_cls=True),
        ),
    ]


def check(exp: Experiment) -> None:
    if not exp.data.exists():
        raise FileNotFoundError(exp.data)
    if not exp.model.exists():
        raise FileNotFoundError(exp.model)


def train(exp: Experiment, device: str) -> Path:
    check(exp)
    run_dir = RUNS_DIR / exp.name
    if (run_dir / "weights" / "best.pt").exists():
        print(f"skip existing train: {exp.name}", flush=True)
        return run_dir

    print(f"training {exp.name}", flush=True)
    model = YOLO(str(exp.model))
    model.train(
        data=str(exp.data),
        device=device,
        project=str(RUNS_DIR),
        name=exp.name,
        exist_ok=True,
        deterministic=True,
        seed=0,
        augment=True,
        **exp.config,
    )
    return run_dir


def metric(metric_obj: Any, attr: str) -> float:
    value = getattr(metric_obj, attr)
    if callable(value):
        value = value()
    return float(value)


def validate(exp: Experiment, run_dir: Path, device: str) -> dict[str, Any]:
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)
    print(f"validating {exp.name}", flush=True)
    model = YOLO(str(weights))
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(exp.data),
        imgsz=int(exp.config["imgsz"]),
        batch=int(exp.config["batch"]),
        device=device,
        workers=int(exp.config["workers"]),
        single_cls=bool(exp.config["single_cls"]),
        plots=False,
        save_json=False,
        project=str(VAL_DIR),
        name=exp.name,
        exist_ok=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return {
        "name": exp.name,
        "group": exp.group,
        "mask_map50_95": metric(metrics.seg, "map"),
        "mask_map50": metric(metrics.seg, "map50"),
        "mask_precision": metric(metrics.seg, "mp"),
        "mask_recall": metric(metrics.seg, "mr"),
        "box_map50_95": metric(metrics.box, "map"),
        "box_map50": metric(metrics.box, "map50"),
        "box_precision": metric(metrics.box, "mp"),
        "box_recall": metric(metrics.box, "mr"),
        "single_cls": bool(exp.config["single_cls"]),
        "model": str(exp.model),
        "data": str(exp.data),
        "epochs": int(exp.config["epochs"]),
        "elapsed_s": round(elapsed, 3),
        "run_dir": str(run_dir),
        "weights": str(run_dir / "weights" / "best.pt"),
    }


def write_summary(rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: float(row["mask_map50_95"]), reverse=True)
    fields = [
        "rank",
        "name",
        "group",
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
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})

    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["group"]), []).append(float(row["mask_map50_95"]))
    with GROUP_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["group", "n", "mean", "std", "best"])
        writer.writeheader()
        for group, values in sorted(grouped.items()):
            writer.writerow(
                {
                    "group": group,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                    "best": max(values),
                }
            )


def load_existing_rows() -> list[dict[str, Any]]:
    if not SUMMARY_CSV.exists():
        return []
    with SUMMARY_CSV.open() as file:
        return [
            {key: value for key, value in row.items() if key != "rank"}
            for row in csv.DictReader(file)
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    selected = experiments(args.epochs)
    if args.experiments:
        allowed = set(args.experiments)
        selected = [exp for exp in selected if exp.name in allowed]
        unknown = allowed - {exp.name for exp in selected}
        if unknown:
            raise ValueError(f"unknown experiments: {sorted(unknown)}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    rows_by_name = {row["name"]: row for row in load_existing_rows()}

    for exp in selected:
        run_dir = RUNS_DIR / exp.name
        if not args.validate_only:
            run_dir = train(exp, args.device)
        if not args.train_only:
            rows_by_name[exp.name] = validate(exp, run_dir, args.device)
            write_summary(list(rows_by_name.values()))

    if not args.train_only:
        write_summary(list(rows_by_name.values()))
        print(f"wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

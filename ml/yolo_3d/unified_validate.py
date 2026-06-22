#!/usr/bin/env python3
"""Run comparable Ultralytics validation for selected YOLO-3D experiments."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    REPO_ROOT / "ml" / "yolo_3d" / "experiment_results" / "unified_validation_20260617"
)
OUT_ROOT = RESULTS_ROOT / "val_runs"
SUMMARY_CSV = RESULTS_ROOT / "summary.csv"

OLD_DATA = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/dataset/data.yaml"
)
FOLLOWUP_DATA = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/preprocess_followup_20260611/datasets/"
    "invalid_aware_intensity_residual_gradient/data.yaml"
)
LOG_DATA = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/preprocess_v2_20260612/datasets/"
    "ia_residual_intensity_log/data.yaml"
)
RANK_DATA = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/preprocess_v2_20260612/datasets/"
    "ia_rank_residual_intensity_gradient/data.yaml"
)
HYB_DOG_DATA = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/next_experiments_20260615/hybrid_preprocess/datasets/"
    "hyb_dog_int_grad/data.yaml"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    group: str
    weights: Path
    data: Path
    imgsz: int = 960


def repo_weight(run_name: str) -> Path:
    return REPO_ROOT / "ml" / "yolo_3d" / "runs" / "segment" / run_name / "weights" / "best.pt"


def strict_weight(run_name: str) -> Path:
    return Path(
        "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
        f"yolo_3d/next_experiments_20260615/runs/strict_repeat/{run_name}/weights/best.pt"
    )


def default_candidates() -> list[Candidate]:
    old_top = [
        "defect_seg_v12",
        "yolov114",
        "defect_seg_v14",
        "defect_seg_med_v24",
        "defect_seg_med_v27",
        "defect_seg_v2",
        "yolov1151",
        "defect_seg_med_v23",
        "defect_seg_med_v25",
        "defect_seg_med_v14",
    ]
    candidates = [
        Candidate(name=run, group="old_runs_segment", weights=repo_weight(run), data=OLD_DATA)
        for run in old_top
    ]
    candidates.extend(
        [
            Candidate(
                name="log_seed0",
                group="strict_log",
                weights=strict_weight("log_seed0"),
                data=LOG_DATA,
            ),
            Candidate(
                name="log_seed1",
                group="strict_log",
                weights=strict_weight("log_seed1"),
                data=LOG_DATA,
            ),
            Candidate(
                name="log_seed2",
                group="strict_log",
                weights=strict_weight("log_seed2"),
                data=LOG_DATA,
            ),
            Candidate(
                name="invalid_aware_seed0",
                group="strict_invalid_aware",
                weights=strict_weight("invalid_aware_seed0"),
                data=FOLLOWUP_DATA,
            ),
            Candidate(
                name="invalid_aware_seed1",
                group="strict_invalid_aware",
                weights=strict_weight("invalid_aware_seed1"),
                data=FOLLOWUP_DATA,
            ),
            Candidate(
                name="invalid_aware_seed2",
                group="strict_invalid_aware",
                weights=strict_weight("invalid_aware_seed2"),
                data=FOLLOWUP_DATA,
            ),
            Candidate(
                name="rank_seed0",
                group="strict_rank",
                weights=strict_weight("rank_seed0"),
                data=RANK_DATA,
            ),
            Candidate(
                name="rank_seed1",
                group="strict_rank",
                weights=strict_weight("rank_seed1"),
                data=RANK_DATA,
            ),
            Candidate(
                name="rank_seed2",
                group="strict_rank",
                weights=strict_weight("rank_seed2"),
                data=RANK_DATA,
            ),
            Candidate(
                name="hyb_dog_int_grad_seed0",
                group="strict_hybrid",
                weights=strict_weight("hyb_dog_int_grad_seed0"),
                data=HYB_DOG_DATA,
            ),
        ]
    )
    return candidates


def check_candidates(candidates: Iterable[Candidate]) -> None:
    for candidate in candidates:
        if not candidate.weights.exists():
            raise FileNotFoundError(f"missing weights for {candidate.name}: {candidate.weights}")
        if not candidate.data.exists():
            raise FileNotFoundError(f"missing data yaml for {candidate.name}: {candidate.data}")


def metric_value(metric, attr: str) -> float:
    value = getattr(metric, attr)
    if callable(value):
        value = value()
    return float(value)


def validate(candidate: Candidate, device: str, batch: int, workers: int) -> dict[str, str | float | int]:
    print(f"validating {candidate.name} on {candidate.data}", flush=True)
    model = YOLO(str(candidate.weights))
    t0 = time.perf_counter()
    metrics = model.val(
        data=str(candidate.data),
        imgsz=candidate.imgsz,
        batch=batch,
        device=device,
        workers=workers,
        plots=False,
        save_json=False,
        project=str(OUT_ROOT),
        name=candidate.name,
        exist_ok=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return {
        "name": candidate.name,
        "group": candidate.group,
        "mask_map50_95": metric_value(metrics.seg, "map"),
        "mask_map50": metric_value(metrics.seg, "map50"),
        "mask_precision": metric_value(metrics.seg, "mp"),
        "mask_recall": metric_value(metrics.seg, "mr"),
        "box_map50_95": metric_value(metrics.box, "map"),
        "box_map50": metric_value(metrics.box, "map50"),
        "box_precision": metric_value(metrics.box, "mp"),
        "box_recall": metric_value(metrics.box, "mr"),
        "imgsz": candidate.imgsz,
        "elapsed_s": round(elapsed, 3),
        "weights": str(candidate.weights),
        "data": str(candidate.data),
    }


def write_summary(rows: list[dict[str, str | float | int]]) -> None:
    rows.sort(key=lambda row: float(row["mask_map50_95"]), reverse=True)
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
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
        "imgsz",
        "elapsed_s",
        "weights",
        "data",
    ]
    with SUMMARY_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    candidates = default_candidates()
    check_candidates(candidates)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidates:
        rows.append(validate(candidate, args.device, args.batch, args.workers))
        write_summary(rows)
    print(f"wrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()

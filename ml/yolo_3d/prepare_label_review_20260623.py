#!/usr/bin/env python3
"""Prepare label-review decision files for the rank-preprocessed dataset."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "ml" / "yolo_3d"
ERROR_ROOT = (
    SCRIPT_DIR / "experiment_results" / "singleclass_error_analysis_20260622"
)
REVIEW_QUEUE = ERROR_ROOT / "review_queue"
OUT_DIR = SCRIPT_DIR / "experiment_results" / "label_review_20260623"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def suggested_action(row: dict[str, str]) -> str:
    kind = row["kind"]
    reason = row["reason"]
    iou = float(row["best_mask_iou"])
    if kind == "fp" and reason == "high_conf_fp_check_unlabeled_defect":
        return "review_add_label_or_ignore_fp"
    if kind == "fn" and row["original_class"] == "pinhole":
        return "review_small_pinhole_visibility"
    if 0.35 <= iou < 0.50:
        return "review_boundary_or_offset"
    if kind == "fn":
        return "review_keep_adjust_or_delete_label"
    return "review_texture_edge_false_positive"


def decision_choices(row: dict[str, str]) -> str:
    if row["kind"] == "fp":
        return "add_label|ignore_fp|ambiguous"
    return "keep_label|adjust_boundary|delete_label|class_change|ambiguous"


def build_decision_rows(instance_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in instance_rows:
        rows.append(
            {
                "priority_rank": row["priority_rank"],
                "image_priority_rank": row["image_priority_rank"],
                "image": row["image"],
                "kind": row["kind"],
                "reason": row["reason"],
                "suggested_action": suggested_action(row),
                "decision_choices": decision_choices(row),
                "decision": "",
                "new_class": "",
                "label_needs_edit": "",
                "notes": "",
                "score": row["score"],
                "best_mask_iou": row["best_mask_iou"],
                "area": row["area"],
                "original_class": row["original_class"],
                "matched_original_class": row["matched_original_class"],
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
                "crop_path": str(REVIEW_QUEUE / row["crop_path"]),
                "image_path": row["image_path"],
                "label_path": row["label_path"],
            }
        )
    return rows


def write_report(
    path: Path,
    instance_rows: list[dict[str, str]],
    threshold_rows: list[dict[str, str]],
    class_rows: list[dict[str, str]],
) -> None:
    reason_counts = Counter(row["reason"] for row in instance_rows)
    kind_counts = Counter(row["kind"] for row in instance_rows)
    near_iou = [
        row for row in instance_rows if 0.35 <= float(row["best_mask_iou"]) < 0.50
    ]
    high_conf_fp = [
        row
        for row in instance_rows
        if row["kind"] == "fp" and float(row["score"]) >= 0.50
    ]
    pinhole_fn = [
        row
        for row in instance_rows
        if row["kind"] == "fn" and row["original_class"] == "pinhole"
    ]

    lines = [
        "# Label Review 20260623",
        "",
        "Scope: manual review preparation for `ia_rank_residual_intensity_gradient`.",
        "No source dataset files are modified by this step.",
        "",
        "## What To Confirm",
        "",
        "- High-confidence FP: decide whether each is an unlabeled real defect.",
        "- FN: decide whether each missed object is a valid small defect, especially pinhole.",
        "- Boundary/category issues: fix labels with undersized, shifted, or inconsistent polygons/classes.",
        "",
        "## Current Evidence",
        "",
        f"- queued instances: {len(instance_rows)}",
        f"- FN: {kind_counts.get('fn', 0)}",
        f"- FP: {kind_counts.get('fp', 0)}",
        f"- high-confidence FP score >= 0.50: {len(high_conf_fp)}",
        f"- pinhole FN: {len(pinhole_fn)}",
        f"- near-match boundary/offset candidates, 0.35 <= IoU < 0.50: {len(near_iou)}",
        "",
        "## Review Inputs",
        "",
        f"- source queue: `{REVIEW_QUEUE}`",
        f"- decision template: `{path.parent / 'review_decisions_template.csv'}`",
        f"- reviewed label drop-in dir: `{path.parent / 'reviewed_labels'}`",
        "",
        "## Review Threshold Summary",
        "",
        "| Conf | Precision50 | Recall50 | F1@50 | Pred/Image |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in threshold_rows:
        lines.append(
            f"| {float(row['threshold']):.3f} | "
            f"{float(row['mask_precision50']):.5f} | "
            f"{float(row['mask_recall50']):.5f} | "
            f"{float(row['mask_f1_50']):.5f} | "
            f"{float(row['predictions_per_image']):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Original-Class Recall At Conf 0.25",
            "",
            "| Class | GT | TP | FN | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in class_rows:
        lines.append(
            f"| `{row['original_class']}` | {row['gt']} | {row['tp']} | "
            f"{row['fn']} | {float(row['recall_at_review_conf']):.5f} |"
        )

    lines.extend(
        [
            "",
            "## Reason Counts",
            "",
        ]
    )
    for reason, count in reason_counts.most_common():
        lines.append(f"- `{reason}`: {count}")

    lines.extend(
        [
            "",
            "## Dataset-Build Gate",
            "",
            "Build `dataset_rank_reviewed_20260623` only after reviewed YOLO label "
            "files are placed in `reviewed_labels/`. Without those files, training "
            "would only repeat the old labels and would not measure label cleaning.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-root", type=Path, default=ERROR_ROOT)
    parser.add_argument("--review-queue", type=Path, default=REVIEW_QUEUE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instance_rows = read_csv(args.review_queue / "instance_manifest.csv")
    threshold_rows = read_csv(args.error_root / "threshold_summary.csv")
    class_rows = read_csv(args.error_root / "original_class_recall.csv")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reviewed_labels" / "train").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reviewed_labels" / "val").mkdir(parents=True, exist_ok=True)

    decision_rows = build_decision_rows(instance_rows)
    write_csv(args.out_dir / "review_decisions_template.csv", decision_rows)
    write_report(args.out_dir / "LABEL_REVIEW_STATUS_20260623.md", instance_rows, threshold_rows, class_rows)
    print(f"wrote {args.out_dir}")
    print(f"decision_rows={len(decision_rows)}")


if __name__ == "__main__":
    main()

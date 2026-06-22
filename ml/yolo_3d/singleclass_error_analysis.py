#!/usr/bin/env python3
"""Single-class validation error analysis for the current YOLO-3D baseline."""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from two_stage_validate import evaluate_predictions, mask_iou, nms_prediction_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "ml" / "yolo_3d"
RESULTS_ROOT = (
    SCRIPT_DIR / "experiment_results" / "singleclass_error_analysis_20260622"
)
DATASET_DIR = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/preprocess_v2_20260612/datasets/"
    "ia_rank_residual_intensity_gradient"
)
BEST_WEIGHTS = (
    SCRIPT_DIR
    / "experiment_results"
    / "singlecls_seed_repeat_20260622"
    / "runs"
    / "rank_yolov114_y11s_singlecls_seed0"
    / "weights"
    / "best.pt"
)
ORIGINAL_CLASS_NAMES = ["pinhole", "crap", "spatter"]
@dataclass
class DefectInstance:
    image_id: int
    cls: int
    score: float
    box: np.ndarray
    polygon: np.ndarray
    original_cls: int = -1


@dataclass
class MatchResult:
    image_id: int
    image_path: Path
    kind: str
    instance: DefectInstance
    matched: DefectInstance | None
    iou: float


def imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    encoded.tofile(str(path))


def polygon_box(points: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            float(np.min(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 0])),
            float(np.max(points[:, 1])),
        ],
        dtype=np.float32,
    )


def box_polygon(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def clip_polygon(points: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).copy()
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points


def instance_area(instance: DefectInstance | None) -> float:
    if instance is None:
        return 0.0
    x1, y1, x2, y2 = instance.box
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def original_class_name(index: int) -> str:
    if 0 <= index < len(ORIGINAL_CLASS_NAMES):
        return ORIGINAL_CLASS_NAMES[index]
    return "predicted_defect"


def read_label_file(path: Path, image_id: int, width: int, height: int) -> list[DefectInstance]:
    instances: list[DefectInstance] = []
    if not path.exists():
        return instances
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        original_cls = int(parts[0])
        coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
        if coords.size % 2:
            continue
        points = coords.reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        points = clip_polygon(points, width, height)
        if len(points) < 3:
            continue
        instances.append(
            DefectInstance(
                image_id=image_id,
                cls=0,
                score=1.0,
                box=polygon_box(points),
                polygon=points,
                original_cls=original_cls,
            )
        )
    return instances


def load_validation_data(dataset_dir: Path) -> tuple[list[Path], dict[int, list[DefectInstance]], dict[int, tuple[int, int]]]:
    image_paths = sorted((dataset_dir / "images" / "val").glob("*.png"))
    gt_by_image: dict[int, list[DefectInstance]] = {}
    shapes: dict[int, tuple[int, int]] = {}
    for image_id, image_path in enumerate(image_paths):
        image = imread(image_path)
        height, width = image.shape[:2]
        shapes[image_id] = (height, width)
        label_path = dataset_dir / "labels" / "val" / f"{image_path.stem}.txt"
        gt_by_image[image_id] = read_label_file(label_path, image_id, width, height)
    return image_paths, gt_by_image, shapes


def prediction_instances(result: Any, image_id: int, width: int, height: int) -> list[DefectInstance]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.cpu().numpy().astype(np.float32)
    mask_polygons = result.masks.xy if result.masks is not None else []
    instances: list[DefectInstance] = []
    for index, box in enumerate(boxes):
        mapped_box = box.copy()
        mapped_box[[0, 2]] = np.clip(mapped_box[[0, 2]], 0, width - 1)
        mapped_box[[1, 3]] = np.clip(mapped_box[[1, 3]], 0, height - 1)
        if index < len(mask_polygons) and len(mask_polygons[index]) >= 3:
            polygon = np.asarray(mask_polygons[index], dtype=np.float32).copy()
            polygon = clip_polygon(polygon, width, height)
        else:
            polygon = box_polygon(mapped_box)
        instances.append(
            DefectInstance(
                image_id=image_id,
                cls=0,
                score=float(scores[index]),
                box=mapped_box,
                polygon=polygon,
            )
        )
    return instances


def filter_by_score(
    predictions: dict[int, list[DefectInstance]], threshold: float
) -> dict[int, list[DefectInstance]]:
    return {
        image_id: [pred for pred in preds if pred.score >= threshold]
        for image_id, preds in predictions.items()
    }


def run_prediction(
    weights: Path,
    image_paths: list[Path],
    shapes: dict[int, tuple[int, int]],
    imgsz: int,
    conf: float,
    device: str,
) -> tuple[dict[int, list[DefectInstance]], float]:
    model = YOLO(str(weights))
    t0 = time.perf_counter()
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        conf=conf,
        iou=0.70,
        device=device,
        batch=8,
        verbose=False,
        stream=True,
    )
    predictions: dict[int, list[DefectInstance]] = {}
    for image_id, result in enumerate(results):
        height, width = shapes[image_id]
        predictions[image_id] = prediction_instances(result, image_id, width, height)
    elapsed_s = time.perf_counter() - t0
    return predictions, elapsed_s


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


def best_iou_same_image(
    item: DefectInstance, candidates: list[DefectInstance]
) -> tuple[float, DefectInstance | None]:
    best_iou = 0.0
    best_item = None
    for candidate in candidates:
        value = mask_iou(item, candidate)
        if value > best_iou:
            best_iou = value
            best_item = candidate
    return best_iou, best_item


def classify_errors(
    image_paths: list[Path],
    gt_by_image: dict[int, list[DefectInstance]],
    pred_by_image: dict[int, list[DefectInstance]],
    iou_threshold: float,
) -> tuple[list[MatchResult], list[dict[str, Any]], list[dict[str, Any]]]:
    matches: list[MatchResult] = []
    per_image_rows: list[dict[str, Any]] = []
    per_original_class = {
        cls: {"gt": 0, "tp": 0, "fn": 0}
        for cls in range(len(ORIGINAL_CLASS_NAMES))
    }

    for image_id, gt_items in gt_by_image.items():
        pred_items = sorted(
            pred_by_image.get(image_id, []), key=lambda item: item.score, reverse=True
        )
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        image_tp = image_fp = image_fn = 0

        for gt in gt_items:
            if gt.original_cls in per_original_class:
                per_original_class[gt.original_cls]["gt"] += 1

        for pred_index, pred in enumerate(pred_items):
            best_gt = -1
            best_value = 0.0
            for gt_index, gt in enumerate(gt_items):
                if gt_index in matched_gt:
                    continue
                value = mask_iou(pred, gt)
                if value > best_value:
                    best_value = value
                    best_gt = gt_index
            if best_gt >= 0 and best_value >= iou_threshold:
                matched_gt.add(best_gt)
                matched_pred.add(pred_index)
                gt = gt_items[best_gt]
                image_tp += 1
                if gt.original_cls in per_original_class:
                    per_original_class[gt.original_cls]["tp"] += 1
                matches.append(
                    MatchResult(image_id, image_paths[image_id], "tp", pred, gt, best_value)
                )

        for pred_index, pred in enumerate(pred_items):
            if pred_index in matched_pred:
                continue
            best_value, nearest_gt = best_iou_same_image(pred, gt_items)
            image_fp += 1
            matches.append(
                MatchResult(image_id, image_paths[image_id], "fp", pred, nearest_gt, best_value)
            )

        for gt_index, gt in enumerate(gt_items):
            if gt_index in matched_gt:
                continue
            best_value, nearest_pred = best_iou_same_image(gt, pred_items)
            image_fn += 1
            if gt.original_cls in per_original_class:
                per_original_class[gt.original_cls]["fn"] += 1
            matches.append(
                MatchResult(image_id, image_paths[image_id], "fn", gt, nearest_pred, best_value)
            )

        per_image_rows.append(
            {
                "image_id": image_id,
                "image": image_paths[image_id].name,
                "gt": len(gt_items),
                "pred": len(pred_items),
                "tp": image_tp,
                "fp": image_fp,
                "fn": image_fn,
                "review_score": image_fn * 2 + image_fp,
                "pinhole_gt": sum(1 for item in gt_items if item.original_cls == 0),
                "crap_gt": sum(1 for item in gt_items if item.original_cls == 1),
                "spatter_gt": sum(1 for item in gt_items if item.original_cls == 2),
            }
        )

    class_rows: list[dict[str, Any]] = []
    for cls, counts in per_original_class.items():
        recall = counts["tp"] / max(counts["gt"], 1)
        class_rows.append(
            {
                "original_class": original_class_name(cls),
                "gt": counts["gt"],
                "tp": counts["tp"],
                "fn": counts["fn"],
                "recall_at_review_conf": recall,
            }
        )
    return matches, per_image_rows, class_rows


def match_to_row(match: MatchResult) -> dict[str, Any]:
    item = match.instance
    paired = match.matched
    return {
        "kind": match.kind,
        "image_id": match.image_id,
        "image": match.image_path.name,
        "score": item.score,
        "best_mask_iou": match.iou,
        "area": instance_area(item),
        "original_class": original_class_name(item.original_cls),
        "matched_original_class": original_class_name(paired.original_cls)
        if paired is not None
        else "",
        "x1": float(item.box[0]),
        "y1": float(item.box[1]),
        "x2": float(item.box[2]),
        "y2": float(item.box[3]),
    }


def draw_instance(
    image: np.ndarray,
    instance: DefectInstance | None,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if instance is None:
        return
    points = np.round(instance.polygon).astype(np.int32)
    cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
    x1, y1, _, _ = np.round(instance.box).astype(np.int32)
    cv2.putText(
        image,
        label,
        (max(0, int(x1)), max(18, int(y1) - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def render_match(match: MatchResult, cell_size: tuple[int, int] = (420, 315)) -> np.ndarray:
    image = imread(match.image_path)
    if match.kind == "fp":
        draw_instance(image, match.matched, (0, 180, 0), "nearest GT")
        draw_instance(image, match.instance, (0, 220, 255), f"FP {match.instance.score:.2f}")
    elif match.kind == "fn":
        draw_instance(image, match.matched, (0, 220, 255), "nearest pred")
        draw_instance(
            image,
            match.instance,
            (0, 0, 255),
            f"FN {original_class_name(match.instance.original_cls)}",
        )
    else:
        draw_instance(image, match.matched, (0, 180, 0), "GT")
        draw_instance(image, match.instance, (0, 255, 0), f"TP {match.instance.score:.2f}")

    title = (
        f"{match.kind.upper()} iou={match.iou:.2f} "
        f"area={instance_area(match.instance):.0f} {match.image_path.name}"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(
        image,
        title[:100],
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.resize(image, cell_size, interpolation=cv2.INTER_AREA)


def make_contact_sheet(items: list[MatchResult], out_path: Path, max_items: int) -> bool:
    selected = items[:max_items]
    if not selected:
        return False
    cells = [render_match(item) for item in selected]
    cols = 3
    rows = int(math.ceil(len(cells) / cols))
    cell_h, cell_w = cells[0].shape[:2]
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 32, dtype=np.uint8)
    for index, cell in enumerate(cells):
        row = index // cols
        col = index % cols
        sheet[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w] = cell
    imwrite(out_path, sheet)
    return True


def write_visual_panels(matches: list[MatchResult], out_dir: Path, max_examples: int) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    panels: list[dict[str, Any]] = []
    specs = [
        (
            "fn_small_area.png",
            sorted(
                [item for item in matches if item.kind == "fn"],
                key=lambda item: instance_area(item.instance),
            ),
        ),
        (
            "fn_low_overlap.png",
            sorted(
                [item for item in matches if item.kind == "fn"],
                key=lambda item: item.iou,
            ),
        ),
        (
            "fp_high_conf.png",
            sorted(
                [item for item in matches if item.kind == "fp"],
                key=lambda item: item.instance.score,
                reverse=True,
            ),
        ),
        (
            "fp_low_overlap.png",
            sorted(
                [item for item in matches if item.kind == "fp"],
                key=lambda item: item.iou,
            ),
        ),
    ]
    for filename, items in specs:
        path = out_dir / filename
        if make_contact_sheet(items, path, max_examples):
            panels.append({"panel": filename, "examples": min(len(items), max_examples)})
    return panels


def summarize_thresholds(
    predictions: dict[int, list[DefectInstance]],
    gt_by_image: dict[int, list[DefectInstance]],
    thresholds: list[float],
    nms_iou: float,
    elapsed_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        filtered = nms_prediction_dict(filter_by_score(predictions, threshold), nms_iou)
        row, _ = evaluate_predictions(
            f"rank_singlecls_conf{threshold:g}", gt_by_image, filtered, elapsed_s
        )
        row["threshold"] = threshold
        tp = row["mask_recall50"] * sum(len(items) for items in gt_by_image.values())
        precision = row["mask_precision50"]
        recall = row["mask_recall50"]
        row["mask_f1_50"] = 2.0 * precision * recall / max(precision + recall, 1e-12)
        row["tp50_estimate"] = tp
        rows.append(row)
    return rows


def write_report(
    path: Path,
    args: argparse.Namespace,
    threshold_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    per_image_rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    panel_rows: list[dict[str, Any]],
) -> None:
    review = next(
        row for row in threshold_rows if abs(row["threshold"] - args.error_conf) < 1e-12
    )
    best_ap = max(threshold_rows, key=lambda row: row["mask_map50_95"])
    best_f1 = max(threshold_rows, key=lambda row: row["mask_f1_50"])
    fp_count = sum(1 for row in error_rows if row["kind"] == "fp")
    fn_count = sum(1 for row in error_rows if row["kind"] == "fn")
    tp_count = sum(1 for row in error_rows if row["kind"] == "tp")
    worst_images = sorted(
        per_image_rows, key=lambda row: (int(row["review_score"]), int(row["fn"]), int(row["fp"])), reverse=True
    )[:10]
    fn_areas = [float(row["area"]) for row in error_rows if row["kind"] == "fn"]
    small_fn = 0
    if fn_areas:
        median_area = float(np.median(fn_areas))
        small_fn = sum(1 for area in fn_areas if area <= median_area)
    else:
        median_area = 0.0

    lines = [
        "# Single-Class Error Analysis",
        "",
        "Date: 2026-06-22",
        "",
        "Scope: defect-level validation for the current best rank-preprocessing "
        "single-class model. Original label classes are kept only for diagnosis; "
        "all matching treats every label as one defect class.",
        "",
        "## Inputs",
        "",
        f"- dataset: `{args.dataset_dir}`",
        f"- weights: `{args.weights}`",
        f"- images: {len(per_image_rows)}",
        f"- review threshold: {args.error_conf:.3f}",
        f"- match rule: mask IoU >= {args.error_iou:.2f}",
        "",
        "## Threshold Summary",
        "",
        "These rows are fixed-confidence filtered metrics for operating-point "
        "selection and review triage; they are not a replacement for the "
        "Ultralytics full PR-curve `model.val` AP used in model ranking.",
        "",
        "| Conf | Mask mAP50 | Mask mAP50-95 | Precision50 | Recall50 | F1@50 | Pred/Image |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in threshold_rows:
        lines.append(
            f"| {row['threshold']:.3f} | {row['mask_map50']:.5f} | "
            f"{row['mask_map50_95']:.5f} | {row['mask_precision50']:.5f} | "
            f"{row['mask_recall50']:.5f} | {row['mask_f1_50']:.5f} | "
            f"{row['predictions_per_image']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Review Threshold Errors",
            "",
            f"- TP: {tp_count}",
            f"- FP: {fp_count}",
            f"- FN: {fn_count}",
            f"- precision@review: {review['mask_precision50']:.5f}",
            f"- recall@review: {review['mask_recall50']:.5f}",
            f"- best AP threshold in this sweep: {best_ap['threshold']:.3f}",
            f"- best F1 threshold in this sweep: {best_f1['threshold']:.3f}",
            f"- FN median bbox area: {median_area:.1f}",
            f"- small-area FN count at or below median: {small_fn}",
            "",
            "## Original Label Class Recall",
            "",
            "| Original Class | GT | TP | FN | Recall |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in class_rows:
        lines.append(
            f"| `{row['original_class']}` | {row['gt']} | {row['tp']} | "
            f"{row['fn']} | {row['recall_at_review_conf']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Highest Priority Review Images",
            "",
            "| Image | GT | Pred | TP | FP | FN | Score |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in worst_images:
        lines.append(
            f"| `{row['image']}` | {row['gt']} | {row['pred']} | {row['tp']} | "
            f"{row['fp']} | {row['fn']} | {row['review_score']} |"
        )
    lines.extend(
        [
            "",
            "## Visual Panels",
            "",
        ]
    )
    for row in panel_rows:
        lines.append(f"- `{row['panel']}`: {row['examples']} examples")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Use the review image list first when cleaning labels; it ranks missed "
            "defects above extra predictions.",
            "- If a high-confidence FP overlaps a real but unlabeled defect, the label "
            "should be added rather than penalizing the model.",
            "- If many FN examples are very small or ambiguous, the next model experiment "
            "should emphasize higher-resolution crops or small-object sampling.",
            "- If FN examples are mostly one original label class, keep single-class "
            "detection for localization but train a crop-level subtype classifier after "
            "the labels are cleaned.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--weights", type=Path, default=BEST_WEIGHTS)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--error-conf", type=float, default=0.25)
    parser.add_argument("--error-iou", type=float, default=0.50)
    parser.add_argument("--nms-iou", type=float, default=0.60)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.001, 0.05, 0.10, 0.20, 0.25, 0.30, 0.50],
    )
    parser.add_argument("--max-examples", type=int, default=12)
    return parser.parse_args()


def check_inputs(args: argparse.Namespace) -> None:
    if not args.dataset_dir.exists():
        raise FileNotFoundError(args.dataset_dir)
    if not (args.dataset_dir / "images" / "val").exists():
        raise FileNotFoundError(args.dataset_dir / "images" / "val")
    if not (args.dataset_dir / "labels" / "val").exists():
        raise FileNotFoundError(args.dataset_dir / "labels" / "val")
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if args.error_conf not in args.thresholds:
        args.thresholds = sorted(set(args.thresholds + [args.error_conf]))


def main() -> None:
    args = parse_args()
    check_inputs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image_paths, gt_by_image, shapes = load_validation_data(args.dataset_dir)
    predictions, elapsed_s = run_prediction(
        args.weights,
        image_paths,
        shapes,
        args.imgsz,
        args.predict_conf,
        args.device if torch.cuda.is_available() else "cpu",
    )
    predictions = nms_prediction_dict(predictions, args.nms_iou)
    threshold_rows = summarize_thresholds(
        predictions,
        gt_by_image,
        args.thresholds,
        args.nms_iou,
        elapsed_s,
    )

    review_predictions = nms_prediction_dict(
        filter_by_score(predictions, args.error_conf), args.nms_iou
    )
    matches, per_image_rows, class_rows = classify_errors(
        image_paths, gt_by_image, review_predictions, args.error_iou
    )
    error_rows = [match_to_row(match) for match in matches]
    panel_rows = write_visual_panels(
        matches,
        args.out_dir / "visual_panels",
        args.max_examples,
    )

    per_image_rows = sorted(
        per_image_rows,
        key=lambda row: (int(row["review_score"]), int(row["fn"]), int(row["fp"])),
        reverse=True,
    )
    write_csv(args.out_dir / "threshold_summary.csv", threshold_rows)
    write_csv(args.out_dir / "error_instances.csv", error_rows)
    write_csv(args.out_dir / "per_image_errors.csv", per_image_rows)
    write_csv(args.out_dir / "original_class_recall.csv", class_rows)
    write_csv(args.out_dir / "visual_panels.csv", panel_rows)
    write_report(
        args.out_dir / "analysis.md",
        args,
        threshold_rows,
        error_rows,
        per_image_rows,
        class_rows,
        panel_rows,
    )
    print(f"wrote {args.out_dir}")
    print(f"images={len(image_paths)} gt={sum(len(items) for items in gt_by_image.values())}")
    print(f"errors_csv={args.out_dir / 'error_instances.csv'}")
    print(f"report={args.out_dir / 'analysis.md'}")


if __name__ == "__main__":
    main()

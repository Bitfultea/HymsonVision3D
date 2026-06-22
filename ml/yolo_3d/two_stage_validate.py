import argparse
import csv
import gc
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/yolo_3d"
)
FOLLOWUP_ROOT = BASE_DIR / "preprocess_followup_20260611"
OPT_ROOT = BASE_DIR / "preprocess_optimization_20260611"
OUT_DIR = OPT_ROOT / "two_stage_validation_20260612"
CLASS_NAMES = ["pinhole", "crap", "spatter"]
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
OFFICIAL_FULL_IMAGE_REFERENCE_M95 = 0.47902


@dataclass
class Instance:
    image_id: int
    cls: int
    score: float
    box: np.ndarray
    polygon: np.ndarray
    proposal_score: float = 0.0
    crop_score: float = 0.0


def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def polygon_box(points):
    points = np.asarray(points, dtype=np.float32)
    return np.asarray(
        [
            float(np.min(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 0])),
            float(np.max(points[:, 1])),
        ],
        dtype=np.float32,
    )


def box_polygon(box):
    x1, y1, x2, y2 = box
    return np.asarray(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
    )


def clip_polygon(points, width, height):
    points = np.asarray(points, dtype=np.float32).copy()
    if len(points) == 0:
        return points
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points


def read_label_file(path, image_id, width, height):
    instances = []
    if not path.exists():
        return instances
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
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
            Instance(
                image_id=image_id,
                cls=cls,
                score=1.0,
                box=polygon_box(points),
                polygon=points,
            )
        )
    return instances


def prediction_instances(result, image_id, width, height, offset=(0, 0)):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.cpu().numpy().astype(np.float32)
    classes = result.boxes.cls.cpu().numpy().astype(np.int32)
    mask_polygons = []
    if result.masks is not None:
        mask_polygons = result.masks.xy
    instances = []
    offset_x, offset_y = offset
    for index, box in enumerate(boxes):
        mapped_box = box.copy()
        mapped_box[[0, 2]] += offset_x
        mapped_box[[1, 3]] += offset_y
        mapped_box[[0, 2]] = np.clip(mapped_box[[0, 2]], 0, width - 1)
        mapped_box[[1, 3]] = np.clip(mapped_box[[1, 3]], 0, height - 1)
        if index < len(mask_polygons) and len(mask_polygons[index]) >= 3:
            polygon = np.asarray(mask_polygons[index], dtype=np.float32).copy()
            polygon[:, 0] += offset_x
            polygon[:, 1] += offset_y
            polygon = clip_polygon(polygon, width, height)
        else:
            polygon = box_polygon(mapped_box)
        instances.append(
            Instance(
                image_id=image_id,
                cls=int(classes[index]),
                score=float(scores[index]),
                box=mapped_box,
                polygon=polygon,
            )
        )
    return instances


def crop_box_for_box(box, image_width, image_height, crop_size):
    x1, y1, x2, y2 = box
    center_x = 0.5 * (float(x1) + float(x2))
    center_y = 0.5 * (float(y1) + float(y2))
    crop_w = min(crop_size, image_width)
    crop_h = min(crop_size, image_height)
    left = int(round(center_x - crop_w * 0.5))
    top = int(round(center_y - crop_h * 0.5))
    left = max(0, min(left, image_width - crop_w))
    top = max(0, min(top, image_height - crop_h))
    return left, top, left + crop_w, top + crop_h


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def rasterize_polygon(points, left, top, right, bottom):
    width = max(1, int(math.ceil(right - left)))
    height = max(1, int(math.ceil(bottom - top)))
    shifted = np.asarray(points, dtype=np.float32).copy()
    shifted[:, 0] -= left
    shifted[:, 1] -= top
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(shifted).astype(np.int32)], 1)
    return mask.astype(bool)


def mask_iou(a, b):
    left = max(0.0, min(float(np.min(a.polygon[:, 0])), float(np.min(b.polygon[:, 0]))))
    top = max(0.0, min(float(np.min(a.polygon[:, 1])), float(np.min(b.polygon[:, 1]))))
    right = max(float(np.max(a.polygon[:, 0])), float(np.max(b.polygon[:, 0]))) + 1.0
    bottom = max(float(np.max(a.polygon[:, 1])), float(np.max(b.polygon[:, 1]))) + 1.0
    if right <= left or bottom <= top:
        return 0.0
    mask_a = rasterize_polygon(a.polygon, left, top, right, bottom)
    mask_b = rasterize_polygon(b.polygon, left, top, right, bottom)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union <= 0:
        return 0.0
    return float(inter / union)


def nms_instances(instances, iou_threshold=0.60):
    kept = []
    for cls in range(len(CLASS_NAMES)):
        cls_items = [item for item in instances if item.cls == cls]
        cls_items.sort(key=lambda item: item.score, reverse=True)
        while cls_items:
            current = cls_items.pop(0)
            kept.append(current)
            cls_items = [
                item
                for item in cls_items
                if box_iou(current.box, item.box) < iou_threshold
            ]
    kept.sort(key=lambda item: item.score, reverse=True)
    return kept


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0.0, 1.0, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def evaluate(gt_by_image, pred_by_image, mode):
    iou_fn = box_iou if mode == "box" else mask_iou
    class_rows = []
    ap_matrix = []
    total_gt = 0
    total_tp50 = 0
    total_fp50 = 0
    for cls, class_name in enumerate(CLASS_NAMES):
        gt_for_class = {}
        n_gt = 0
        for image_id, gts in gt_by_image.items():
            selected = [gt for gt in gts if gt.cls == cls]
            gt_for_class[image_id] = selected
            n_gt += len(selected)
        total_gt += n_gt
        preds = [
            pred
            for image_id, preds_for_image in pred_by_image.items()
            for pred in preds_for_image
            if pred.cls == cls
        ]
        preds.sort(key=lambda item: item.score, reverse=True)
        if n_gt == 0:
            continue

        class_aps = []
        tp50 = 0
        fp50 = 0
        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            matched = {image_id: set() for image_id in gt_for_class}
            tp = np.zeros(len(preds), dtype=np.float32)
            fp = np.zeros(len(preds), dtype=np.float32)
            for pred_index, pred in enumerate(preds):
                gts = gt_for_class.get(pred.image_id, [])
                best_iou = 0.0
                best_gt = -1
                for gt_index, gt in enumerate(gts):
                    if gt_index in matched[pred.image_id]:
                        continue
                    iou = iou_fn(pred, gt) if mode == "mask" else iou_fn(pred.box, gt.box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_index
                if best_iou >= threshold and best_gt >= 0:
                    tp[pred_index] = 1.0
                    matched[pred.image_id].add(best_gt)
                else:
                    fp[pred_index] = 1.0

            if len(preds) == 0:
                class_aps.append(0.0)
                if threshold_index == 0:
                    tp50 = 0
                    fp50 = 0
                continue
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / max(n_gt, 1)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
            class_aps.append(compute_ap(recall, precision))
            if threshold_index == 0:
                tp50 = int(tp.sum())
                fp50 = int(fp.sum())

        total_tp50 += tp50
        total_fp50 += fp50
        precision50 = tp50 / max(tp50 + fp50, 1)
        recall50 = tp50 / max(n_gt, 1)
        class_rows.append(
            {
                "class": class_name,
                f"{mode}_ap50": class_aps[0],
                f"{mode}_ap50_95": float(np.mean(class_aps)),
                f"{mode}_precision50": precision50,
                f"{mode}_recall50": recall50,
                "gt": n_gt,
                "pred": len(preds),
            }
        )
        ap_matrix.append(class_aps)

    if not ap_matrix:
        return {
            f"{mode}_map50": 0.0,
            f"{mode}_map50_95": 0.0,
            f"{mode}_precision50": 0.0,
            f"{mode}_recall50": 0.0,
            "per_class": class_rows,
        }
    ap_matrix = np.asarray(ap_matrix, dtype=np.float32)
    return {
        f"{mode}_map50": float(np.mean(ap_matrix[:, 0])),
        f"{mode}_map50_95": float(np.mean(ap_matrix)),
        f"{mode}_precision50": total_tp50 / max(total_tp50 + total_fp50, 1),
        f"{mode}_recall50": total_tp50 / max(total_gt, 1),
        "per_class": class_rows,
    }


def evaluate_predictions(name, gt_by_image, pred_by_image, elapsed_s=0.0):
    box_metrics = evaluate(gt_by_image, pred_by_image, "box")
    mask_metrics = evaluate(gt_by_image, pred_by_image, "mask")
    n_images = len(gt_by_image)
    n_preds = sum(len(items) for items in pred_by_image.values())
    row = {
        "name": name,
        "images": n_images,
        "predictions": n_preds,
        "predictions_per_image": n_preds / max(n_images, 1),
        "elapsed_s": elapsed_s,
    }
    for key, value in box_metrics.items():
        if key != "per_class":
            row[key] = value
    for key, value in mask_metrics.items():
        if key != "per_class":
            row[key] = value
    per_class = []
    by_name = {item["class"]: item for item in box_metrics["per_class"]}
    for mask_row in mask_metrics["per_class"]:
        combined = dict(by_name.get(mask_row["class"], {}))
        combined.update(mask_row)
        combined["name"] = name
        per_class.append(combined)
    return row, per_class


def load_validation_data(dataset_dir):
    image_paths = sorted((dataset_dir / "images" / "val").glob("*.png"))
    gt_by_image = {}
    shapes = {}
    for image_id, image_path in enumerate(image_paths):
        image = imread(image_path)
        height, width = image.shape[:2]
        shapes[image_id] = (height, width)
        label_path = dataset_dir / "labels" / "val" / f"{image_path.stem}.txt"
        gt_by_image[image_id] = read_label_file(label_path, image_id, width, height)
    return image_paths, gt_by_image, shapes


def build_crop_predictions(
    image_paths,
    full_predictions,
    crop_model,
    proposal_conf_min,
    crop_size,
    crop_imgsz,
    crop_batch,
    max_proposals,
):
    crops = []
    mapping = []
    t0 = time.perf_counter()
    for image_id, image_path in enumerate(image_paths):
        image = imread(image_path)
        height, width = image.shape[:2]
        proposals = [
            pred for pred in full_predictions[image_id] if pred.score >= proposal_conf_min
        ]
        proposals.sort(key=lambda item: item.score, reverse=True)
        for proposal in proposals[:max_proposals]:
            left, top, right, bottom = crop_box_for_box(
                proposal.box, width, height, crop_size
            )
            crops.append(image[top:bottom, left:right])
            mapping.append((image_id, left, top, proposal.score))

    crop_predictions = {image_id: [] for image_id in range(len(image_paths))}
    if crops:
        for crop, (image_id, left, top, proposal_score) in zip(crops, mapping):
            results = crop_model.predict(
                source=crop,
                imgsz=crop_imgsz,
                conf=0.001,
                iou=0.70,
                device=0,
                batch=max(1, crop_batch),
                verbose=False,
            )
            result = results[0]
            height, width = imread(image_paths[image_id]).shape[:2]
            for pred in prediction_instances(result, image_id, width, height, (left, top)):
                pred.crop_score = pred.score
                pred.proposal_score = proposal_score
                pred.score = math.sqrt(max(0.0, proposal_score) * max(0.0, pred.score))
                crop_predictions[image_id].append(pred)
            del results
            del result
    elapsed = time.perf_counter() - t0
    return crop_predictions, len(crops), elapsed


def filter_by_score(predictions, threshold):
    return {
        image_id: [pred for pred in preds if pred.score >= threshold]
        for image_id, preds in predictions.items()
    }


def filter_crop_predictions(crop_predictions, proposal_conf, crop_conf):
    filtered = {}
    for image_id, preds in crop_predictions.items():
        filtered[image_id] = [
            pred
            for pred in preds
            if pred.proposal_score >= proposal_conf and pred.crop_score >= crop_conf
        ]
    return filtered


def nms_prediction_dict(predictions, iou_threshold=0.60):
    return {
        image_id: nms_instances(preds, iou_threshold=iou_threshold)
        for image_id, preds in predictions.items()
    }


def union_predictions(a, b):
    keys = sorted(set(a) | set(b))
    return {key: list(a.get(key, [])) + list(b.get(key, [])) for key in keys}


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary_rows, per_class_rows, out_path):
    ranked = sorted(summary_rows, key=lambda row: row["mask_map50_95"], reverse=True)
    full_rows = [row for row in summary_rows if row["name"].startswith("full_only_")]
    two_stage_rows = [
        row for row in summary_rows if row["name"].startswith("two_stage_")
    ]
    best_full = max(full_rows, key=lambda row: row["mask_map50_95"])
    best_two_stage = max(two_stage_rows, key=lambda row: row["mask_map50_95"])
    delta = best_two_stage["mask_map50_95"] - best_full["mask_map50_95"]
    official_delta = (
        best_two_stage["mask_map50_95"] - OFFICIAL_FULL_IMAGE_REFERENCE_M95
    )
    lines = [
        "# Two-stage Validation 2026-06-12",
        "",
        "Validation is computed in original full-image coordinates. Full-only and two-stage rows use the same custom evaluator, so relative differences are comparable.",
        "",
        "## Conclusion",
        "",
        f"- Best two-stage: `{best_two_stage['name']}` with Mask mAP50-95 {best_two_stage['mask_map50_95']:.5f}.",
        f"- Best custom full-only baseline: `{best_full['name']}` with Mask mAP50-95 {best_full['mask_map50_95']:.5f}; two-stage delta is {delta:+.5f}.",
        f"- Previous Ultralytics full-image reference Mask mAP50-95 is {OFFICIAL_FULL_IMAGE_REFERENCE_M95:.5f}; best two-stage is {official_delta:+.5f} against that reference.",
        f"- Best two-stage takes {best_two_stage['elapsed_s']:.1f}s on this val run versus {best_full['elapsed_s']:.1f}s for full-only, and uses {best_two_stage['predictions_per_image']:.2f} predictions/image versus {best_full['predictions_per_image']:.2f}.",
        "",
        "Current conclusion: this proposal-crop two-stage path is not a clear win. It slightly improves custom thresholded Mask mAP50-95, mainly by adding recall, but it lowers precision and remains below the previous full-image reference.",
        "",
        "## Summary",
        "",
        "| Rank | Experiment | Mask mAP50 | Mask mAP50-95 | Mask R@50 | Mask P@50 | Pred/Image | Time(s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, 1):
        lines.append(
            f"| {index} | `{row['name']}` | {row['mask_map50']:.5f} | "
            f"{row['mask_map50_95']:.5f} | {row['mask_recall50']:.5f} | "
            f"{row['mask_precision50']:.5f} | {row['predictions_per_image']:.2f} | "
            f"{row['elapsed_s']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Per-class",
            "",
            "| Experiment | Class | Mask mAP50 | Mask mAP50-95 | Mask R@50 | Mask P@50 | GT | Pred |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(per_class_rows, key=lambda item: (item["name"], item["class"])):
        lines.append(
            f"| `{row['name']}` | `{row['class']}` | {row['mask_ap50']:.5f} | "
            f"{row['mask_ap50_95']:.5f} | {row['mask_recall50']:.5f} | "
            f"{row['mask_precision50']:.5f} | {row['gt']} | {row['pred']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `two_stage_refine_*` uses only crop-model detections generated from full-image proposals.",
            "- `two_stage_union_*` keeps full-image detections and adds crop-model detections, followed by box NMS.",
            "- Two-stage cannot recover objects that the full-image proposal stage never sees.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    dataset_dir = FOLLOWUP_ROOT / "datasets" / args.full_dataset
    full_model_path = (
        FOLLOWUP_ROOT / "runs" / "followup" / args.full_model / "weights" / "best.pt"
    )
    crop_model_path = OPT_ROOT / "runs" / "crop" / args.crop_model / "weights" / "best.pt"
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    if not full_model_path.exists():
        raise FileNotFoundError(full_model_path)
    if not crop_model_path.exists():
        raise FileNotFoundError(crop_model_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths, gt_by_image, shapes = load_validation_data(dataset_dir)
    full_model = YOLO(str(full_model_path))

    t0 = time.perf_counter()
    full_results = full_model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.full_imgsz,
        conf=args.full_conf_min,
        iou=0.70,
        device=0,
        batch=8,
        verbose=False,
        stream=True,
    )
    full_predictions = {}
    for image_id, result in enumerate(full_results):
        height, width = shapes[image_id]
        full_predictions[image_id] = prediction_instances(result, image_id, width, height)
    full_elapsed = time.perf_counter() - t0

    del full_results
    del full_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    crop_model = YOLO(str(crop_model_path))
    crop_predictions, crop_count, crop_elapsed = build_crop_predictions(
        image_paths=image_paths,
        full_predictions=full_predictions,
        crop_model=crop_model,
        proposal_conf_min=min(args.proposal_confs),
        crop_size=args.crop_size,
        crop_imgsz=args.crop_imgsz,
        crop_batch=args.crop_batch,
        max_proposals=args.max_proposals,
    )

    summary_rows = []
    per_class_rows = []

    full_row, full_classes = evaluate_predictions(
        f"full_only_{args.full_model}_conf{args.full_conf_min:g}",
        gt_by_image,
        nms_prediction_dict(full_predictions, args.nms_iou),
        full_elapsed,
    )
    summary_rows.append(full_row)
    per_class_rows.extend(full_classes)

    for conf in args.proposal_confs:
        if abs(conf - args.full_conf_min) < 1e-12:
            continue
        filtered_full = filter_by_score(full_predictions, conf)
        row, classes = evaluate_predictions(
            f"full_only_{args.full_model}_conf{conf:g}",
            gt_by_image,
            nms_prediction_dict(filtered_full, args.nms_iou),
            full_elapsed,
        )
        summary_rows.append(row)
        per_class_rows.extend(classes)

    for proposal_conf in args.proposal_confs:
        for crop_conf in args.crop_confs:
            filtered_crop = filter_crop_predictions(
                crop_predictions, proposal_conf, crop_conf
            )
            refined = nms_prediction_dict(filtered_crop, args.nms_iou)
            name = (
                f"two_stage_refine_{args.full_model}_{args.crop_model}_"
                f"p{proposal_conf:g}_c{crop_conf:g}"
            )
            row, classes = evaluate_predictions(
                name, gt_by_image, refined, full_elapsed + crop_elapsed
            )
            row["proposal_crops"] = crop_count
            summary_rows.append(row)
            per_class_rows.extend(classes)

            full_for_union = filter_by_score(full_predictions, proposal_conf)
            unioned = nms_prediction_dict(
                union_predictions(full_for_union, filtered_crop), args.nms_iou
            )
            name = (
                f"two_stage_union_{args.full_model}_{args.crop_model}_"
                f"p{proposal_conf:g}_c{crop_conf:g}"
            )
            row, classes = evaluate_predictions(
                name, gt_by_image, unioned, full_elapsed + crop_elapsed
            )
            row["proposal_crops"] = crop_count
            summary_rows.append(row)
            per_class_rows.extend(classes)

    summary_path = OUT_DIR / "two_stage_summary.csv"
    per_class_path = OUT_DIR / "two_stage_per_class.csv"
    report_path = OUT_DIR / "TWO_STAGE_VALIDATION_20260612.md"
    write_csv(summary_path, summary_rows)
    write_csv(per_class_path, per_class_rows)
    write_report(summary_rows, per_class_rows, report_path)

    shutil.copy2(summary_path, SCRIPT_DIR / "two_stage_validation_20260612_summary.csv")
    shutil.copy2(per_class_path, SCRIPT_DIR / "two_stage_validation_20260612_per_class.csv")
    shutil.copy2(report_path, SCRIPT_DIR / "TWO_STAGE_VALIDATION_20260612.md")

    best = max(summary_rows, key=lambda row: row["mask_map50_95"])
    print(f"images={len(image_paths)} crops={crop_count}")
    print(f"full_elapsed_s={full_elapsed:.2f} crop_elapsed_s={crop_elapsed:.2f}")
    print(
        "best "
        f"{best['name']} mask_map50={best['mask_map50']:.5f} "
        f"mask_map50_95={best['mask_map50_95']:.5f} "
        f"recall50={best['mask_recall50']:.5f} "
        f"precision50={best['mask_precision50']:.5f}"
    )
    print(f"summary: {summary_path}")
    print(f"report: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dataset", default="best_no_hsv")
    parser.add_argument("--full-model", default="best_no_hsv")
    parser.add_argument("--crop-model", default="crop_irig_384")
    parser.add_argument("--full-conf-min", type=float, default=0.05)
    parser.add_argument("--full-imgsz", type=int, default=960)
    parser.add_argument("--crop-imgsz", type=int, default=640)
    parser.add_argument("--crop-batch", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--max-proposals", type=int, default=40)
    parser.add_argument("--nms-iou", type=float, default=0.60)
    parser.add_argument(
        "--proposal-confs", nargs="*", type=float, default=[0.05, 0.10, 0.20]
    )
    parser.add_argument(
        "--crop-confs", nargs="*", type=float, default=[0.05, 0.10, 0.20]
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

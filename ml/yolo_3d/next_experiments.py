import argparse
import csv
import gc
import math
import shutil
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import ablation_experiments as ab
import pre_process_data
from two_stage_validate import (
    CLASS_NAMES,
    Instance,
    box_iou,
    evaluate_predictions,
    filter_by_score,
    imread,
    load_validation_data,
    mask_iou,
    nms_prediction_dict,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/yolo_3d"
)
FOLLOWUP_ROOT = BASE_DIR / "preprocess_followup_20260611"
V2_ROOT = BASE_DIR / "preprocess_v2_20260612"
NEXT_ROOT = BASE_DIR / "next_experiments_20260615"
ERROR_ROOT = NEXT_ROOT / "error_analysis"
HYBRID_ROOT = NEXT_ROOT / "hybrid_preprocess"

HYBRID_METHODS = [
    "hyb_lzres_int_log",
    "hyb_lzres_log_grad",
    "hyb_res_ihp_grad",
    "hyb_dog_int_grad",
    "hyb_morph_int_grad",
    "hyb_robustres_int_grad",
    "hyb_validity_res_grad",
    "hyb_abslog_int_grad",
]

HYBRID_SCREEN_CONFIG = deepcopy(ab.SCREEN_CONFIG)
STRICT_CONFIG = deepcopy(ab.FOLLOWUP_CONFIG)
STRICT_SUMMARY_FIELDS = [
    "candidate",
    "seed",
    "run",
    "score",
    "mask_map50",
    "mask_map50_epoch",
    "mask_map50_95",
    "mask_map50_95_epoch",
    "mask_recall",
    "mask_recall_epoch",
    "mask_precision",
    "mask_precision_epoch",
    "box_map50",
    "box_map50_epoch",
    "box_map50_95",
    "box_map50_95_epoch",
    "rows",
    "last_epoch",
    "dataset",
    "path",
    "best_pt",
]


def strict_candidates():
    candidates = {
        "invalid_aware": FOLLOWUP_ROOT
        / "datasets"
        / "invalid_aware_intensity_residual_gradient",
        "rank": V2_ROOT / "datasets" / "ia_rank_residual_intensity_gradient",
        "log": V2_ROOT / "datasets" / "ia_residual_intensity_log",
    }
    summary_path = HYBRID_ROOT / "hybrid_screen_summary.csv"
    if summary_path.exists():
        rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        for row in rows[:2]:
            method = row["name"].removeprefix("screen_")
            candidates[method] = hybrid_dataset_dir(method)
    else:
        candidates["hyb_dog_int_grad"] = hybrid_dataset_dir("hyb_dog_int_grad")
        candidates["hyb_lzres_log_grad"] = hybrid_dataset_dir("hyb_lzres_log_grad")
    return candidates


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dataset_dir: Path
    weights: Path
    imgsz: int = 960


MODEL_SPECS = {
    "invalid_aware": ModelSpec(
        name="invalid_aware",
        dataset_dir=FOLLOWUP_ROOT
        / "datasets"
        / "invalid_aware_intensity_residual_gradient",
        weights=FOLLOWUP_ROOT
        / "runs"
        / "followup"
        / "invalid_aware_intensity_residual_gradient"
        / "weights"
        / "best.pt",
    ),
    "rank": ModelSpec(
        name="rank",
        dataset_dir=V2_ROOT / "datasets" / "ia_rank_residual_intensity_gradient",
        weights=V2_ROOT
        / "runs"
        / "promote"
        / "ia_rank_residual_intensity_gradient"
        / "weights"
        / "best.pt",
    ),
    "log": ModelSpec(
        name="log",
        dataset_dir=V2_ROOT / "datasets" / "ia_residual_intensity_log",
        weights=V2_ROOT
        / "runs"
        / "promote"
        / "ia_residual_intensity_log"
        / "weights"
        / "best.pt",
    ),
}


@dataclass
class ErrorExample:
    model: str
    kind: str
    image_id: int
    image_path: Path
    cls: int
    score: float
    iou: float
    area: float
    pred: Instance | None
    gt: Instance | None


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def imwrite(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    encoded.tofile(str(path))


def instance_area(instance):
    if instance is None:
        return 0.0
    x1, y1, x2, y2 = instance.box
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def check_model_spec(spec):
    if not spec.dataset_dir.exists():
        raise FileNotFoundError(f"missing dataset: {spec.dataset_dir}")
    if not (spec.dataset_dir / "images" / "val").exists():
        raise FileNotFoundError(f"missing val images: {spec.dataset_dir}")
    if not (spec.dataset_dir / "labels" / "val").exists():
        raise FileNotFoundError(f"missing val labels: {spec.dataset_dir}")
    if not spec.weights.exists():
        raise FileNotFoundError(f"missing weights: {spec.weights}")


def run_prediction(spec, image_paths, shapes, conf):
    model = YOLO(str(spec.weights))
    t0 = time.perf_counter()
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=spec.imgsz,
        conf=conf,
        iou=0.70,
        device=0 if torch.cuda.is_available() else "cpu",
        batch=8,
        verbose=False,
        stream=True,
    )
    predictions = {}
    from two_stage_validate import prediction_instances

    for image_id, result in enumerate(results):
        height, width = shapes[image_id]
        predictions[image_id] = prediction_instances(result, image_id, width, height)
    elapsed_s = time.perf_counter() - t0
    del results
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions, elapsed_s


def best_iou_same_class(item, candidates, mode):
    best = 0.0
    for candidate in candidates:
        if item.cls != candidate.cls:
            continue
        if mode == "mask":
            value = mask_iou(item, candidate)
        else:
            value = box_iou(item.box, candidate.box)
        best = max(best, value)
    return best


def classify_errors(model_name, image_paths, gt_by_image, pred_by_image, iou_threshold):
    rows = []
    examples = []
    counts = {
        (cls, kind): 0
        for cls in range(len(CLASS_NAMES))
        for kind in ("tp", "fp", "fn")
    }
    for image_id, gt_items in gt_by_image.items():
        pred_items = sorted(
            pred_by_image.get(image_id, []), key=lambda item: item.score, reverse=True
        )
        matched_gt = set()
        matched_pred = set()
        for pred_index, pred in enumerate(pred_items):
            best_gt = -1
            best_iou = 0.0
            for gt_index, gt in enumerate(gt_items):
                if gt_index in matched_gt or pred.cls != gt.cls:
                    continue
                value = mask_iou(pred, gt)
                if value > best_iou:
                    best_iou = value
                    best_gt = gt_index
            if best_gt >= 0 and best_iou >= iou_threshold:
                matched_gt.add(best_gt)
                matched_pred.add(pred_index)
                counts[(pred.cls, "tp")] += 1
                rows.append(
                    error_row(model_name, "tp", image_paths[image_id], pred.cls, pred, gt_items[best_gt], best_iou)
                )
        for pred_index, pred in enumerate(pred_items):
            if pred_index in matched_pred:
                continue
            best_iou = best_iou_same_class(pred, gt_items, "mask")
            counts[(pred.cls, "fp")] += 1
            rows.append(
                error_row(model_name, "fp", image_paths[image_id], pred.cls, pred, None, best_iou)
            )
            examples.append(
                ErrorExample(
                    model_name,
                    "fp",
                    image_id,
                    image_paths[image_id],
                    pred.cls,
                    pred.score,
                    best_iou,
                    instance_area(pred),
                    pred,
                    None,
                )
            )
        for gt_index, gt in enumerate(gt_items):
            if gt_index in matched_gt:
                continue
            best_iou = best_iou_same_class(gt, pred_items, "mask")
            counts[(gt.cls, "fn")] += 1
            rows.append(
                error_row(model_name, "fn", image_paths[image_id], gt.cls, None, gt, best_iou)
            )
            examples.append(
                ErrorExample(
                    model_name,
                    "fn",
                    image_id,
                    image_paths[image_id],
                    gt.cls,
                    0.0,
                    best_iou,
                    instance_area(gt),
                    None,
                    gt,
                )
            )
    summary_rows = []
    for cls, class_name in enumerate(CLASS_NAMES):
        tp = counts[(cls, "tp")]
        fp = counts[(cls, "fp")]
        fn = counts[(cls, "fn")]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        summary_rows.append(
            {
                "model": model_name,
                "class": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision_at_error_conf": precision,
                "recall_at_error_conf": recall,
                "fp_per_image": fp / max(len(image_paths), 1),
            }
        )
    return summary_rows, rows, examples


def error_row(model, kind, image_path, cls, pred, gt, iou):
    item = pred or gt
    x1, y1, x2, y2 = item.box
    return {
        "model": model,
        "kind": kind,
        "image": image_path.name,
        "class": CLASS_NAMES[cls],
        "score": float(pred.score) if pred is not None else 0.0,
        "best_mask_iou": float(iou),
        "area": instance_area(item),
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
    }


def draw_polygon(image, instance, color, label):
    if instance is None:
        return
    points = np.round(instance.polygon).astype(np.int32)
    cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
    x1, y1, _, _ = np.round(instance.box).astype(np.int32)
    cv2.putText(
        image,
        label,
        (max(0, x1), max(18, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def render_example(example, cell_size=(360, 270)):
    image = imread(example.image_path)
    if example.kind == "fp":
        draw_polygon(image, example.pred, (0, 220, 255), f"FP {example.score:.2f}")
    elif example.kind == "fn":
        draw_polygon(image, example.gt, (0, 0, 255), "FN")
    else:
        draw_polygon(image, example.gt, (0, 180, 0), "GT")
        draw_polygon(image, example.pred, (0, 255, 0), f"TP {example.score:.2f}")

    title = (
        f"{example.model} {example.kind.upper()} {CLASS_NAMES[example.cls]} "
        f"iou={example.iou:.2f} area={example.area:.0f}"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 32), (0, 0, 0), -1)
    cv2.putText(
        image,
        title[:92],
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    resized = cv2.resize(image, cell_size, interpolation=cv2.INTER_AREA)
    return resized


def make_contact_sheet(examples, out_path, max_examples):
    if not examples:
        return False
    cells = [render_example(item) for item in examples[:max_examples]]
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


def write_error_report(path, metric_rows, error_summary_rows, verification_rows):
    best = max(metric_rows, key=lambda row: row["mask_map50_95"])
    lines = [
        "# Next Experiments Error Analysis 2026-06-15",
        "",
        "## Plan Verification",
        "",
    ]
    for row in verification_rows:
        lines.append(
            f"- `{row['model']}`: dataset={row['dataset_exists']}, "
            f"weights={row['weights_exists']}, images={row['images']}, "
            f"gt={row['gt_instances']}, predictions={row['predictions']}, "
            f"panels={row['panels']}"
        )
    lines.extend(
        [
            "",
            "## Metric Summary",
            "",
            f"- Best by mask mAP50-95: `{best['name']}` = {best['mask_map50_95']:.5f}.",
            "",
            "| Model | Threshold | Mask mAP50 | Mask mAP50-95 | Recall50 | Precision50 | Predictions/Image |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(metric_rows, key=lambda item: item["mask_map50_95"], reverse=True):
        lines.append(
            f"| `{row['name']}` | {row['threshold']:.3f} | "
            f"{row['mask_map50']:.5f} | {row['mask_map50_95']:.5f} | "
            f"{row['mask_recall50']:.5f} | {row['mask_precision50']:.5f} | "
            f"{row['predictions_per_image']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Error Counts At Review Threshold",
            "",
            "| Model | Class | TP | FP | FN | Precision | Recall | FP/Image |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in error_summary_rows:
        lines.append(
            f"| `{row['model']}` | `{row['class']}` | {row['tp']} | {row['fp']} | "
            f"{row['fn']} | {row['precision_at_error_conf']:.5f} | "
            f"{row['recall_at_error_conf']:.5f} | {row['fp_per_image']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- FP/FN visualization panels are generated under the external experiment root.",
            "- Error matching uses mask IoU >= 0.50 on same-class predictions after box NMS.",
            "- Threshold rows are useful for confidence tuning; AP rows still use low-conf predictions.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_error_analysis(args):
    selected = args.models or list(MODEL_SPECS)
    unknown = [name for name in selected if name not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}")

    ERROR_ROOT.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    per_class_rows = []
    error_summary_rows = []
    error_instance_rows = []
    verification_rows = []

    for model_name in selected:
        spec = MODEL_SPECS[model_name]
        check_model_spec(spec)
        image_paths, gt_by_image, shapes = load_validation_data(spec.dataset_dir)
        if args.limit_images:
            image_paths = image_paths[: args.limit_images]
            gt_by_image = {key: gt_by_image[key] for key in range(len(image_paths))}
            shapes = {key: shapes[key] for key in range(len(image_paths))}
        predictions, elapsed_s = run_prediction(spec, image_paths, shapes, args.predict_conf)
        predictions = nms_prediction_dict(predictions, args.nms_iou)
        base_row, base_classes = evaluate_predictions(
            f"{model_name}_all_predictions", gt_by_image, predictions, elapsed_s
        )
        base_row["model"] = model_name
        base_row["threshold"] = args.predict_conf
        metric_rows.append(base_row)
        per_class_rows.extend(base_classes)

        for threshold in args.thresholds:
            filtered = nms_prediction_dict(filter_by_score(predictions, threshold), args.nms_iou)
            row, classes = evaluate_predictions(
                f"{model_name}_conf{threshold:g}", gt_by_image, filtered, elapsed_s
            )
            row["model"] = model_name
            row["threshold"] = threshold
            metric_rows.append(row)
            per_class_rows.extend(classes)
            if abs(threshold - args.error_conf) < 1e-12:
                summary, instances, examples = classify_errors(
                    model_name, image_paths, gt_by_image, filtered, args.error_iou
                )
                error_summary_rows.extend(summary)
                error_instance_rows.extend(instances)
                panel_count = write_error_panels(
                    examples,
                    ERROR_ROOT / "visual_panels" / model_name,
                    args.max_examples,
                )
                verification_rows.append(
                    {
                        "model": model_name,
                        "dataset_exists": spec.dataset_dir.exists(),
                        "weights_exists": spec.weights.exists(),
                        "images": len(image_paths),
                        "gt_instances": sum(len(items) for items in gt_by_image.values()),
                        "predictions": sum(len(items) for items in filtered.values()),
                        "panels": panel_count,
                    }
                )

    metric_path = ERROR_ROOT / "error_analysis_summary.csv"
    per_class_path = ERROR_ROOT / "error_analysis_per_class.csv"
    error_summary_path = ERROR_ROOT / "error_analysis_error_counts.csv"
    error_instances_path = ERROR_ROOT / "error_analysis_instances.csv"
    verification_path = ERROR_ROOT / "error_analysis_verification.csv"
    report_path = ERROR_ROOT / "ERROR_ANALYSIS_20260615.md"
    write_csv(metric_path, metric_rows)
    write_csv(per_class_path, per_class_rows)
    write_csv(error_summary_path, error_summary_rows)
    write_csv(error_instances_path, error_instance_rows)
    write_csv(verification_path, verification_rows)
    write_error_report(report_path, metric_rows, error_summary_rows, verification_rows)

    copy_outputs_to_repo(
        [
            (metric_path, "next_error_analysis_summary.csv"),
            (per_class_path, "next_error_analysis_per_class.csv"),
            (error_summary_path, "next_error_analysis_error_counts.csv"),
            (error_instances_path, "next_error_analysis_instances.csv"),
            (verification_path, "next_error_analysis_verification.csv"),
            (report_path, "NEXT_ERROR_ANALYSIS_20260615.md"),
        ]
    )
    print(f"error analysis root: {ERROR_ROOT}")
    print(f"summary: {metric_path}")
    print(f"report: {report_path}")
    for row in verification_rows:
        print(
            f"verify {row['model']}: images={row['images']} gt={row['gt_instances']} "
            f"predictions={row['predictions']} panels={row['panels']}"
        )


def write_error_panels(examples, out_dir, max_examples):
    panel_count = 0
    for cls, class_name in enumerate(CLASS_NAMES):
        class_examples = [item for item in examples if item.cls == cls]
        fp = sorted(
            [item for item in class_examples if item.kind == "fp"],
            key=lambda item: item.score,
            reverse=True,
        )
        fn = sorted(
            [item for item in class_examples if item.kind == "fn"],
            key=lambda item: item.area,
        )
        if make_contact_sheet(
            fp, out_dir / class_name / "fp_high_conf.png", max_examples
        ):
            panel_count += 1
        if make_contact_sheet(
            fn, out_dir / class_name / "fn_small_area.png", max_examples
        ):
            panel_count += 1
    return panel_count


def copy_outputs_to_repo(items):
    for src, name in items:
        if src.exists():
            shutil.copy2(src, SCRIPT_DIR / name)


def intensity_channels(intensity):
    if intensity is None:
        raise ValueError("hybrid methods require intensity SubIFD")
    intensity_f = np.asarray(intensity, dtype=np.float32)
    intensity_u8 = pre_process_data.robust_normalize(intensity_f, 0.5, 99.5)
    blurred = cv2.GaussianBlur(intensity_f, (0, 0), 7.0)
    highpass_u8 = ab.signed_u8(intensity_f - blurred, 99.0)
    return intensity_u8, highpass_u8


def hybrid_feature_channels(height, intensity):
    repaired, repair_mask = ab.repair_isolated_height_outliers(height)
    features = ab.invalid_aware_float_features(height)
    clean = features["clean"]
    residual = features["residual"]
    gradient = features["gradient"]
    intensity_u8, intensity_hp_u8 = intensity_channels(intensity)

    residual_u8 = ab.signed_u8(residual, 99.0)
    lzres_u8 = ab.local_zscore_u8(residual)
    gradient_u8 = ab.normalize_u8(gradient, 0.0, 99.0)

    log_response = cv2.Laplacian(
        cv2.GaussianBlur(clean, (0, 0), 1.5), cv2.CV_32F, ksize=3
    )
    log_u8 = ab.signed_u8(log_response, 99.0)
    abslog_u8 = ab.normalize_u8(np.abs(log_response), 0.0, 99.0)

    fine = cv2.GaussianBlur(clean, (0, 0), 3.0)
    coarse = cv2.GaussianBlur(clean, (0, 0), 18.0)
    dog_u8 = ab.signed_u8(fine - coarse, 99.0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    top_hat = cv2.morphologyEx(residual_u8, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(residual_u8, cv2.MORPH_BLACKHAT, kernel)
    morph_u8 = np.maximum(top_hat, black_hat)

    robust_baseline = cv2.bilateralFilter(
        np.asarray(clean, dtype=np.float32), d=9, sigmaColor=0.4, sigmaSpace=31
    )
    robust_residual_u8 = ab.signed_u8(clean - robust_baseline, 99.0)

    invalid = repair_mask | ~np.isfinite(np.asarray(height, dtype=np.float32))
    invalid |= np.asarray(height == 0)
    invalid = cv2.dilate(invalid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    validity_u8 = np.where(invalid, 0, 255).astype(np.uint8)

    return {
        "residual": residual_u8,
        "lzres": lzres_u8,
        "gradient": gradient_u8,
        "intensity": intensity_u8,
        "intensity_hp": intensity_hp_u8,
        "log": log_u8,
        "abslog": abslog_u8,
        "dog": dog_u8,
        "morph": morph_u8,
        "robustres": robust_residual_u8,
        "validity": validity_u8,
    }


def hybrid_method_image(method, tiff_path):
    height, intensity = pre_process_data.read_tiff_height_intensity(str(tiff_path))
    channels = hybrid_feature_channels(height, intensity)
    if method == "hyb_lzres_int_log":
        return cv2.merge([channels["lzres"], channels["intensity"], channels["log"]])
    if method == "hyb_lzres_log_grad":
        return cv2.merge([channels["lzres"], channels["log"], channels["gradient"]])
    if method == "hyb_res_ihp_grad":
        return cv2.merge(
            [channels["residual"], channels["intensity_hp"], channels["gradient"]]
        )
    if method == "hyb_dog_int_grad":
        return cv2.merge([channels["dog"], channels["intensity"], channels["gradient"]])
    if method == "hyb_morph_int_grad":
        return cv2.merge(
            [channels["morph"], channels["intensity"], channels["gradient"]]
        )
    if method == "hyb_robustres_int_grad":
        return cv2.merge(
            [channels["robustres"], channels["intensity"], channels["gradient"]]
        )
    if method == "hyb_validity_res_grad":
        return cv2.merge(
            [channels["validity"], channels["residual"], channels["gradient"]]
        )
    if method == "hyb_abslog_int_grad":
        return cv2.merge(
            [channels["abslog"], channels["intensity"], channels["gradient"]]
        )
    raise ValueError(f"unknown hybrid method: {method}")


def hybrid_dataset_dir(method):
    return HYBRID_ROOT / "datasets" / method


def build_hybrid_dataset(method, limit_images=0):
    out = hybrid_dataset_dir(method)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    generated = 0
    label_count = 0
    filled_copy_labels = []
    cache = {}
    for split in ("train", "val"):
        image_paths = sorted((ab.SRC_DATASET / "images" / split).glob("*.png"))
        if limit_images:
            image_paths = image_paths[:limit_images]
        for image_path in image_paths:
            tiff_path, raw_stem = ab.source_tiff(image_path.stem)
            if tiff_path not in cache:
                cache[tiff_path] = hybrid_method_image(method, tiff_path)
            ab.imwrite(out / "images" / split / image_path.name, cache[tiff_path])
            generated += 1

            exact_label = ab.SRC_DATASET / "labels" / split / f"{image_path.stem}.txt"
            base_label = ab.SRC_DATASET / "labels" / split / f"{raw_stem}.txt"
            out_label = out / "labels" / split / f"{image_path.stem}.txt"
            if exact_label.exists():
                shutil.copy2(exact_label, out_label)
            elif base_label.exists():
                shutil.copy2(base_label, out_label)
                filled_copy_labels.append(image_path.name)
            else:
                raise FileNotFoundError(f"missing label for {image_path}")
            label_count += 1

    (out / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {out}",
                "train: images/train",
                "val: images/val",
                "nc: 3",
                "names: ['pinhole','crap','spatter']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "method": method,
        "generated": generated,
        "labels": label_count,
        "filled_copy_labels": len(filled_copy_labels),
        "dataset": str(out),
    }


def count_dataset_files(dataset):
    result = {}
    for split in ("train", "val"):
        result[f"{split}_images"] = len(list((dataset / "images" / split).glob("*.png")))
        result[f"{split}_labels"] = len(list((dataset / "labels" / split).glob("*.txt")))
    result["data_yaml"] = (dataset / "data.yaml").exists()
    return result


def hybrid_methods_from_args(args):
    methods = args.methods or HYBRID_METHODS
    unknown = [method for method in methods if method not in HYBRID_METHODS]
    if unknown:
        raise ValueError(f"unknown hybrid methods: {unknown}")
    return methods


def run_hybrid_build(args):
    HYBRID_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in hybrid_methods_from_args(args):
        row = build_hybrid_dataset(method, args.limit_images)
        row.update(count_dataset_files(Path(row["dataset"])))
        rows.append(row)
        print(
            f"{method}: generated={row['generated']} labels={row['labels']} "
            f"train={row['train_images']}/{row['train_labels']} "
            f"val={row['val_images']}/{row['val_labels']}"
        )
    out_csv = HYBRID_ROOT / "hybrid_build_verification.csv"
    write_csv(out_csv, rows)
    copy_outputs_to_repo([(out_csv, "next_hybrid_build_verification.csv")])
    print(f"hybrid build verification: {out_csv}")


def run_hybrid_screen(args):
    config = deepcopy(HYBRID_SCREEN_CONFIG)
    config["epochs"] = args.epochs
    config["imgsz"] = args.imgsz
    config["batch"] = args.batch
    for method in hybrid_methods_from_args(args):
        data_dir = hybrid_dataset_dir(method)
        if not (data_dir / "data.yaml").exists():
            raise FileNotFoundError(f"build dataset first: {data_dir}")
        ab.run_train(
            f"screen_{method}",
            data_dir,
            "screen",
            config,
            root=HYBRID_ROOT,
        )
    records = ab.summarize("screen", HYBRID_ROOT, "hybrid_screen_summary.csv")
    summary_path = HYBRID_ROOT / "hybrid_screen_summary.csv"
    copy_outputs_to_repo([(summary_path, "next_hybrid_screen_summary.csv")])
    return records


def run_hybrid_report(args):
    summary_path = HYBRID_ROOT / "hybrid_screen_summary.csv"
    build_path = HYBRID_ROOT / "hybrid_build_verification.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing screen summary: {summary_path}")
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    rows.sort(key=lambda row: float(row["score"]), reverse=True)
    build_rows = (
        list(csv.DictReader(build_path.open(encoding="utf-8")))
        if build_path.exists()
        else []
    )
    report_path = HYBRID_ROOT / "HYBRID_PREPROCESS_20260615.md"
    lines = [
        "# Hybrid Preprocess Screen 2026-06-15",
        "",
        "## Plan Verification",
        "",
    ]
    for row in build_rows:
        lines.append(
            f"- `{row['method']}`: train={row['train_images']}/{row['train_labels']}, "
            f"val={row['val_images']}/{row['val_labels']}, data_yaml={row['data_yaml']}"
        )
    lines.extend(
        [
            "",
            "## Screen Ranking",
            "",
            "| Rank | Method | Score | Mask mAP50 | Mask mAP50-95 | Recall | Precision |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for index, row in enumerate(rows, 1):
        method = row["name"].removeprefix("screen_")
        lines.append(
            f"| {index} | `{method}` | {float(row['score']):.5f} | "
            f"{float(row['mask_map50']):.5f} | {float(row['mask_map50_95']):.5f} | "
            f"{float(row['mask_recall']):.5f} | {float(row['mask_precision']):.5f} |"
        )
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "- Promote the top 2-3 hybrid methods only if they beat or tie the v2 screen leaders.",
            "- Do not compare screen numbers directly with 160-epoch promote runs.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    copy_outputs_to_repo([(report_path, "NEXT_HYBRID_PREPROCESS_20260615.md")])
    print(f"hybrid report: {report_path}")


def selected_strict_candidates(args):
    candidates = strict_candidates()
    selected = args.candidates or list(candidates)
    unknown = [name for name in selected if name not in candidates]
    if unknown:
        raise ValueError(f"unknown strict candidates: {unknown}")
    return {name: candidates[name] for name in selected}


def run_strict_repeat(args):
    seeds = args.seeds or [0, 1, 2]
    config = deepcopy(STRICT_CONFIG)
    config["epochs"] = args.epochs
    config["imgsz"] = args.imgsz
    config["batch"] = args.batch
    for candidate, data_dir in selected_strict_candidates(args).items():
        if not (data_dir / "data.yaml").exists():
            raise FileNotFoundError(f"missing data.yaml for {candidate}: {data_dir}")
        for seed in seeds:
            run_config = deepcopy(config)
            run_config["seed"] = seed
            run_name = f"{candidate}_seed{seed}"
            save_dir = ab.run_train(
                run_name,
                data_dir,
                "strict_repeat",
                run_config,
                root=NEXT_ROOT,
            )
            if not (save_dir / "results.csv").exists():
                raise FileNotFoundError(f"missing training results: {save_dir}")
    out_csv = NEXT_ROOT / "strict_repeat_summary.csv"
    rows = collect_strict_repeat_rows(min_last_epoch=max(10, args.epochs // 2))
    write_strict_summary(out_csv, rows)
    copy_outputs_to_repo([(out_csv, "next_strict_repeat_summary.csv")])
    print(f"strict repeat summary: {out_csv}")
    summarize_strict_rows(rows)


def parse_strict_run_name(run_name):
    if "_seed" not in run_name:
        return None, None
    candidate, seed_text = run_name.rsplit("_seed", 1)
    try:
        return candidate, int(seed_text)
    except ValueError:
        return None, None


def collect_strict_repeat_rows(min_last_epoch=10):
    candidates = strict_candidates()
    rows = []
    run_root = NEXT_ROOT / "runs" / "strict_repeat"
    if not run_root.exists():
        return rows
    for save_dir in sorted(run_root.iterdir()):
        result_csv = save_dir / "results.csv"
        if not result_csv.exists():
            continue
        candidate, seed = parse_strict_run_name(save_dir.name)
        if candidate is None:
            continue
        metrics = ab.read_best_metrics(result_csv)
        last_epoch = int(float(metrics.get("last_epoch", 0)))
        if last_epoch < min_last_epoch:
            continue
        data_dir = candidates.get(candidate, "")
        rec = {
            "candidate": candidate,
            "seed": seed,
            "run": save_dir.name,
            "dataset": str(data_dir),
            "path": str(save_dir),
            "best_pt": str(save_dir / "weights" / "best.pt"),
        }
        rec.update(metrics)
        rows.append(rec)
    rows.sort(key=lambda row: (row["candidate"], int(row["seed"])))
    return rows


def write_strict_summary(out_csv, rows):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STRICT_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize_strict_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["candidate"], []).append(row)
    ranked = []
    for candidate, items in grouped.items():
        values = np.asarray([float(item["mask_map50_95"]) for item in items])
        scores = np.asarray([float(item["score"]) for item in items])
        ranked.append(
            {
                "candidate": candidate,
                "runs": len(items),
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
                "mask_map50_95_mean": float(np.mean(values)),
                "mask_map50_95_std": float(np.std(values)),
                "mask_map50_95_best": float(np.max(values)),
            }
        )
    ranked.sort(key=lambda row: row["mask_map50_95_mean"], reverse=True)
    for index, row in enumerate(ranked, 1):
        print(
            f"{index:2d} {row['candidate']:<28s} runs={row['runs']} "
            f"M95 mean/std={row['mask_map50_95_mean']:.5f}/"
            f"{row['mask_map50_95_std']:.5f} best={row['mask_map50_95_best']:.5f} "
            f"score={row['score_mean']:.5f}"
        )
    return ranked


def run_strict_report(args):
    summary_path = NEXT_ROOT / "strict_repeat_summary.csv"
    rows = collect_strict_repeat_rows()
    if rows:
        write_strict_summary(summary_path, rows)
        copy_outputs_to_repo([(summary_path, "next_strict_repeat_summary.csv")])
    elif summary_path.exists():
        rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    else:
        raise FileNotFoundError(f"missing strict summary: {summary_path}")
    ranked = summarize_strict_rows(rows)
    report_path = NEXT_ROOT / "STRICT_REPEAT_20260615.md"
    baseline = next(
        (row for row in ranked if row["candidate"] == "invalid_aware"), None
    )
    best = ranked[0] if ranked else None
    lines = [
        "# Strict Repeat Comparison 2026-06-15",
        "",
        "## Plan Verification",
        "",
        f"- Candidates: {', '.join(sorted({row['candidate'] for row in rows}))}",
        f"- Runs completed: {len(rows)}",
        "- Expected per selected candidate: one run per requested seed.",
        "",
        "## Candidate Ranking",
        "",
        "| Rank | Candidate | Runs | Mean Mask mAP50-95 | Std | Best | Mean Score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranked, 1):
        lines.append(
            f"| {index} | `{row['candidate']}` | {row['runs']} | "
            f"{row['mask_map50_95_mean']:.5f} | "
            f"{row['mask_map50_95_std']:.5f} | "
            f"{row['mask_map50_95_best']:.5f} | "
            f"{row['score_mean']:.5f} |"
        )
    if best and baseline:
        delta = best["mask_map50_95_mean"] - baseline["mask_map50_95_mean"]
        lines.extend(
            [
                "",
                "## Gate Decision",
                "",
                f"- Best mean candidate: `{best['candidate']}`.",
                f"- Delta versus `invalid_aware`: {delta:.5f}.",
                "- Replacement gate: mean delta >= 0.006 and std not worse than baseline.",
            ]
        )
    lines.extend(
        [
            "",
            "## Per-run Results",
            "",
            "| Candidate | Seed | Score | Mask mAP50 | Mask mAP50-95 | Recall | Precision |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(rows, key=lambda item: (item["candidate"], int(item["seed"]))):
        lines.append(
            f"| `{row['candidate']}` | {int(row['seed'])} | "
            f"{float(row['score']):.5f} | {float(row['mask_map50']):.5f} | "
            f"{float(row['mask_map50_95']):.5f} | "
            f"{float(row['mask_recall']):.5f} | {float(row['mask_precision']):.5f} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    copy_outputs_to_repo([(report_path, "NEXT_STRICT_REPEAT_20260615.md")])
    print(f"strict report: {report_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    error = subparsers.add_parser("error-analysis")
    error.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS))
    error.add_argument("--predict-conf", type=float, default=0.001)
    error.add_argument("--error-conf", type=float, default=0.25)
    error.add_argument("--error-iou", type=float, default=0.50)
    error.add_argument("--nms-iou", type=float, default=0.60)
    error.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.10, 0.20, 0.25, 0.30, 0.50]
    )
    error.add_argument("--limit-images", type=int, default=0)
    error.add_argument("--max-examples", type=int, default=12)

    hybrid_build = subparsers.add_parser("hybrid-build")
    hybrid_build.add_argument("--methods", nargs="*", default=None)
    hybrid_build.add_argument("--limit-images", type=int, default=0)

    hybrid_screen = subparsers.add_parser("hybrid-screen")
    hybrid_screen.add_argument("--methods", nargs="*", default=None)
    hybrid_screen.add_argument("--epochs", type=int, default=80)
    hybrid_screen.add_argument("--imgsz", type=int, default=640)
    hybrid_screen.add_argument("--batch", type=int, default=16)

    subparsers.add_parser("hybrid-report")

    strict_repeat = subparsers.add_parser("strict-repeat")
    strict_repeat.add_argument("--candidates", nargs="*", default=None)
    strict_repeat.add_argument("--seeds", nargs="*", type=int, default=None)
    strict_repeat.add_argument("--epochs", type=int, default=160)
    strict_repeat.add_argument("--imgsz", type=int, default=960)
    strict_repeat.add_argument("--batch", type=int, default=16)

    subparsers.add_parser("strict-report")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "error-analysis":
        run_error_analysis(args)
    elif args.command == "hybrid-build":
        run_hybrid_build(args)
    elif args.command == "hybrid-screen":
        run_hybrid_screen(args)
    elif args.command == "hybrid-report":
        run_hybrid_report(args)
    elif args.command == "strict-repeat":
        run_strict_repeat(args)
    elif args.command == "strict-report":
        run_strict_report(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

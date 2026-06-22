import argparse
import csv
import math
import shutil
import sys
from copy import deepcopy
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import pre_process_data  # noqa: E402


BASE_DIR = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/yolo_3d"
)
SRC_DATASET = BASE_DIR / "dataset"
TIFF_DIR = BASE_DIR / "rename_tiff"
OUT_ROOT = BASE_DIR / "preprocess_ablation_20260608"
FOLLOWUP_ROOT = BASE_DIR / "preprocess_followup_20260611"
OPT_ROOT = BASE_DIR / "preprocess_optimization_20260611"
V2_ROOT = BASE_DIR / "preprocess_v2_20260612"
MODEL_S = SCRIPT_DIR / "yolo11s-seg.pt"
MODEL_M = SCRIPT_DIR / "yolo11m-seg.pt"
CLASS_NAMES = ["pinhole", "crap", "spatter"]


METHODS = [
    "legacy_height_grad_clahe",
    "defect_residual_gradient_rough",
    "convex_height_sobelxy",
    "convex_height_sobelx_clahe",
    "normal_height_nx_ny",
    "gradlimit_height_sobelxy",
    "intensity_residual_intensity_gradient",
    "intensity_residual_intensity_rough",
    "custom_residual_abs_rough",
]


SCREEN_CONFIG = {
    "epochs": 80,
    "imgsz": 640,
    "batch": 16,
    "model": MODEL_S,
    "degrees": 90.0,
    "scale": 0.5,
    "copy_paste": 0.0,
    "dropout": 0.0,
    "mosaic": 1.0,
    "cutmix": 0.0,
    "close_mosaic": 10,
}


HYPERPARAM_CONFIGS = {
    "moderate_aug": {
        "epochs": 160,
        "imgsz": 960,
        "batch": 16,
        "model": MODEL_M,
        "degrees": 90.0,
        "scale": 0.5,
        "copy_paste": 0.0,
        "dropout": 0.0,
        "mosaic": 1.0,
        "cutmix": 0.0,
        "close_mosaic": 20,
    },
    "balanced_aug": {
        "epochs": 160,
        "imgsz": 960,
        "batch": 16,
        "model": MODEL_M,
        "degrees": 180.0,
        "scale": 0.5,
        "copy_paste": 0.5,
        "dropout": 0.1,
        "mosaic": 1.0,
        "cutmix": 0.0,
        "close_mosaic": 20,
    },
    "strong_aug": {
        "epochs": 160,
        "imgsz": 960,
        "batch": 16,
        "model": MODEL_M,
        "degrees": 180.0,
        "scale": 0.8,
        "copy_paste": 0.8,
        "dropout": 0.4,
        "mosaic": 1.0,
        "cutmix": 0.3,
        "close_mosaic": 30,
    },
}


FOLLOWUP_CONFIG = {
    "epochs": 160,
    "imgsz": 960,
    "batch": 16,
    "model": MODEL_M,
    "degrees": 90.0,
    "scale": 0.5,
    "copy_paste": 0.0,
    "dropout": 0.0,
    "mosaic": 1.0,
    "cutmix": 0.0,
    "close_mosaic": 20,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
}


FOLLOWUP_EXPERIMENTS = {
    "best_no_hsv": {
        "method": "intensity_residual_intensity_gradient",
        "config": FOLLOWUP_CONFIG,
    },
    "invalid_aware_intensity_residual_gradient": {
        "method": "invalid_aware_intensity_residual_gradient",
        "config": FOLLOWUP_CONFIG,
    },
    "normal_height_nx_ny_moderate": {
        "method": "normal_height_nx_ny",
        "config": FOLLOWUP_CONFIG,
    },
}


CROP_CONFIG = {
    "epochs": 120,
    "imgsz": 640,
    "batch": 32,
    "model": MODEL_M,
    "degrees": 90.0,
    "scale": 0.4,
    "copy_paste": 0.0,
    "dropout": 0.0,
    "mosaic": 0.5,
    "cutmix": 0.0,
    "close_mosaic": 15,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
}


CROP_EXPERIMENTS = {
    "crop_irig_384": {
        "method": "intensity_residual_intensity_gradient",
        "crop_size": 384,
        "config": CROP_CONFIG,
    },
    "crop_normal_384": {
        "method": "normal_height_nx_ny",
        "crop_size": 384,
        "config": CROP_CONFIG,
    },
}


PREPROCESS_V2_METHODS = [
    "ia_residual_intensity_log",
    "ia_residual_intensity_curvature",
    "ia_residual_intensity_rough15",
    "ia_multiscale_residual_intensity",
    "ia_residual_absres_intensity",
    "ia_residual_gradient_curvature",
    "ia_normalz_residual_intensity",
    "ia_rank_residual_intensity_gradient",
]


PREPROCESS_V2_SCREEN_CONFIG = deepcopy(SCREEN_CONFIG)
PREPROCESS_V2_PROMOTE_CONFIG = deepcopy(FOLLOWUP_CONFIG)
PREPROCESS_V2_REFERENCE_M95 = 0.47901
PREPROCESS_V2_REPLACE_DELTA = 0.006


def imwrite(path, image):
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    encoded.tofile(str(path))


def normalize_u8(data, lower_percent=0.0, upper_percent=99.0):
    data = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(data)
    vals = data[finite]
    if vals.size == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    lower = np.percentile(vals, lower_percent)
    upper = np.percentile(vals, upper_percent)
    if upper <= lower:
        return np.zeros(data.shape, dtype=np.uint8)
    clipped = np.clip(data, lower, upper)
    return ((clipped - lower) / (upper - lower) * 255).astype(np.uint8)


def signed_u8(data, percentile=99.0):
    data = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(data)
    vals = np.abs(data[finite])
    if vals.size == 0:
        return np.full(data.shape, 128, dtype=np.uint8)
    limit = np.percentile(vals, percentile)
    if limit <= 0:
        return np.full(data.shape, 128, dtype=np.uint8)
    clipped = np.clip(data, -limit, limit)
    return ((clipped + limit) / (2 * limit) * 255).astype(np.uint8)


def repair_isolated_height_outliers(
    height,
    global_percentiles=(0.05, 99.95),
    median_ksize=5,
    max_component_area=4,
):
    """Repair only non-finite values and tiny extreme connected components."""
    height = np.asarray(height, dtype=np.float32)
    finite = np.isfinite(height)
    if not np.any(finite):
        return np.zeros_like(height, dtype=np.float32), np.ones_like(height, dtype=bool)

    repaired = height.copy()
    finite_values = height[finite]
    fill_value = float(np.median(finite_values))
    filled = np.where(finite, height, fill_value).astype(np.float32)
    median_ksize = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
    local_median = cv2.medianBlur(filled, median_ksize)

    low, high = np.percentile(finite_values, global_percentiles)
    candidate = (~finite) | (filled < low) | (filled > high)
    repair_mask = ~finite

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    for label_index in range(1, labels_count):
        area = stats[label_index, cv2.CC_STAT_AREA]
        if area <= max_component_area:
            repair_mask |= labels == label_index

    repaired[repair_mask] = local_median[repair_mask]
    return repaired, repair_mask


def invalid_aware_intensity_residual_gradient(height, intensity):
    if intensity is None:
        raise ValueError("missing intensity SubIFD")
    repaired_height, _ = repair_isolated_height_outliers(height)
    residual, gradient, _ = pre_process_data.height_defect_feature_channels(
        repaired_height
    )
    intensity_u8 = pre_process_data.robust_normalize(intensity, 0.5, 99.5)
    return cv2.merge([residual, intensity_u8, gradient])


def invalid_aware_float_features(height, baseline_sigma=15):
    repaired_height, _ = repair_isolated_height_outliers(height)
    clean = pre_process_data.preprocess_height_for_features(repaired_height)
    baseline = cv2.GaussianBlur(clean, (0, 0), baseline_sigma)
    residual = clean - baseline
    sobel_x = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    laplacian = cv2.Laplacian(residual, cv2.CV_32F, ksize=3)
    return {
        "clean": clean,
        "residual": residual,
        "gradient": gradient,
        "laplacian": laplacian,
    }


def local_zscore_u8(data, ksize=31):
    if ksize % 2 == 0:
        ksize += 1
    data = np.asarray(data, dtype=np.float32)
    mean = cv2.blur(data, (ksize, ksize))
    mean_sq = cv2.blur(data * data, (ksize, ksize))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    zscore = (data - mean) / (std + 1e-3)
    return signed_u8(zscore, 99.0)


def normal_slope_u8(clean_height):
    dz_dx, dz_dy = np.gradient(np.asarray(clean_height, dtype=np.float32))
    slope = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)
    normal_z = 1.0 / np.sqrt(1.0 + slope * slope)
    return normalize_u8(1.0 - normal_z, 0.0, 99.0)


def preprocess_v2_image(method, height, intensity):
    if intensity is None and "intensity" in method:
        raise ValueError(f"{method} requires intensity SubIFD")

    features = invalid_aware_float_features(height)
    residual = features["residual"]
    clean = features["clean"]
    gradient = features["gradient"]
    laplacian = features["laplacian"]
    residual_u8 = signed_u8(residual, 99.0)
    gradient_u8 = normalize_u8(gradient, 0.0, 99.0)
    curvature_u8 = normalize_u8(np.abs(laplacian), 0.0, 99.0)
    intensity_u8 = (
        pre_process_data.robust_normalize(intensity, 0.5, 99.5)
        if intensity is not None
        else np.zeros_like(residual_u8)
    )

    if method == "ia_residual_intensity_log":
        blurred = cv2.GaussianBlur(clean, (0, 0), 1.5)
        log_response = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        return cv2.merge([residual_u8, intensity_u8, signed_u8(log_response, 99.0)])

    if method == "ia_residual_intensity_curvature":
        return cv2.merge([residual_u8, intensity_u8, curvature_u8])

    if method == "ia_residual_intensity_rough15":
        rough15 = pre_process_data.local_roughness(residual, 15)
        return cv2.merge([residual_u8, intensity_u8, normalize_u8(rough15, 0.0, 99.0)])

    if method == "ia_multiscale_residual_intensity":
        fine = invalid_aware_float_features(height, baseline_sigma=8)["residual"]
        coarse = invalid_aware_float_features(height, baseline_sigma=25)["residual"]
        return cv2.merge([signed_u8(fine, 99.0), intensity_u8, signed_u8(coarse, 99.0)])

    if method == "ia_residual_absres_intensity":
        abs_residual = normalize_u8(np.abs(residual), 0.0, 99.0)
        return cv2.merge([residual_u8, abs_residual, intensity_u8])

    if method == "ia_residual_gradient_curvature":
        return cv2.merge([residual_u8, gradient_u8, curvature_u8])

    if method == "ia_normalz_residual_intensity":
        return cv2.merge([residual_u8, intensity_u8, normal_slope_u8(clean)])

    if method == "ia_rank_residual_intensity_gradient":
        return cv2.merge([local_zscore_u8(residual), intensity_u8, gradient_u8])

    raise ValueError(f"unknown preprocess-v2 method: {method}")


def custom_residual_abs_rough(height):
    height = np.asarray(height, dtype=np.float32)
    height = pre_process_data.preprocess_height_for_features(height)
    baseline = cv2.GaussianBlur(height, (0, 0), 15)
    residual = height - baseline
    roughness = pre_process_data.local_roughness(residual, 7)
    return cv2.merge(
        [
            signed_u8(residual, 99.0),
            normalize_u8(np.abs(residual), 0.0, 99.0),
            normalize_u8(roughness, 0.0, 99.0),
        ]
    )


def method_image(method, tiff_path):
    height, intensity = pre_process_data.read_tiff_height_intensity(str(tiff_path))
    if method in PREPROCESS_V2_METHODS:
        return preprocess_v2_image(method, height, intensity)
    if method == "legacy_height_grad_clahe":
        raw = np.asarray(height, dtype=np.float32)
        norm_height = pre_process_data.robust_normalize(raw, 0.5, 99.5)
        lower = np.percentile(raw, 0.5)
        upper = np.percentile(raw, 99.5)
        raw_clipped = np.clip(raw, lower, upper)
        sobelx = cv2.Sobel(raw_clipped, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(raw_clipped, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        norm_grad = pre_process_data.robust_normalize(gradient_mag, 0, 98.0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(norm_height)
        return cv2.merge([norm_height, norm_grad, enhanced])
    if method == "defect_residual_gradient_rough":
        residual, gradient, rough = pre_process_data.height_defect_feature_channels(
            height
        )
        return cv2.merge([residual, gradient, rough])
    if method == "convex_height_sobelxy":
        raw = np.nan_to_num(np.asarray(height, dtype=np.float32), nan=0.0)
        lower = np.percentile(raw, 0.1)
        upper = np.percentile(raw, 99.9)
        raw_clamped = np.clip(raw, lower, upper)
        norm_height = pre_process_data.robust_normalize(raw_clamped, 0, 100)
        sobel_x = cv2.Sobel(raw_clamped, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(raw_clamped, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.merge(
            [
                norm_height,
                pre_process_data.robust_normalize(sobel_x, 0.1, 99.9),
                pre_process_data.robust_normalize(sobel_y, 0.1, 99.9),
            ]
        )
    if method == "convex_height_sobelx_clahe":
        raw = np.nan_to_num(np.asarray(height, dtype=np.float32), nan=0.0)
        lower = np.percentile(raw, 0.1)
        upper = np.percentile(raw, 99.9)
        raw_clamped = np.clip(raw, lower, upper)
        norm_height = pre_process_data.robust_normalize(raw_clamped, 0, 100)
        sobel_x = cv2.Sobel(raw_clamped, cv2.CV_64F, 1, 0, ksize=3)
        norm_sobel_x = pre_process_data.robust_normalize(sobel_x, 0.1, 99.9)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(norm_height)
        return cv2.merge([norm_height, norm_sobel_x, enhanced])
    if method == "normal_height_nx_ny":
        raw = np.nan_to_num(np.asarray(height, dtype=np.float32), nan=0.0)
        lower = np.percentile(raw, 0.1)
        upper = np.percentile(raw, 99.9)
        raw_clamped = np.clip(raw, lower, upper)
        denom = upper - lower
        if denom == 0:
            denom = 1
        norm_height = ((raw_clamped - lower) / denom * 255).astype(np.uint8)
        nx, ny = pre_process_data.calculate_normal_map(raw)
        return cv2.merge(
            [norm_height, pre_process_data.normalize_to_rgb(nx), pre_process_data.normalize_to_rgb(ny)]
        )
    if method == "gradlimit_height_sobelxy":
        raw = np.nan_to_num(np.asarray(height, dtype=np.float32), nan=0.0)
        min_pt = np.min(raw)
        max_pt = np.max(raw)
        denom = max_pt - min_pt
        if denom == 0:
            denom = 1
        norm_height = ((raw - min_pt) / denom * 255).astype(np.uint8)
        grad_limit = 0.75
        sobel_x = cv2.Sobel(raw, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(raw, cv2.CV_64F, 0, 1, ksize=3)

        def fixed_normalize(grad):
            clipped = np.clip(grad, -grad_limit, grad_limit)
            return ((clipped + grad_limit) / (2 * grad_limit) * 255).astype(
                np.uint8
            )

        return cv2.merge([norm_height, fixed_normalize(sobel_x), fixed_normalize(sobel_y)])
    if method == "intensity_residual_intensity_gradient":
        if intensity is None:
            raise ValueError(f"{tiff_path} has no intensity SubIFD")
        residual, gradient, _ = pre_process_data.height_defect_feature_channels(height)
        intensity_u8 = pre_process_data.robust_normalize(intensity, 0.5, 99.5)
        return cv2.merge([residual, intensity_u8, gradient])
    if method == "invalid_aware_intensity_residual_gradient":
        if intensity is None:
            raise ValueError(f"{tiff_path} has no intensity SubIFD")
        return invalid_aware_intensity_residual_gradient(height, intensity)
    if method == "intensity_residual_intensity_rough":
        if intensity is None:
            raise ValueError(f"{tiff_path} has no intensity SubIFD")
        residual, _, rough = pre_process_data.height_defect_feature_channels(height)
        intensity_u8 = pre_process_data.robust_normalize(intensity, 0.5, 99.5)
        return cv2.merge([residual, intensity_u8, rough])
    if method == "custom_residual_abs_rough":
        return custom_residual_abs_rough(height)
    raise ValueError(f"unknown method: {method}")


def source_tiff(image_stem):
    raw_stem = image_stem.split(" (", 1)[0]
    for suffix in (".tif", ".tiff"):
        path = TIFF_DIR / f"{raw_stem}{suffix}"
        if path.exists():
            return path, raw_stem
    raise FileNotFoundError(f"missing TIFF for image stem: {image_stem}")


def dataset_dir(method, root=OUT_ROOT):
    return root / "datasets" / method


def run_dir(group, name, root=OUT_ROOT):
    return root / "runs" / group / name


def build_dataset(dataset_name, method=None, root=OUT_ROOT):
    method = method or dataset_name
    out = dataset_dir(dataset_name, root)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    generated = 0
    label_count = 0
    filled_copy_labels = []
    cache = {}
    for split in ("train", "val"):
        image_paths = sorted((SRC_DATASET / "images" / split).glob("*.png"))
        for image_path in image_paths:
            tiff_path, raw_stem = source_tiff(image_path.stem)
            if tiff_path not in cache:
                cache[tiff_path] = method_image(method, tiff_path)
            imwrite(out / "images" / split / image_path.name, cache[tiff_path])
            generated += 1

            exact_label = SRC_DATASET / "labels" / split / f"{image_path.stem}.txt"
            base_label = SRC_DATASET / "labels" / split / f"{raw_stem}.txt"
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
    print(
        f"{dataset_name}: method={method}, generated={generated}, labels={label_count}, "
        f"filled_copy_labels={len(filled_copy_labels)}"
    )


def read_label_polygons(path):
    polygons = []
    if not path.exists():
        return polygons
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
        if coords.size % 2 != 0:
            continue
        polygons.append({"class": cls, "points": coords.reshape(-1, 2)})
    return polygons


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def clip_polygon_axis(points, axis, lower, keep_greater):
    if len(points) == 0:
        return points
    clipped = []
    previous = points[-1]
    previous_inside = (
        previous[axis] >= lower if keep_greater else previous[axis] <= lower
    )
    for current in points:
        current_inside = (
            current[axis] >= lower if keep_greater else current[axis] <= lower
        )
        if current_inside != previous_inside:
            denom = current[axis] - previous[axis]
            if abs(float(denom)) > 1e-6:
                ratio = (lower - previous[axis]) / denom
                clipped.append(previous + ratio * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    if not clipped:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(clipped, dtype=np.float32)


def clip_polygon_to_box(points, left, top, right, bottom):
    clipped = np.asarray(points, dtype=np.float32)
    clipped = clip_polygon_axis(clipped, 0, left, True)
    clipped = clip_polygon_axis(clipped, 0, right, False)
    clipped = clip_polygon_axis(clipped, 1, top, True)
    clipped = clip_polygon_axis(clipped, 1, bottom, False)
    return clipped


def crop_box_for_points(points, image_width, image_height, crop_size):
    xmin = float(np.min(points[:, 0]))
    xmax = float(np.max(points[:, 0]))
    ymin = float(np.min(points[:, 1]))
    ymax = float(np.max(points[:, 1]))
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    crop_w = min(crop_size, image_width)
    crop_h = min(crop_size, image_height)
    left = int(round(center_x - crop_w * 0.5))
    top = int(round(center_y - crop_h * 0.5))
    left = max(0, min(left, image_width - crop_w))
    top = max(0, min(top, image_height - crop_h))
    return left, top, left + crop_w, top + crop_h


def transform_polygons_for_crop(polygons, crop_box, image_shape):
    image_h, image_w = image_shape[:2]
    left, top, right, bottom = crop_box
    crop_w = right - left
    crop_h = bottom - top
    transformed = []
    for polygon in polygons:
        points = polygon["points"].copy()
        points[:, 0] *= image_w
        points[:, 1] *= image_h
        clipped = clip_polygon_to_box(points, left, top, right, bottom)
        if polygon_area(clipped) < 1.0:
            continue
        clipped[:, 0] = np.clip((clipped[:, 0] - left) / crop_w, 0.0, 1.0)
        clipped[:, 1] = np.clip((clipped[:, 1] - top) / crop_h, 0.0, 1.0)
        if len(clipped) >= 3:
            transformed.append({"class": polygon["class"], "points": clipped})
    return transformed


def write_label_polygons(path, polygons):
    lines = []
    for polygon in polygons:
        coords = []
        for x, y in polygon["points"]:
            coords.extend([f"{float(x):.6f}", f"{float(y):.6f}"])
        lines.append(f"{polygon['class']} {' '.join(coords)}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_crop_dataset(dataset_name, method, crop_size=384, root=OPT_ROOT):
    out = dataset_dir(dataset_name, root)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "dataset": dataset_name,
        "method": method,
        "crop_size": crop_size,
        "train_crops": 0,
        "val_crops": 0,
        "train_objects": 0,
        "val_objects": 0,
    }
    cache = {}
    for split in ("train", "val"):
        image_paths = sorted((SRC_DATASET / "images" / split).glob("*.png"))
        for image_path in image_paths:
            tiff_path, raw_stem = source_tiff(image_path.stem)
            if tiff_path not in cache:
                cache[tiff_path] = method_image(method, tiff_path)
            image = cache[tiff_path]
            image_h, image_w = image.shape[:2]

            exact_label = SRC_DATASET / "labels" / split / f"{image_path.stem}.txt"
            base_label = SRC_DATASET / "labels" / split / f"{raw_stem}.txt"
            label_path = exact_label if exact_label.exists() else base_label
            polygons = read_label_polygons(label_path)
            stats[f"{split}_objects"] += len(polygons)

            for crop_index, anchor in enumerate(polygons):
                points = anchor["points"].copy()
                points[:, 0] *= image_w
                points[:, 1] *= image_h
                crop_box = crop_box_for_points(points, image_w, image_h, crop_size)
                crop_polygons = transform_polygons_for_crop(
                    polygons, crop_box, image.shape
                )
                if not crop_polygons:
                    continue
                left, top, right, bottom = crop_box
                crop = image[top:bottom, left:right]
                crop_name = (
                    f"{image_path.stem}_crop{crop_index:03d}_"
                    f"c{anchor['class']}.png"
                )
                imwrite(out / "images" / split / crop_name, crop)
                write_label_polygons(
                    out / "labels" / split / crop_name.replace(".png", ".txt"),
                    crop_polygons,
                )
                stats[f"{split}_crops"] += 1

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
    print("crop_dataset", stats)
    return stats


def label_distribution(data_dir):
    counts = Counter()
    for path in sorted((data_dir / "labels" / "train").glob("*.txt")):
        for line in path.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return {CLASS_NAMES[index]: counts[index] for index in range(len(CLASS_NAMES))}


def run_train(name, data_dir, group, config, root=OUT_ROOT):
    save_dir = run_dir(group, name, root)
    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        try:
            rows = list(csv.DictReader(results_csv.open()))
            if len(rows) >= int(config["epochs"]):
                print(f"skip completed run: {name}")
                return save_dir
        except Exception:
            pass

    print(f"training {name}")
    print("labels", label_distribution(data_dir))
    model = YOLO(str(config["model"]))
    model.train(
        data=str(data_dir / "data.yaml"),
        epochs=int(config["epochs"]),
        imgsz=int(config["imgsz"]),
        batch=int(config["batch"]),
        device=0,
        workers=4,
        patience=0,
        dropout=float(config["dropout"]),
        project=str(root / "runs" / group),
        name=name,
        exist_ok=True,
        seed=int(config.get("seed", 0)),
        augment=True,
        single_cls=False,
        hsv_h=float(config.get("hsv_h", 0.015)),
        hsv_s=float(config.get("hsv_s", 0.7)),
        hsv_v=float(config.get("hsv_v", 0.4)),
        degrees=float(config["degrees"]),
        scale=float(config["scale"]),
        fliplr=0.5,
        flipud=0.5,
        mosaic=float(config["mosaic"]),
        close_mosaic=int(config["close_mosaic"]),
        copy_paste=float(config["copy_paste"]),
        cache=True,
        mixup=0.0,
        cutmix=float(config["cutmix"]),
        plots=True,
    )
    return save_dir


def read_best_metrics(results_csv):
    rows = list(csv.DictReader(results_csv.open()))
    keys = {
        "box_map50": "metrics/mAP50(B)",
        "box_map50_95": "metrics/mAP50-95(B)",
        "mask_precision": "metrics/precision(M)",
        "mask_recall": "metrics/recall(M)",
        "mask_map50": "metrics/mAP50(M)",
        "mask_map50_95": "metrics/mAP50-95(M)",
    }
    record = {"rows": len(rows), "last_epoch": int(float(rows[-1]["epoch"]))}
    for out_key, csv_key in keys.items():
        best = max(rows, key=lambda row: float(row[csv_key]))
        record[out_key] = float(best[csv_key])
        record[f"{out_key}_epoch"] = int(float(best["epoch"]))
    record["score"] = (
        0.60 * record["mask_map50_95"]
        + 0.20 * record["mask_map50"]
        + 0.15 * record["box_map50_95"]
        + 0.05 * record["box_map50"]
    )
    return record


def summarize(group, root=OUT_ROOT, out_name=None):
    runs_root = root / "runs" / group
    records = []
    for results_csv in sorted(runs_root.glob("*/results.csv")):
        rec = {
            "name": results_csv.parent.name,
            "path": str(results_csv.parent),
            "best_pt": str(results_csv.parent / "weights" / "best.pt"),
        }
        rec.update(read_best_metrics(results_csv))
        records.append(rec)
    records.sort(key=lambda item: item["score"], reverse=True)
    out_csv = root / (out_name or f"{group}_summary.csv")
    fields = [
        "name",
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
        "path",
        "best_pt",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(f"summary: {out_csv}")
    for index, rec in enumerate(records[:10], 1):
        print(
            f"{index:2d} {rec['name']:<42s} score={rec['score']:.5f} "
            f"M50={rec['mask_map50']:.5f} M95={rec['mask_map50_95']:.5f} "
            f"R={rec['mask_recall']:.5f} P={rec['mask_precision']:.5f}"
        )
    return records


def build_all(methods):
    for method in methods:
        build_dataset(method)


def train_screen(methods):
    for method in methods:
        run_train(f"screen_{method}", dataset_dir(method), "screen", SCREEN_CONFIG)
    summarize("screen")


def train_hyperparams(method):
    data_dir = dataset_dir(method)
    for name, config in HYPERPARAM_CONFIGS.items():
        run_train(f"hp_{method}_{name}", data_dir, "hyperparams", config)
    summarize("hyperparams")


def count_labels():
    result = {}
    for split in ("train", "val"):
        counts = Counter()
        files = 0
        objects = 0
        for path in sorted((SRC_DATASET / "labels" / split).glob("*.txt")):
            files += 1
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    cls = int(line.split()[0])
                    counts[CLASS_NAMES[cls]] += 1
                    objects += 1
        result[split] = {
            "files": files,
            "objects": objects,
            "classes": {name: counts[name] for name in CLASS_NAMES},
        }
    return result


def split_leakage_report():
    raw_to_splits = {}
    for split in ("train", "val"):
        for path in sorted((SRC_DATASET / "images" / split).glob("*.png")):
            raw_stem = path.stem.split(" (", 1)[0]
            raw_to_splits.setdefault(raw_stem, []).append((split, path.name))
    cross_split = {
        stem: items
        for stem, items in raw_to_splits.items()
        if len({split for split, _ in items}) > 1
    }
    copies = {
        stem: items for stem, items in raw_to_splits.items() if len(items) > 1
    }
    return {
        "unique_raw": len(raw_to_splits),
        "with_copies": len(copies),
        "cross_split_raw": len(cross_split),
        "cross_split_examples": list(cross_split.items())[:10],
        "copy_examples": list(copies.items())[:10],
    }


def tiff_height_stats():
    paths = sorted(list(TIFF_DIR.glob("*.tif")) + list(TIFF_DIR.glob("*.tiff")))
    zero_ratios = []
    nan_ratios = []
    mins = []
    maxs = []
    p001 = []
    p999 = []
    for path in paths:
        height, intensity = pre_process_data.read_tiff_height_intensity(str(path))
        height = np.asarray(height, dtype=np.float32)
        finite = np.isfinite(height)
        values = height[finite]
        zero_ratios.append(float(np.mean(height == 0) * 100.0))
        nan_ratios.append(float(np.mean(~finite) * 100.0))
        mins.append(float(np.nanmin(height)))
        maxs.append(float(np.nanmax(height)))
        p001.append(float(np.percentile(values, 0.1)))
        p999.append(float(np.percentile(values, 99.9)))
        if intensity is not None and height.shape[:2] != intensity.shape[:2]:
            raise ValueError(
                f"{path.name}: height/intensity mismatch "
                f"{height.shape} vs {intensity.shape}"
            )

    def summary(values):
        arr = np.asarray(values, dtype=np.float32)
        return {
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    return {
        "n_tiffs": len(paths),
        "zero_percent": summary(zero_ratios),
        "nan_percent": summary(nan_ratios),
        "min": summary(mins),
        "p0.1": summary(p001),
        "p99.9": summary(p999),
        "max": summary(maxs),
    }


def audit_followup_inputs():
    audit = {
        "labels": count_labels(),
        "splits": split_leakage_report(),
        "tiff": tiff_height_stats(),
    }
    print("label_counts", audit["labels"])
    print("split_report", audit["splits"])
    print("tiff_stats", audit["tiff"])
    return audit


def build_followup(experiments):
    audit_followup_inputs()
    for experiment in experiments:
        spec = FOLLOWUP_EXPERIMENTS[experiment]
        build_dataset(experiment, method=spec["method"], root=FOLLOWUP_ROOT)


def train_followup(experiments):
    for experiment in experiments:
        spec = FOLLOWUP_EXPERIMENTS[experiment]
        data_dir = dataset_dir(experiment, FOLLOWUP_ROOT)
        run_train(
            experiment,
            data_dir,
            "followup",
            spec["config"],
            root=FOLLOWUP_ROOT,
        )
    summarize("followup", FOLLOWUP_ROOT, "followup_summary.csv")


def validate_followup_classes(experiments):
    rows = []
    for experiment in experiments:
        data_dir = dataset_dir(experiment, FOLLOWUP_ROOT)
        best_pt = run_dir("followup", experiment, FOLLOWUP_ROOT) / "weights" / "best.pt"
        if not best_pt.exists():
            print(f"skip validation without best.pt: {experiment}")
            continue
        print(f"validating per-class metrics: {experiment}")
        model = YOLO(str(best_pt))
        metrics = model.val(
            data=str(data_dir / "data.yaml"),
            imgsz=int(FOLLOWUP_CONFIG["imgsz"]),
            batch=int(FOLLOWUP_CONFIG["batch"]),
            device=0,
            workers=4,
            plots=False,
            save_json=False,
            project=str(FOLLOWUP_ROOT / "runs" / "per_class_val"),
            name=f"val_{experiment}",
            exist_ok=True,
        )
        for class_index, class_name in enumerate(CLASS_NAMES):
            try:
                box_p, box_r, box_m50, box_m95 = metrics.box.class_result(
                    class_index
                )
                mask_p, mask_r, mask_m50, mask_m95 = metrics.seg.class_result(
                    class_index
                )
            except Exception as exc:
                print(f"per-class metric unavailable for {experiment}: {exc}")
                break
            rows.append(
                {
                    "experiment": experiment,
                    "class": class_name,
                    "box_precision": float(box_p),
                    "box_recall": float(box_r),
                    "box_map50": float(box_m50),
                    "box_map50_95": float(box_m95),
                    "mask_precision": float(mask_p),
                    "mask_recall": float(mask_r),
                    "mask_map50": float(mask_m50),
                    "mask_map50_95": float(mask_m95),
                }
            )

    out_csv = FOLLOWUP_ROOT / "followup_per_class_summary.csv"
    fields = [
        "experiment",
        "class",
        "box_precision",
        "box_recall",
        "box_map50",
        "box_map50_95",
        "mask_precision",
        "mask_recall",
        "mask_map50",
        "mask_map50_95",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"per-class summary: {out_csv}")
    return rows


def crop_experiments_from_args(args):
    experiments = args.experiments or list(CROP_EXPERIMENTS)
    unknown = [name for name in experiments if name not in CROP_EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown crop experiments: {unknown}")
    return experiments


def build_crop_experiments(experiments):
    audit_followup_inputs()
    for experiment in experiments:
        spec = CROP_EXPERIMENTS[experiment]
        build_crop_dataset(
            experiment,
            method=spec["method"],
            crop_size=int(spec["crop_size"]),
            root=OPT_ROOT,
        )


def train_crop_experiments(experiments):
    for experiment in experiments:
        spec = CROP_EXPERIMENTS[experiment]
        data_dir = dataset_dir(experiment, OPT_ROOT)
        run_train(experiment, data_dir, "crop", spec["config"], root=OPT_ROOT)
    summarize("crop", OPT_ROOT, "crop_summary.csv")


def validate_crop_classes(experiments):
    rows = []
    for experiment in experiments:
        spec = CROP_EXPERIMENTS[experiment]
        data_dir = dataset_dir(experiment, OPT_ROOT)
        best_pt = run_dir("crop", experiment, OPT_ROOT) / "weights" / "best.pt"
        if not best_pt.exists():
            print(f"skip validation without best.pt: {experiment}")
            continue
        print(f"validating crop per-class metrics: {experiment}")
        model = YOLO(str(best_pt))
        metrics = model.val(
            data=str(data_dir / "data.yaml"),
            imgsz=int(spec["config"]["imgsz"]),
            batch=int(spec["config"]["batch"]),
            device=0,
            workers=4,
            plots=False,
            save_json=False,
            project=str(OPT_ROOT / "runs" / "crop_per_class_val"),
            name=f"val_{experiment}",
            exist_ok=True,
        )
        for class_index, class_name in enumerate(CLASS_NAMES):
            try:
                box_p, box_r, box_m50, box_m95 = metrics.box.class_result(
                    class_index
                )
                mask_p, mask_r, mask_m50, mask_m95 = metrics.seg.class_result(
                    class_index
                )
            except Exception as exc:
                print(f"crop per-class metric unavailable for {experiment}: {exc}")
                break
            rows.append(
                {
                    "experiment": experiment,
                    "class": class_name,
                    "box_precision": float(box_p),
                    "box_recall": float(box_r),
                    "box_map50": float(box_m50),
                    "box_map50_95": float(box_m95),
                    "mask_precision": float(mask_p),
                    "mask_recall": float(mask_r),
                    "mask_map50": float(mask_m50),
                    "mask_map50_95": float(mask_m95),
                }
            )

    out_csv = OPT_ROOT / "crop_per_class_summary.csv"
    fields = [
        "experiment",
        "class",
        "box_precision",
        "box_recall",
        "box_map50",
        "box_map50_95",
        "mask_precision",
        "mask_recall",
        "mask_map50",
        "mask_map50_95",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"crop per-class summary: {out_csv}")
    return rows


def train_repeat_seeds(experiment="best_no_hsv", seeds=None):
    seeds = seeds or [1, 2]
    if experiment not in FOLLOWUP_EXPERIMENTS:
        raise ValueError(f"unknown follow-up experiment: {experiment}")
    data_dir = dataset_dir(experiment, FOLLOWUP_ROOT)
    if not (data_dir / "data.yaml").exists():
        build_followup([experiment])
    for seed in seeds:
        config = deepcopy(FOLLOWUP_EXPERIMENTS[experiment]["config"])
        config["seed"] = int(seed)
        run_train(
            f"{experiment}_seed{seed}",
            data_dir,
            "repeat_seed",
            config,
            root=OPT_ROOT,
        )
    summarize("repeat_seed", OPT_ROOT, "repeat_seed_summary.csv")


def copy_optimization_summaries_to_repo():
    copies = [
        (OPT_ROOT / "crop_summary.csv", SCRIPT_DIR / "optimization_crop_summary.csv"),
        (
            OPT_ROOT / "crop_per_class_summary.csv",
            SCRIPT_DIR / "optimization_crop_per_class_summary.csv",
        ),
        (
            OPT_ROOT / "repeat_seed_summary.csv",
            SCRIPT_DIR / "optimization_repeat_seed_summary.csv",
        ),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)


def write_optimization_report():
    crop_records = read_csv_records(OPT_ROOT / "crop_summary.csv")
    crop_per_class = read_csv_records(OPT_ROOT / "crop_per_class_summary.csv")
    repeat_records = read_csv_records(OPT_ROOT / "repeat_seed_summary.csv")
    followup_records = read_csv_records(FOLLOWUP_ROOT / "followup_summary.csv")
    best_followup = None
    if followup_records:
        best_followup = max(
            followup_records,
            key=lambda row: float(row.get("mask_map50_95", 0.0)),
        )

    lines = [
        "# YOLO 3D Optimization Experiments 2026-06-11",
        "",
        "This report tracks optimizations that do not require new manual labels.",
        "",
        "## Crop Dataset",
        "",
        "Object-centered crop datasets are second-stage candidates. Their metrics are not directly comparable with full-image detection because validation happens on crop images.",
        "",
    ]
    if crop_records:
        lines.extend(
            [
                "| Rank | Experiment | Score | Mask mAP50 | Mask mAP50-95 | Mask Recall | Mask Precision |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        sorted_crop = sorted(
            crop_records, key=lambda row: float(row.get("score", 0.0)), reverse=True
        )
        for index, record in enumerate(sorted_crop, 1):
            lines.append(
                f"| {index} | `{record['name']}` | {float(record['score']):.5f} | "
                f"{float(record['mask_map50']):.5f} | "
                f"{float(record['mask_map50_95']):.5f} | "
                f"{float(record['mask_recall']):.5f} | "
                f"{float(record['mask_precision']):.5f} |"
            )
    else:
        lines.append("No crop training results found yet.")

    if crop_per_class:
        lines.extend(
            [
                "",
                "## Crop Per-class Validation",
                "",
                "| Experiment | Class | Mask mAP50 | Mask mAP50-95 | Mask Recall | Mask Precision |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for record in crop_per_class:
            lines.append(
                f"| `{record['experiment']}` | `{record['class']}` | "
                f"{float(record['mask_map50']):.5f} | "
                f"{float(record['mask_map50_95']):.5f} | "
                f"{float(record['mask_recall']):.5f} | "
                f"{float(record['mask_precision']):.5f} |"
            )

    lines.extend(["", "## Repeat Seed Stability", ""])
    if repeat_records:
        mask_values = [
            float(record["mask_map50_95"])
            for record in repeat_records
            if record.get("mask_map50_95")
        ]
        lines.extend(
            [
                "| Run | Score | Mask mAP50 | Mask mAP50-95 | Mask Recall | Mask Precision |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for record in repeat_records:
            lines.append(
                f"| `{record['name']}` | {float(record['score']):.5f} | "
                f"{float(record['mask_map50']):.5f} | "
                f"{float(record['mask_map50_95']):.5f} | "
                f"{float(record['mask_recall']):.5f} | "
                f"{float(record['mask_precision']):.5f} |"
            )
        if mask_values:
            lines.extend(
                [
                    "",
                    (
                        "Repeat-seed Mask mAP50-95 mean/std: "
                        f"{float(np.mean(mask_values)):.5f} / "
                        f"{float(np.std(mask_values)):.5f}."
                    ),
                ]
            )
    else:
        lines.append("No repeat-seed training results found yet.")

    if best_followup:
        lines.extend(
            [
                "",
                "## Current Full-image Reference",
                "",
                (
                    f"Best follow-up full-image Mask mAP50-95: `{best_followup['name']}` "
                    f"= {float(best_followup['mask_map50_95']):.5f}."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py optimization-build",
            "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py optimization-train-crop",
            "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py optimization-validate-crop",
            "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py optimization-repeat --method best_no_hsv --seeds 1 2",
            "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py optimization-report",
            "```",
            "",
        ]
    )

    report_path = SCRIPT_DIR / "OPTIMIZATION_EXPERIMENTS_20260611.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    copy_optimization_summaries_to_repo()
    print(f"optimization report: {report_path}")


def preprocess_v2_methods_from_args(args):
    methods = args.v2_methods or PREPROCESS_V2_METHODS
    unknown = [method for method in methods if method not in PREPROCESS_V2_METHODS]
    if unknown:
        raise ValueError(f"unknown preprocess-v2 methods: {unknown}")
    return methods


def build_preprocess_v2(methods):
    audit_followup_inputs()
    for method in methods:
        build_dataset(method, method=method, root=V2_ROOT)


def train_preprocess_v2_screen(methods):
    for method in methods:
        data_dir = dataset_dir(method, V2_ROOT)
        run_train(
            f"screen_{method}",
            data_dir,
            "screen",
            PREPROCESS_V2_SCREEN_CONFIG,
            root=V2_ROOT,
        )
    summarize("screen", V2_ROOT, "preprocess_v2_screen_summary.csv")
    copy_preprocess_v2_summaries_to_repo()


def preprocess_v2_promote_methods(methods=None, top_k=3):
    if methods:
        return methods
    records = read_csv_records(V2_ROOT / "preprocess_v2_screen_summary.csv")
    if not records:
        raise RuntimeError("no preprocess-v2 screen summary; run preprocess-v2-screen")
    records = sorted(records, key=lambda row: float(row["score"]), reverse=True)
    selected = []
    for record in records:
        method = record["name"].removeprefix("screen_")
        if method in PREPROCESS_V2_METHODS and method not in selected:
            selected.append(method)
        if len(selected) >= top_k:
            break
    return selected


def train_preprocess_v2_promote(methods=None, top_k=3):
    selected = preprocess_v2_promote_methods(methods, top_k)
    for method in selected:
        data_dir = dataset_dir(method, V2_ROOT)
        run_train(
            method,
            data_dir,
            "promote",
            PREPROCESS_V2_PROMOTE_CONFIG,
            root=V2_ROOT,
        )
    summarize("promote", V2_ROOT, "preprocess_v2_promote_summary.csv")
    copy_preprocess_v2_summaries_to_repo()


def validate_preprocess_v2_classes(methods=None, top_k=3):
    selected = preprocess_v2_promote_methods(methods, top_k)
    rows = []
    for method in selected:
        data_dir = dataset_dir(method, V2_ROOT)
        best_pt = run_dir("promote", method, V2_ROOT) / "weights" / "best.pt"
        if not best_pt.exists():
            print(f"skip preprocess-v2 validation without best.pt: {method}")
            continue
        print(f"validating preprocess-v2 per-class metrics: {method}")
        model = YOLO(str(best_pt))
        metrics = model.val(
            data=str(data_dir / "data.yaml"),
            imgsz=int(PREPROCESS_V2_PROMOTE_CONFIG["imgsz"]),
            batch=int(PREPROCESS_V2_PROMOTE_CONFIG["batch"]),
            device=0,
            workers=4,
            plots=False,
            save_json=False,
            project=str(V2_ROOT / "runs" / "per_class_val"),
            name=f"val_{method}",
            exist_ok=True,
        )
        for class_index, class_name in enumerate(CLASS_NAMES):
            try:
                box_p, box_r, box_m50, box_m95 = metrics.box.class_result(
                    class_index
                )
                mask_p, mask_r, mask_m50, mask_m95 = metrics.seg.class_result(
                    class_index
                )
            except Exception as exc:
                print(f"preprocess-v2 per-class metric unavailable: {exc}")
                break
            rows.append(
                {
                    "experiment": method,
                    "class": class_name,
                    "box_precision": float(box_p),
                    "box_recall": float(box_r),
                    "box_map50": float(box_m50),
                    "box_map50_95": float(box_m95),
                    "mask_precision": float(mask_p),
                    "mask_recall": float(mask_r),
                    "mask_map50": float(mask_m50),
                    "mask_map50_95": float(mask_m95),
                }
            )

    out_csv = V2_ROOT / "preprocess_v2_per_class_summary.csv"
    fields = [
        "experiment",
        "class",
        "box_precision",
        "box_recall",
        "box_map50",
        "box_map50_95",
        "mask_precision",
        "mask_recall",
        "mask_map50",
        "mask_map50_95",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"preprocess-v2 per-class summary: {out_csv}")
    copy_preprocess_v2_summaries_to_repo()
    return rows


def train_preprocess_v2_repeat(method=None, seeds=None):
    seeds = seeds or [1, 2]
    if method is None:
        promote_records = read_csv_records(V2_ROOT / "preprocess_v2_promote_summary.csv")
        if not promote_records:
            raise RuntimeError("no preprocess-v2 promote summary; pass --method")
        method = max(
            promote_records,
            key=lambda row: float(row.get("mask_map50_95", 0.0)),
        )["name"]
    if method not in PREPROCESS_V2_METHODS:
        raise ValueError(f"unknown preprocess-v2 method: {method}")
    data_dir = dataset_dir(method, V2_ROOT)
    for seed in seeds:
        config = deepcopy(PREPROCESS_V2_PROMOTE_CONFIG)
        config["seed"] = int(seed)
        run_train(
            f"{method}_seed{seed}",
            data_dir,
            "repeat_seed",
            config,
            root=V2_ROOT,
        )
    summarize("repeat_seed", V2_ROOT, "preprocess_v2_repeat_seed_summary.csv")
    copy_preprocess_v2_summaries_to_repo()


def copy_preprocess_v2_summaries_to_repo():
    copies = [
        (
            V2_ROOT / "preprocess_v2_screen_summary.csv",
            SCRIPT_DIR / "preprocess_v2_20260612_screen_summary.csv",
        ),
        (
            V2_ROOT / "preprocess_v2_promote_summary.csv",
            SCRIPT_DIR / "preprocess_v2_20260612_promote_summary.csv",
        ),
        (
            V2_ROOT / "preprocess_v2_per_class_summary.csv",
            SCRIPT_DIR / "preprocess_v2_20260612_per_class.csv",
        ),
        (
            V2_ROOT / "preprocess_v2_repeat_seed_summary.csv",
            SCRIPT_DIR / "preprocess_v2_20260612_repeat_seed_summary.csv",
        ),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)


def write_preprocess_v2_report():
    screen_records = read_csv_records(V2_ROOT / "preprocess_v2_screen_summary.csv")
    promote_records = read_csv_records(V2_ROOT / "preprocess_v2_promote_summary.csv")
    per_class = read_csv_records(V2_ROOT / "preprocess_v2_per_class_summary.csv")
    repeat_records = read_csv_records(V2_ROOT / "preprocess_v2_repeat_seed_summary.csv")
    followup_records = read_csv_records(FOLLOWUP_ROOT / "followup_summary.csv")
    baseline_records = read_csv_records(OPT_ROOT / "repeat_seed_summary.csv")

    best_screen = max(screen_records, key=lambda row: float(row["score"]), default=None)
    best_promote = max(
        promote_records,
        key=lambda row: float(row.get("mask_map50_95", 0.0)),
        default=None,
    )
    best_followup = max(
        followup_records,
        key=lambda row: float(row.get("mask_map50_95", 0.0)),
        default=None,
    )
    baseline_m95 = PREPROCESS_V2_REFERENCE_M95
    baseline_values = [
        float(row["mask_map50_95"])
        for row in baseline_records
        if row.get("name", "").startswith("best_no_hsv_seed")
    ]
    if baseline_values:
        baseline_m95 = float(np.mean(baseline_values))
    repeat_values = [
        float(row["mask_map50_95"])
        for row in repeat_records
        if row.get("mask_map50_95")
    ]
    repeat_mean = float(np.mean(repeat_values)) if repeat_values else None
    candidate_m95 = repeat_mean
    if candidate_m95 is None and best_promote:
        candidate_m95 = float(best_promote["mask_map50_95"])
    recommend_replace = (
        candidate_m95 is not None
        and candidate_m95 >= baseline_m95 + PREPROCESS_V2_REPLACE_DELTA
    )

    lines = [
        "# YOLO 3D Preprocess V2 Experiments 2026-06-12",
        "",
        "This report records the second round of targeted preprocessing ablations.",
        "",
        "## Conclusion",
        "",
    ]
    if best_promote:
        delta = float(best_promote["mask_map50_95"]) - baseline_m95
        lines.append(
            f"- Best long-run candidate: `{best_promote['name']}` "
            f"Mask mAP50-95 {float(best_promote['mask_map50_95']):.5f} "
            f"({delta:+.5f} vs repeat-seed baseline {baseline_m95:.5f})."
        )
    if repeat_mean is not None:
        lines.append(
            f"- New-method repeat-seed Mask mAP50-95 mean/std: "
            f"{repeat_mean:.5f} / {float(np.std(repeat_values)):.5f}."
        )
    if best_followup:
        lines.append(
            f"- Existing follow-up reference: `{best_followup['name']}` "
            f"Mask mAP50-95 {float(best_followup['mask_map50_95']):.5f}."
        )
    lines.append(
        "- Recommendation: "
        + (
            "replace the default preprocessing with the best V2 candidate."
            if recommend_replace
            else "do not replace the current default unless qualitative review favors a class-specific tradeoff."
        )
    )

    lines.extend(["", "## Screen Results", ""])
    if screen_records:
        lines.extend(
            [
                "| Rank | Method | Score | Mask mAP50 | Mask mAP50-95 | Recall | Precision |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        sorted_screen = sorted(
            screen_records, key=lambda row: float(row["score"]), reverse=True
        )
        for index, record in enumerate(sorted_screen, 1):
            name = record["name"].removeprefix("screen_")
            lines.append(
                f"| {index} | `{name}` | {float(record['score']):.5f} | "
                f"{float(record['mask_map50']):.5f} | "
                f"{float(record['mask_map50_95']):.5f} | "
                f"{float(record['mask_recall']):.5f} | "
                f"{float(record['mask_precision']):.5f} |"
            )
    else:
        lines.append("No screen results found yet.")

    lines.extend(["", "## Promoted Long Runs", ""])
    if promote_records:
        lines.extend(
            [
                "| Rank | Method | Score | Mask mAP50 | Mask mAP50-95 | Recall | Precision |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        sorted_promote = sorted(
            promote_records,
            key=lambda row: float(row.get("score", 0.0)),
            reverse=True,
        )
        for index, record in enumerate(sorted_promote, 1):
            lines.append(
                f"| {index} | `{record['name']}` | {float(record['score']):.5f} | "
                f"{float(record['mask_map50']):.5f} | "
                f"{float(record['mask_map50_95']):.5f} | "
                f"{float(record['mask_recall']):.5f} | "
                f"{float(record['mask_precision']):.5f} |"
            )
    else:
        lines.append("No promoted runs found yet.")

    if per_class:
        lines.extend(
            [
                "",
                "## Per-class Validation",
                "",
                "| Method | Class | Mask mAP50 | Mask mAP50-95 | Recall | Precision |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for record in per_class:
            lines.append(
                f"| `{record['experiment']}` | `{record['class']}` | "
                f"{float(record['mask_map50']):.5f} | "
                f"{float(record['mask_map50_95']):.5f} | "
                f"{float(record['mask_recall']):.5f} | "
                f"{float(record['mask_precision']):.5f} |"
            )

    if repeat_records:
        lines.extend(
            [
                "",
                "## Repeat Seed",
                "",
                "| Run | Score | Mask mAP50 | Mask mAP50-95 | Recall | Precision |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for record in repeat_records:
            lines.append(
                f"| `{record['name']}` | {float(record['score']):.5f} | "
                f"{float(record['mask_map50']):.5f} | "
                f"{float(record['mask_map50_95']):.5f} | "
                f"{float(record['mask_recall']):.5f} | "
                f"{float(record['mask_precision']):.5f} |"
            )

    if best_screen:
        lines.extend(
            [
                "",
                "## Reproduction",
                "",
                "```bash",
                "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py preprocess-v2-build",
                "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py preprocess-v2-screen",
                "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py preprocess-v2-promote",
                "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py preprocess-v2-validate",
                "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py preprocess-v2-repeat",
                "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py preprocess-v2-report",
                "```",
                "",
            ]
        )

    report_path = SCRIPT_DIR / "PREPROCESS_V2_EXPERIMENTS_20260612.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    copy_preprocess_v2_summaries_to_repo()
    print(f"preprocess-v2 report: {report_path}")


def read_csv_records(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def copy_followup_summaries_to_repo():
    copies = [
        (FOLLOWUP_ROOT / "followup_summary.csv", SCRIPT_DIR / "followup_experiments_20260611_summary.csv"),
        (
            FOLLOWUP_ROOT / "followup_per_class_summary.csv",
            SCRIPT_DIR / "followup_experiments_20260611_per_class.csv",
        ),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)


def write_followup_report():
    records = read_csv_records(FOLLOWUP_ROOT / "followup_summary.csv")
    per_class = read_csv_records(FOLLOWUP_ROOT / "followup_per_class_summary.csv")
    baseline_records = read_csv_records(
        SCRIPT_DIR / "preprocess_ablation_20260608_hyperparams_summary.csv"
    )
    baseline = None
    for record in baseline_records:
        if record["name"] == "hp_intensity_residual_intensity_gradient_moderate_aug":
            baseline = record
            break

    audit = audit_followup_inputs()
    lines = [
        "# YOLO 3D Follow-up Experiments 2026-06-11",
        "",
        "This report records the follow-up experiments after reading the 2.5D TIFF research report.",
        "",
        "## Data Audit",
        "",
        f"- Unique raw TIFF stems: {audit['splits']['unique_raw']}",
        f"- Raw stems with copied images: {audit['splits']['with_copies']}",
        f"- Raw stems crossing train/val: {audit['splits']['cross_split_raw']}",
        f"- TIFF count: {audit['tiff']['n_tiffs']}",
        f"- Height zero percent median/max: {audit['tiff']['zero_percent']['median']:.5f}% / {audit['tiff']['zero_percent']['max']:.5f}%",
        f"- Height NaN percent median/max: {audit['tiff']['nan_percent']['median']:.5f}% / {audit['tiff']['nan_percent']['max']:.5f}%",
        "",
        "## Follow-up Runs",
        "",
        "| Rank | Experiment | Score | Mask mAP50 | Mask mAP50-95 | Mask Recall | Mask Precision |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    sorted_records = sorted(
        records, key=lambda row: float(row.get("score", 0.0)), reverse=True
    )
    for index, record in enumerate(sorted_records, 1):
        lines.append(
            f"| {index} | `{record['name']}` | {float(record['score']):.5f} | "
            f"{float(record['mask_map50']):.5f} | "
            f"{float(record['mask_map50_95']):.5f} | "
            f"{float(record['mask_recall']):.5f} | "
            f"{float(record['mask_precision']):.5f} |"
        )

    if baseline:
        lines.extend(
            [
                "",
                "## Baseline",
                "",
                "| Experiment | Score | Mask mAP50 | Mask mAP50-95 | Mask Recall | Mask Precision |",
                "|---|---:|---:|---:|---:|---:|",
                (
                    f"| `{baseline['name']}` | {float(baseline['score']):.5f} | "
                    f"{float(baseline['mask_map50']):.5f} | "
                    f"{float(baseline['mask_map50_95']):.5f} | "
                    f"{float(baseline['mask_recall']):.5f} | "
                    f"{float(baseline['mask_precision']):.5f} |"
                ),
            ]
        )

    if sorted_records:
        best = sorted_records[0]
        best_mask = max(
            records, key=lambda row: float(row.get("mask_map50_95", 0.0))
        )
        lines.extend(
            [
                "",
                "## Recommendation",
                "",
                f"Best follow-up by summary score: `{best['name']}`.",
                f"Best follow-up by Mask mAP50-95: `{best_mask['name']}`.",
                "",
            ]
        )
        if baseline:
            best_mask_delta = float(best_mask["mask_map50_95"]) - float(
                baseline["mask_map50_95"]
            )
            best_score_delta = float(best["score"]) - float(baseline["score"])
            lines.extend(
                [
                    (
                        "Compared with the baseline, the best follow-up score delta is "
                        f"{best_score_delta:+.5f}, and the best follow-up Mask mAP50-95 "
                        f"delta is {best_mask_delta:+.5f}."
                    ),
                    "No follow-up run should replace the baseline on Mask mAP50-95 alone unless repeat-seed or qualitative inspection confirms the gain.",
                    "",
                ]
            )
        if per_class:
            pinhole_rows = [row for row in per_class if row.get("class") == "pinhole"]
            if pinhole_rows:
                best_pinhole = max(
                    pinhole_rows,
                    key=lambda row: float(row.get("mask_map50_95", 0.0)),
                )
                lines.extend(
                    [
                        (
                            "For pinhole specifically, the no-augment per-class validation "
                            f"favors `{best_pinhole['experiment']}` "
                            f"(Mask mAP50-95 {float(best_pinhole['mask_map50_95']):.5f}, "
                            f"recall {float(best_pinhole['mask_recall']):.5f})."
                        ),
                        "",
                    ]
                )
        lines.extend(
            [
                "Recommended weights for the best follow-up score:",
                "",
                "```text",
                best["best_pt"],
                "```",
            ]
        )

    if per_class:
        lines.extend(
            [
                "",
                "## Per-class Validation",
                "",
                "| Experiment | Class | Mask mAP50 | Mask mAP50-95 | Mask Recall | Mask Precision |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for record in per_class:
            lines.append(
                f"| `{record['experiment']}` | `{record['class']}` | "
                f"{float(record['mask_map50']):.5f} | "
                f"{float(record['mask_map50_95']):.5f} | "
                f"{float(record['mask_recall']):.5f} | "
                f"{float(record['mask_precision']):.5f} |"
            )

    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py followup-build",
            "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py followup-train",
            "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py followup-validate",
            "/home/charles/miniconda3/envs/yolo/bin/python ml/yolo_3d/ablation_experiments.py followup-report",
            "```",
            "",
        ]
    )

    report_path = SCRIPT_DIR / "FOLLOWUP_EXPERIMENTS_20260611.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    copy_followup_summaries_to_repo()
    print(f"report: {report_path}")


def followup_experiments_from_args(args):
    experiments = args.experiments or list(FOLLOWUP_EXPERIMENTS)
    unknown = [name for name in experiments if name not in FOLLOWUP_EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown follow-up experiments: {unknown}")
    return experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "build",
            "screen",
            "hyperparams",
            "summarize-screen",
            "summarize-hp",
            "followup-build",
            "followup-train",
            "followup-summarize",
            "followup-validate",
            "followup-report",
            "followup-all",
            "optimization-build",
            "optimization-train-crop",
            "optimization-validate-crop",
            "optimization-repeat",
            "optimization-report",
            "optimization-all",
            "preprocess-v2-build",
            "preprocess-v2-screen",
            "preprocess-v2-promote",
            "preprocess-v2-validate",
            "preprocess-v2-repeat",
            "preprocess-v2-report",
            "preprocess-v2-all",
        ],
    )
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--v2-methods", nargs="*", default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FOLLOWUP_ROOT.mkdir(parents=True, exist_ok=True)
    OPT_ROOT.mkdir(parents=True, exist_ok=True)
    V2_ROOT.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        build_all(args.methods)
    elif args.command == "screen":
        train_screen(args.methods)
    elif args.command == "hyperparams":
        if not args.method:
            screen_records = summarize("screen")
            if not screen_records:
                raise RuntimeError("no screen results; pass --method or run screen first")
            args.method = screen_records[0]["name"].removeprefix("screen_")
        train_hyperparams(args.method)
    elif args.command == "summarize-screen":
        summarize("screen")
    elif args.command == "summarize-hp":
        summarize("hyperparams")
    elif args.command == "followup-build":
        build_followup(followup_experiments_from_args(args))
    elif args.command == "followup-train":
        train_followup(followup_experiments_from_args(args))
    elif args.command == "followup-summarize":
        summarize("followup", FOLLOWUP_ROOT, "followup_summary.csv")
        copy_followup_summaries_to_repo()
    elif args.command == "followup-validate":
        validate_followup_classes(followup_experiments_from_args(args))
        copy_followup_summaries_to_repo()
    elif args.command == "followup-report":
        write_followup_report()
    elif args.command == "followup-all":
        experiments = followup_experiments_from_args(args)
        build_followup(experiments)
        train_followup(experiments)
        validate_followup_classes(experiments)
        write_followup_report()
    elif args.command == "optimization-build":
        build_crop_experiments(crop_experiments_from_args(args))
    elif args.command == "optimization-train-crop":
        train_crop_experiments(crop_experiments_from_args(args))
    elif args.command == "optimization-validate-crop":
        validate_crop_classes(crop_experiments_from_args(args))
        copy_optimization_summaries_to_repo()
    elif args.command == "optimization-repeat":
        train_repeat_seeds(args.method or "best_no_hsv", args.seeds)
    elif args.command == "optimization-report":
        write_optimization_report()
    elif args.command == "optimization-all":
        experiments = crop_experiments_from_args(args)
        build_crop_experiments(experiments)
        train_crop_experiments(experiments)
        validate_crop_classes(experiments)
        train_repeat_seeds(args.method or "best_no_hsv", args.seeds)
        write_optimization_report()
    elif args.command == "preprocess-v2-build":
        build_preprocess_v2(preprocess_v2_methods_from_args(args))
    elif args.command == "preprocess-v2-screen":
        train_preprocess_v2_screen(preprocess_v2_methods_from_args(args))
    elif args.command == "preprocess-v2-promote":
        methods = args.v2_methods or None
        train_preprocess_v2_promote(methods, args.top_k)
    elif args.command == "preprocess-v2-validate":
        methods = args.v2_methods or None
        validate_preprocess_v2_classes(methods, args.top_k)
    elif args.command == "preprocess-v2-repeat":
        train_preprocess_v2_repeat(args.method, args.seeds)
    elif args.command == "preprocess-v2-report":
        write_preprocess_v2_report()
    elif args.command == "preprocess-v2-all":
        methods = preprocess_v2_methods_from_args(args)
        build_preprocess_v2(methods)
        train_preprocess_v2_screen(methods)
        promoted = preprocess_v2_promote_methods(args.v2_methods, args.top_k)
        train_preprocess_v2_promote(promoted, args.top_k)
        validate_preprocess_v2_classes(promoted, args.top_k)
        train_preprocess_v2_repeat(args.method, args.seeds)
        write_preprocess_v2_report()


if __name__ == "__main__":
    main()

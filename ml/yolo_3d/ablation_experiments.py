import argparse
import csv
import math
import shutil
import sys
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


def dataset_dir(method):
    return OUT_ROOT / "datasets" / method


def run_dir(group, name):
    return OUT_ROOT / "runs" / group / name


def build_dataset(method):
    out = dataset_dir(method)
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
        f"{method}: generated={generated}, labels={label_count}, "
        f"filled_copy_labels={len(filled_copy_labels)}"
    )


def label_distribution(data_dir):
    counts = Counter()
    for path in sorted((data_dir / "labels" / "train").glob("*.txt")):
        for line in path.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return {CLASS_NAMES[index]: counts[index] for index in range(len(CLASS_NAMES))}


def run_train(name, data_dir, group, config):
    save_dir = run_dir(group, name)
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
        project=str(OUT_ROOT / "runs" / group),
        name=name,
        exist_ok=True,
        augment=True,
        single_cls=False,
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


def summarize(group):
    runs_root = OUT_ROOT / "runs" / group
    records = []
    for results_csv in sorted(runs_root.glob("*/results.csv")):
        rec = {"name": results_csv.parent.name, "path": str(results_csv.parent)}
        rec.update(read_best_metrics(results_csv))
        records.append(rec)
    records.sort(key=lambda item: item["score"], reverse=True)
    out_csv = OUT_ROOT / f"{group}_summary.csv"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["build", "screen", "hyperparams", "summarize-screen", "summarize-hp"],
    )
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--method", default=None)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()

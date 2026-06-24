#!/usr/bin/env python3
"""Build a manual label-review queue from single-class FP/FN analysis."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "ml" / "yolo_3d"
ERROR_ROOT = (
    SCRIPT_DIR / "experiment_results" / "singleclass_error_analysis_20260622"
)
DATASET_DIR = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/preprocess_v2_20260612/datasets/"
    "ia_rank_residual_intensity_gradient"
)
CLASS_NAMES = ["pinhole", "crap", "spatter"]


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


def clip_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))
    return left, top, right, bottom


def expand_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: int,
    min_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    side = max(x2 - x1, y2 - y1, float(min_size)) + 2.0 * margin
    left = int(round(cx - side * 0.5))
    top = int(round(cy - side * 0.5))
    right = int(round(cx + side * 0.5))
    bottom = int(round(cy + side * 0.5))
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    return max(0, left), max(0, top), max(1, right), max(1, bottom)


def row_box(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        float(row["x1"]),
        float(row["y1"]),
        float(row["x2"]),
        float(row["y2"]),
    )


def class_name(index: int) -> str:
    if 0 <= index < len(CLASS_NAMES):
        return CLASS_NAMES[index]
    return "unknown"


def read_label_polygons(label_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    items: list[tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return items
    for line in label_path.read_text(encoding="utf-8").splitlines():
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
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        if len(points) >= 3:
            items.append((cls, points))
    return items


def draw_label_polygons(image: np.ndarray, label_path: Path) -> None:
    height, width = image.shape[:2]
    colors = [(255, 120, 0), (0, 180, 0), (180, 80, 255)]
    for cls, points in read_label_polygons(label_path, width, height):
        color = colors[cls % len(colors)]
        pts = np.round(points).astype(np.int32)
        cv2.polylines(image, [pts], True, color, 1, cv2.LINE_AA)
        x, y = pts[0]
        cv2.putText(
            image,
            class_name(cls),
            (int(x), max(14, int(y) - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_error_box(image: np.ndarray, row: dict[str, str]) -> None:
    color = (0, 0, 255) if row["kind"] == "fn" else (0, 220, 255)
    left, top, right, bottom = clip_box(row_box(row), image.shape[1], image.shape[0])
    cv2.rectangle(image, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
    text = row["kind"].upper()
    if row["kind"] == "fp":
        text += f" {float(row['score']):.2f}"
    else:
        text += f" {row['original_class']}"
    cv2.putText(
        image,
        text,
        (left, max(18, top - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def priority_reason(row: dict[str, str]) -> str:
    if row["kind"] == "fn":
        if row["original_class"] == "pinhole":
            return "missed_pinhole_verify_visibility"
        return "missed_labeled_defect_verify_label"
    if float(row["score"]) >= 0.50:
        return "high_conf_fp_check_unlabeled_defect"
    return "fp_check_texture_or_edge_confusion"


def build_crop(
    image: np.ndarray,
    row: dict[str, str],
    out_path: Path,
    margin: int,
    min_size: int,
) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    left, top, right, bottom = expand_box(row_box(row), width, height, margin, min_size)
    crop = image[top:bottom, left:right].copy()
    local = dict(row)
    local["x1"] = str(float(row["x1"]) - left)
    local["x2"] = str(float(row["x2"]) - left)
    local["y1"] = str(float(row["y1"]) - top)
    local["y2"] = str(float(row["y2"]) - top)
    draw_error_box(crop, local)
    imwrite(out_path, crop)
    return left, top, right, bottom


def make_full_overlay(
    image_path: Path,
    label_path: Path,
    rows: list[dict[str, str]],
    out_path: Path,
) -> None:
    image = imread(image_path)
    draw_label_polygons(image, label_path)
    for row in rows:
        draw_error_box(image, row)
    header = f"{image_path.name}  GT polygons + FN(red) / FP(yellow)"
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 32), (0, 0, 0), -1)
    cv2.putText(
        image,
        header[:110],
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    imwrite(out_path, image)


def make_sheet(image_paths: list[Path], out_path: Path, thumb_width: int = 420) -> None:
    cells: list[np.ndarray] = []
    for path in image_paths:
        image = imread(path)
        scale = thumb_width / image.shape[1]
        resized = cv2.resize(
            image,
            (thumb_width, max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        cells.append(resized)
    if not cells:
        return
    cols = 3
    cell_h = max(cell.shape[0] for cell in cells)
    cell_w = thumb_width
    rows = (len(cells) + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 32, dtype=np.uint8)
    for index, cell in enumerate(cells):
        row = index // cols
        col = index % cols
        y = row * cell_h
        x = col * cell_w
        sheet[y : y + cell.shape[0], x : x + cell.shape[1]] = cell
    imwrite(out_path, sheet)


def write_report(
    out_path: Path,
    image_rows: list[dict[str, Any]],
    instance_rows: list[dict[str, Any]],
    top_images: int,
) -> None:
    fn = [row for row in instance_rows if row["kind"] == "fn"]
    fp = [row for row in instance_rows if row["kind"] == "fp"]
    pinhole_fn = [row for row in fn if row["original_class"] == "pinhole"]
    high_fp = [row for row in fp if float(row["score"]) >= 0.50]
    lines = [
        "# Label Review Queue",
        "",
        "This queue is generated from the single-class FP/FN validation analysis.",
        "It does not modify the source dataset.",
        "",
        "## Summary",
        "",
        f"- prioritized images: {len(image_rows)}",
        f"- instance crops: {len(instance_rows)}",
        f"- FN crops: {len(fn)}",
        f"- FP crops: {len(fp)}",
        f"- pinhole FN crops: {len(pinhole_fn)}",
        f"- high-confidence FP crops: {len(high_fp)}",
        "",
        "## Suggested Review Order",
        "",
        "1. Review `image_manifest.csv` from top to bottom.",
        "2. For FP crops with reason `high_conf_fp_check_unlabeled_defect`, add a label if the crop is a real defect.",
        "3. For FN crops, verify whether the existing label is visible and accurate. If it is valid, keep it as a hard positive for the next training run.",
        "4. Pay special attention to `missed_pinhole_verify_visibility`; this is the weakest current class group.",
        "",
        f"## Top {min(top_images, len(image_rows))} Images",
        "",
        "| Rank | Image | GT | TP | FP | FN | Score | Overlay |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in image_rows[:top_images]:
        lines.append(
            f"| {row['priority_rank']} | `{row['image']}` | {row['gt']} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | {row['review_score']} | "
            f"`{row['overlay_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `image_manifest.csv`: image-level queue.",
            "- `instance_manifest.csv`: FP/FN crop-level queue.",
            "- `overlays/`: full-image overlays with labels and FP/FN boxes.",
            "- `crops/fn/` and `crops/fp/`: local crops for annotation decisions.",
            "- `top_overlays_sheet.png`: contact sheet of the highest-priority overlays.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-root", type=Path, default=ERROR_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=ERROR_ROOT / "review_queue")
    parser.add_argument("--top-images", type=int, default=24)
    parser.add_argument("--margin", type=int, default=48)
    parser.add_argument("--min-crop-size", type=int, default=144)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    error_rows = read_csv(args.error_root / "error_instances.csv")
    per_image_rows = read_csv(args.error_root / "per_image_errors.csv")
    review_items = [row for row in error_rows if row["kind"] in {"fp", "fn"}]
    by_image: dict[str, list[dict[str, str]]] = {}
    for row in review_items:
        by_image.setdefault(row["image"], []).append(row)

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out_dir / "overlays"
    crop_root = args.out_dir / "crops"
    image_manifest: list[dict[str, Any]] = []
    instance_manifest: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []

    selected_images = per_image_rows[: args.top_images]
    selected_names = {row["image"] for row in selected_images}

    for rank, row in enumerate(selected_images, start=1):
        image_path = args.dataset_dir / "images" / "val" / row["image"]
        label_path = args.dataset_dir / "labels" / "val" / f"{Path(row['image']).stem}.txt"
        overlay_path = overlay_dir / f"{rank:03d}_{row['image']}"
        make_full_overlay(image_path, label_path, by_image.get(row["image"], []), overlay_path)
        overlay_paths.append(overlay_path)
        image_manifest.append(
            {
                "priority_rank": rank,
                **row,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "overlay_path": str(overlay_path.relative_to(args.out_dir)),
            }
        )

    image_rank = {row["image"]: int(row["priority_rank"]) for row in image_manifest}
    queued_items = [
        row
        for row in review_items
        if row["image"] in selected_names
        or (row["kind"] == "fp" and float(row["score"]) >= 0.50)
    ]

    def item_sort_key(row: dict[str, str]) -> tuple[int, int, float, float]:
        rank = image_rank.get(row["image"], 9999)
        if row["kind"] == "fn" and row["original_class"] == "pinhole":
            kind_rank = 0
        elif row["kind"] == "fn":
            kind_rank = 1
        elif float(row["score"]) >= 0.50:
            kind_rank = 2
        else:
            kind_rank = 3
        return (rank, kind_rank, -float(row["score"]), float(row["area"]))

    queued_items.sort(key=item_sort_key)

    for index, row in enumerate(queued_items, start=1):
        image_name = row["image"]
        image_path = args.dataset_dir / "images" / "val" / image_name
        label_path = args.dataset_dir / "labels" / "val" / f"{Path(image_name).stem}.txt"
        image = imread(image_path)
        crop_dir = crop_root / row["kind"]
        crop_name = (
            f"{index:04d}_{Path(image_name).stem}_{row['kind']}_"
            f"iou{float(row['best_mask_iou']):.2f}_score{float(row['score']):.2f}.png"
        )
        crop_path = crop_dir / crop_name
        left, top, right, bottom = build_crop(
            image, row, crop_path, args.margin, args.min_crop_size
        )
        instance_manifest.append(
            {
                "priority_rank": index,
                "image_priority_rank": image_rank.get(image_name, ""),
                "kind": row["kind"],
                "image": image_name,
                "reason": priority_reason(row),
                "score": row["score"],
                "best_mask_iou": row["best_mask_iou"],
                "area": row["area"],
                "original_class": row["original_class"],
                "matched_original_class": row["matched_original_class"],
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
                "crop_left": left,
                "crop_top": top,
                "crop_right": right,
                "crop_bottom": bottom,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "crop_path": str(crop_path.relative_to(args.out_dir)),
            }
        )

    write_csv(args.out_dir / "image_manifest.csv", image_manifest)
    write_csv(args.out_dir / "instance_manifest.csv", instance_manifest)
    make_sheet(overlay_paths[: min(12, len(overlay_paths))], args.out_dir / "top_overlays_sheet.png")
    write_report(
        args.out_dir / "README.md",
        image_manifest,
        instance_manifest,
        args.top_images,
    )
    print(f"wrote {args.out_dir}")
    print(f"images={len(image_manifest)} instances={len(instance_manifest)}")
    print(f"fn={sum(row['kind'] == 'fn' for row in instance_manifest)}")
    print(f"fp={sum(row['kind'] == 'fp' for row in instance_manifest)}")


if __name__ == "__main__":
    main()

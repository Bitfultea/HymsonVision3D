"""Verify the shared ia_rank_residual_intensity_gradient preprocessing output."""

from pathlib import Path
import sys

import cv2
import numpy as np

import pre_process_data


SAMPLE_TIFF = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/rename_tiff/10.tif"
)
TIFF_ROOT = SAMPLE_TIFF.parent
EXPECTED_PNG = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/"
    "yolo_3d/preprocess_v2_20260612/datasets/"
    "ia_rank_residual_intensity_gradient/images/val/10.png"
)
EXPECTED_ROOT = EXPECTED_PNG.parents[2]


def imread(path):
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def render(tiff_path):
    height, intensity = pre_process_data.read_tiff_height_intensity(str(tiff_path))
    return pre_process_data.ia_rank_residual_intensity_gradient_image(
        height, intensity
    )


def compare(tiff_path, expected_png):
    actual = render(tiff_path)
    expected = imread(expected_png)

    if actual.shape != expected.shape:
        raise AssertionError(
            f"{expected_png}: shape {actual.shape} vs {expected.shape}"
        )
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{expected_png}: dtype {actual.dtype} vs {expected.dtype}")

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    return int(np.max(np.abs(diff))), int(np.count_nonzero(diff)), actual.shape


def source_tiff(image_stem):
    raw_stem = image_stem.split(" (", 1)[0]
    for suffix in (".tif", ".tiff"):
        path = TIFF_ROOT / f"{raw_stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"missing source TIFF for image stem: {image_stem}")


def compare_all():
    compared = 0
    max_abs_diff = 0
    nonzero = 0
    for expected_png in sorted((EXPECTED_ROOT / "images").glob("*/*.png")):
        tiff_path = source_tiff(expected_png.stem)
        item_max, item_nonzero, _ = compare(tiff_path, expected_png)
        max_abs_diff = max(max_abs_diff, item_max)
        nonzero += item_nonzero
        compared += 1

    if compared == 0:
        raise RuntimeError(f"no expected PNGs under {EXPECTED_ROOT / 'images'}")
    if max_abs_diff != 0 or nonzero != 0:
        raise AssertionError(
            f"pixel mismatch: compared={compared}, "
            f"max_abs_diff={max_abs_diff}, nonzero={nonzero}"
        )

    print(
        f"ok compared={compared} max_abs_diff={max_abs_diff} nonzero={nonzero}"
    )


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        compare_all()
        return

    sample_tiff = Path(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE_TIFF
    expected_png = Path(sys.argv[2]) if len(sys.argv) > 2 else EXPECTED_PNG

    max_abs_diff, nonzero, shape = compare(sample_tiff, expected_png)
    if max_abs_diff != 0 or nonzero != 0:
        raise AssertionError(
            f"pixel mismatch: max_abs_diff={max_abs_diff}, nonzero={nonzero}"
        )

    print(
        f"ok shape={shape} dtype=uint8 "
        f"max_abs_diff={max_abs_diff} nonzero={nonzero}"
    )


if __name__ == "__main__":
    main()

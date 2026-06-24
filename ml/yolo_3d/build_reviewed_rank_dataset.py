#!/usr/bin/env python3
"""Build a rank-preprocessed dataset with reviewed labels only."""

from __future__ import annotations

import argparse
import csv
import filecmp
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "ml" / "yolo_3d"
BASE_DIR = Path(
    "/home/charles/Data/Dataset/Collected/密封钉/密封钉3D缺陷收集/yolo_3d"
)
SOURCE_DATASET = (
    BASE_DIR
    / "preprocess_v2_20260612"
    / "datasets"
    / "ia_rank_residual_intensity_gradient"
)
DEFAULT_REVIEWED_LABELS = (
    SCRIPT_DIR / "experiment_results" / "label_review_20260623" / "reviewed_labels"
)
DEFAULT_OUT = BASE_DIR / "dataset_rank_reviewed_20260623"


def raw_stem(stem: str) -> str:
    return stem.split(" (", 1)[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reviewed_label_path(reviewed_dir: Path, split: str, label_name: str) -> Path | None:
    stem = Path(label_name).stem
    candidates = [
        reviewed_dir / split / label_name,
        reviewed_dir / label_name,
        reviewed_dir / split / f"{raw_stem(stem)}.txt",
        reviewed_dir / f"{raw_stem(stem)}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def label_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def copy_or_link_images(source_dataset: Path, out_dir: Path, copy_images: bool) -> str:
    source_images = source_dataset / "images"
    out_images = out_dir / "images"
    if copy_images:
        shutil.copytree(source_images, out_images)
        return "copied"

    out_images.symlink_to(source_images, target_is_directory=True)
    return "symlink"


def build_dataset(
    source_dataset: Path,
    reviewed_dir: Path,
    out_dir: Path,
    copy_images: bool,
    force: bool,
) -> None:
    if not source_dataset.exists():
        raise FileNotFoundError(source_dataset)
    if not reviewed_dir.exists():
        raise FileNotFoundError(
            f"{reviewed_dir} does not exist; put reviewed label files there first"
        )
    reviewed_files = list(reviewed_dir.glob("**/*.txt"))
    if not reviewed_files:
        raise FileNotFoundError(
            f"no reviewed .txt labels found in {reviewed_dir}; refusing to build "
            "an unchanged reviewed dataset"
        )
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists; pass --force to replace it")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True)
    image_mode = copy_or_link_images(source_dataset, out_dir, copy_images)
    manifest_rows: list[dict[str, Any]] = []
    changed_count = 0
    override_count = 0

    for split in ("train", "val"):
        src_label_dir = source_dataset / "labels" / split
        out_label_dir = out_dir / "labels" / split
        out_label_dir.mkdir(parents=True, exist_ok=True)
        for src_label in sorted(src_label_dir.glob("*.txt")):
            reviewed_label = reviewed_label_path(reviewed_dir, split, src_label.name)
            chosen_label = reviewed_label or src_label
            out_label = out_label_dir / src_label.name
            shutil.copy2(chosen_label, out_label)
            changed = not filecmp.cmp(src_label, out_label, shallow=False)
            if reviewed_label is not None:
                override_count += 1
            if changed:
                changed_count += 1
            manifest_rows.append(
                {
                    "split": split,
                    "label": src_label.name,
                    "source_label": str(src_label),
                    "chosen_label": str(chosen_label),
                    "review_override": reviewed_label is not None,
                    "changed": changed,
                    "source_count": label_count(src_label),
                    "chosen_count": label_count(chosen_label),
                }
            )

    data_yaml = "\n".join(
        [
            f"path: {out_dir}",
            "train: images/train",
            "val: images/val",
            "nc: 3",
            "names: ['pinhole','crap','spatter']",
            "",
        ]
    )
    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    write_csv(out_dir / "reviewed_label_manifest.csv", manifest_rows)
    readme = [
        "# dataset_rank_reviewed_20260623",
        "",
        "Images are unchanged from `ia_rank_residual_intensity_gradient`.",
        "Only labels are copied or overridden from reviewed label files.",
        "",
        f"- source_dataset: `{source_dataset}`",
        f"- reviewed_labels: `{reviewed_dir}`",
        f"- image_mode: {image_mode}",
        f"- label files: {len(manifest_rows)}",
        f"- reviewed overrides: {override_count}",
        f"- changed labels: {changed_count}",
        "",
        "Use `reviewed_label_manifest.csv` to audit exactly which labels changed.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"wrote {out_dir}")
    print(f"labels={len(manifest_rows)} overrides={override_count} changed={changed_count}")
    print(f"data={out_dir / 'data.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--reviewed-labels-dir", type=Path, default=DEFAULT_REVIEWED_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        args.source_dataset,
        args.reviewed_labels_dir,
        args.out_dir,
        args.copy_images,
        args.force,
    )


if __name__ == "__main__":
    main()

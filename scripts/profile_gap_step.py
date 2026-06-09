#!/usr/bin/env python3
"""Profile gap/step detection over a TIFF folder.

Example:
  python3 scripts/profile_gap_step.py \
      --input-dir /path/to/tiffs \
      --exe ./build/test/bspline_test \
      --ratio 1,1,60 \
      --repeat 3 \
      --out-dir /tmp/gap_step_profile
"""

import argparse
import csv
import os
import re
import statistics
import subprocess
import time
from pathlib import Path


PROFILE_MS_RE = re.compile(r"^\[profile\] ([^:]+):\s*([0-9.]+)\s*ms")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_]+)=([A-Za-z0-9_.+-]+)")
THREAD_SUM_RE = re.compile(
    r"(resample|group|filter|line|measure|plot)=([0-9.]+)\s*ms"
)
STEP_RE = re.compile(r"^(step_height|step_width):\s*([-+0-9.eE]+)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder containing TIFF files.")
    parser.add_argument("--exe", default="./build/test/bspline_test")
    parser.add_argument("--ratio", default="1,1,60")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument(
        "--debug-dir",
        default="/tmp/gap_step_profile_debug",
        help="Passed to bspline_test; debug output is disabled by --no-debug.",
    )
    return parser.parse_args()


def list_tiffs(input_dir):
    root = Path(input_dir)
    files = []
    for suffix in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        files.extend(root.rglob(suffix))
    return sorted(set(files))


def parse_output(stdout, stderr):
    row = {}
    for line in stdout.splitlines():
        match = STEP_RE.match(line.strip())
        if match:
            row[match.group(1)] = float(match.group(2))

    for line in stderr.splitlines():
        line = line.strip()
        match = PROFILE_MS_RE.match(line)
        if match:
            key = match.group(1).replace(".", "_")
            row[f"{key}_ms"] = float(match.group(2))
            continue
        if line.startswith("[profile] bspline_interpolation_dll2:"):
            for key, value in KEY_VALUE_RE.findall(line):
                try:
                    row[key] = int(value)
                except ValueError:
                    row[key] = value
            continue
        if line.startswith("[profile] bspline_interpolation_dll2.thread_sum:"):
            for key, value in THREAD_SUM_RE.findall(line):
                row[f"thread_{key}_ms"] = float(value)
            continue
        if line.startswith("[profile] fast_path fallback reasons:"):
            for key, value in KEY_VALUE_RE.findall(line):
                row[f"fallback_{key}"] = int(value)
    return row


def run_one(exe, tiff_path, ratio, debug_dir, omp_threads):
    env = os.environ.copy()
    env["HYMSON3D_PROFILE_BSPLINE"] = "1"
    env["OMP_NUM_THREADS"] = str(omp_threads)
    cmd = [exe, str(tiff_path), ratio, debug_dir, "--no-debug"]
    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        input="\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    row = parse_output(completed.stdout, completed.stderr)
    row.update(
        {
            "file": str(tiff_path),
            "return_code": completed.returncode,
            "elapsed_ms": elapsed_ms,
        }
    )
    return row


def median_or_blank(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(values) if values else ""


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tiffs = list_tiffs(args.input_dir)
    if not tiffs:
        raise SystemExit(f"No TIFF files found under {args.input_dir}")

    raw_rows = []
    for tiff in tiffs:
        for repeat_idx in range(args.repeat):
            row = run_one(args.exe, tiff, args.ratio, args.debug_dir, args.omp_threads)
            row["repeat"] = repeat_idx
            row["ratio"] = args.ratio
            row["omp_threads"] = args.omp_threads
            raw_rows.append(row)
            print(
                f"{tiff.name} repeat={repeat_idx} rc={row['return_code']} "
                f"detect={row.get('detect_gap_step_dll_plot2_impl_ms', '')} ms "
                f"width={row.get('step_width', '')}"
            )

    all_keys = sorted({key for row in raw_rows for key in row.keys()})
    raw_csv = out_dir / "raw_runs.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_rows = []
    for tiff in tiffs:
        rows = [r for r in raw_rows if r["file"] == str(tiff)]
        summary_rows.append(
            {
                "file": str(tiff),
                "runs": len(rows),
                "return_code_max": max(r["return_code"] for r in rows),
                "elapsed_ms_median": median_or_blank(rows, "elapsed_ms"),
                "detect_ms_median": median_or_blank(
                    rows, "detect_gap_step_dll_plot2_impl_ms"
                ),
                "bspline_ms_median": median_or_blank(
                    rows, "bspline_interpolation_dll2_ms"
                ),
                "resample_thread_ms_median": median_or_blank(
                    rows, "thread_resample_ms"
                ),
                "fast_hit_median": median_or_blank(rows, "fast_hit"),
                "fast_miss_median": median_or_blank(rows, "fast_miss"),
                "raw_grid_hit_median": median_or_blank(rows, "raw_grid_hit"),
                "raw_grid_miss_median": median_or_blank(rows, "raw_grid_miss"),
                "raw_grid_enabled_median": median_or_blank(
                    rows, "raw_grid_enabled"
                ),
                "step_height_median": median_or_blank(rows, "step_height"),
                "step_width_median": median_or_blank(rows, "step_width"),
            }
        )

    summary_csv = out_dir / "summary_by_file.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {raw_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()

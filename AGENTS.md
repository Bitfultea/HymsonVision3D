# Repository Guidelines

## Project Structure & Module Organization

HymsonVision3D is a C++17 industrial 3D defect detection library for turbine blade depth maps and point clouds. Module flow is `utility -> geometry -> core -> ml -> pipeline`.

- `utility/`: logging, file I/O, math helpers, Eigen utilities.
- `geometry/`: 2D surface fitting and 3D point cloud geometry.
- `core/`: conversion, normals, curvature, clustering, filtering, raster, calibration, and features.
- `pipeline/`: high-level inspection algorithms such as defect detection, disk level measurement, gap/step detection, and camera calibration.
- `ml/`: libtorch/CUDA-backed 3D ML integration, including YOLO-3D scripts and export tools.
- `test/`: C++ test/demo executables and related fixtures.
- `tools/` and `scripts/`: Python utilities for data, labels, binary conversion, and analysis.
- `thirdparty/`: vendored CGAL, libtorch, Open3D-derived code, and nanoflann.

## Build, Test, and Development Commands

Configure and build from the repository root:

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --parallel
```

Build a single target when iterating:

```bash
cmake --build build --target def_test --parallel
```

Run test executables directly:

```bash
./build/test/def_test
./build/test/planar_test
./build/test/bspline_test
./build/test/calib_test
```

`CMAKE_BUILD_TYPE` defaults to `Release`; pass `-DCMAKE_BUILD_TYPE=Debug` during configure when needed. Linux CUDA architecture is set to `86` for RTX 3090-class GPUs.

## Coding Style & Naming Conventions

Use `.clang-format`: Google base style, 4-space indentation, no tabs, 80-column limit, sorted includes, and C++17. Keep headers and implementations paired as `Name.h` and `Name.cpp` where practical. Existing types use PascalCase (`PointCloud`, `DefectDetection`); functions and locals generally use lower snake case or the surrounding file style.

## Testing Guidelines

Add C++ checks under `test/` and register them in `test/CMakeLists.txt` as executable targets linked to the relevant libraries. Prefer names matching current targets, such as `def_test`, `calib_test`, or `ml_test`. For Python utilities, keep validation close to the tool or add a focused test script under `test/`.

## Commit & Pull Request Guidelines

Recent commits commonly use bracketed prefixes such as `[feat]`, `[fix]`, `[docs]`, `[update]`, and `[optimization]`. Keep messages imperative and specific, for example `[fix] handle noisy TIFF depth maps`.

Pull requests should include the problem, change summary, build/test commands run, and GPU/CUDA or dependency assumptions. Attach screenshots or sample outputs when visual inspection or generated data changes.

## Security & Configuration Tips

Do not commit generated build trees, local model exports, private datasets, or machine-specific paths. Review changes to CUDA, vcpkg, libtorch, and `thirdparty/` paths carefully because they affect reproducibility across Linux and Windows.

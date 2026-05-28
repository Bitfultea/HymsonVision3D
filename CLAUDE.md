# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
# Configure (from repo root)
mkdir -p build && cd build
cmake ..

# Build all
cmake --build . --parallel

# Build specific target
cmake --build . --target def_test --parallel
```

Build type defaults to `Release` (`set(CMAKE_BUILD_TYPE Release)` in `CMakeLists.txt`). Pass `-DCMAKE_BUILD_TYPE=Debug` to cmake configure for debug mode. Debug builds add `-fsanitize=address` on Linux.

On Linux, CUDA architecture is hardcoded to `86` (RTX 3090). Change `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` if using a different GPU.

## Running Tests

Test executables are built into `build/test/`. Run them directly:

```bash
./build/test/def_test        # defect detection
./build/test/planar_test     # plane fitting
./build/test/bspline_test    # B-spline interpolation
./build/test/calib_test      # camera calibration
./build/test/ml_test         # ML inference (libtorch)
./build/test/tiff2ply        # TIFF to PLY conversion
./build/test/CGAL_test       # CGAL integration
./build/test/LoggerTest      # logging
./build/test/TorchTest       # libtorch basics
./build/test/TorchTest2      # libtorch extended
./build/test/ConvertTiff     # TIFF conversion
./build/test/test_plana_2    # plane detection v2
```

Or via CTest from the build directory:

```bash
cd build && ctest
```

## Architecture

C++17 industrial 3D defect detection library for turbine blade inspection. Processes depth maps (TIFF) into point clouds and runs geometric analysis pipelines.

Everything lives under the `hymson3d` namespace with sub-namespaces matching the module structure: `hymson3d::geometry`, `hymson3d::core` (plus `hymson3d::core::converter`), `hymson3d::pipeline`, `hymson3d::utility`.

### Module dependency order

```
utility → geometry (2D/3D) → core → ml → pipeline
```

- `utility/` — logging (spdlog singleton via `LOG_TRACE/DEBUG/INFO/WARN/ERROR` macros), file I/O, math helpers, Eigen utilities
- `geometry/` — abstract `Geometry` base with `GeometryType` enum; `Geometry2D` → `Surface` (quadratic surface fitting); `Geometry3D` → `PointCloud` (points, normals, colors, intensities, labels, curvatures), `KDTree` (nanoflann-backed), `Plane`, `Mesh`, `Line3D`, bounding boxes
- `core/` — `converter` namespace (TIFF↔PointCloud↔PCL↔CGAL), `Normal`, `Curvature`, `PlaneDetection`, `Cluster`, `Filter`, `Feature`, `Distance`, `Raster`, `CameraCalib`
- `pipeline/` — high-level inspection: `DefectDetection`, `DiskLevelMeasurement`, `GapStepDetection`, `CalibCamera`, `PlanarDegree`, `Preprocess`, `SegmentationMFD`, `ImageTemplateMatch`
- `ml/` — `pointpillar/` and `yolo_3d/` inference via libtorch; CUDA arch 86
- `thirdparty/` — vendored CGAL 5.6.2, libtorch, open3d headers, PoissonRecon, Möller–Trumbore intersection
- `tools/` — Python utilities for data generation, labeling, binary conversion
- `scripts/` — Python helper scripts (e.g. plane angle computation)

### PointCloud is the central data structure

`geometry::PointCloud` (typedef `Ptr` = `std::shared_ptr<PointCloud>`) is the shared currency across the entire pipeline. It carries:
- `points_`, `normals_`, `colors_`, `intensities_`, `labels_`, `curvatures_`, `covariances_`
- Slice data: `y_slices_`, `x_slices_` and associated normal/index vectors (populated by pipeline slicing methods)
- `width_`/`height_` from TIFF conversion origin

Pipeline methods read/write labels and slice data directly on the PointCloud.

### Key data flow

1. TIFF depth map → `converter::tiff_to_pointcloud()` → `PointCloud::Ptr`
2. `Normal::compute()` + `Curvature::compute()` on the cloud
3. Pipeline static methods (`DefectDetection`, `GapStepDetection`, etc.) operate on the cloud with `KDTreeSearchParam` (KNN/Radius/Hybrid)
4. Results stored as labels/clusters/slice data on the `PointCloud`

### DLL export pattern

Many pipeline methods have `_dll` suffix variants (e.g. `detect_pinholes_nva_dll`, `detect_gap_step_dll`). These accept output references (`std::vector<PointCloud::Ptr>&`, `double&`) and `std::string& debug_path` for external API usage, instead of writing to the input cloud directly.

### Geometry base hierarchy

```
Geometry (abstract, GeometryType enum)
├── Geometry2D → Surface
└── Geometry3D → PointCloud, Plane, Mesh, Line3D
```

### External dependencies (Linux)

OpenCV 4.8, Boost, spdlog, Eigen3, fmt, PCL, CGAL 5.6.2 (vendored), nanoflann, CUDA Toolkit, libtorch (vendored), qhull.

## Coding Style

`.clang-format` at repo root: Google base style, 4-space indent, no tabs, 80-column limit, sorted includes, C++17. Types use PascalCase (`PointCloud`, `DefectDetection`); functions and locals use lower snake_case. Headers paired with matching `.cpp` files as `Name.h` / `Name.cpp`.

## Commit Conventions

Commit messages use bracketed prefixes: `[feat]`, `[fix]`, `[docs]`, `[update]`, `[optimization]`. Keep messages imperative and specific (e.g. `[fix] handle noisy TIFF depth maps`).

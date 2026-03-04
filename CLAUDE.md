# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build

```bash
# Configure (from repo root)
mkdir -p build && cd build
cmake ..

# Build
cmake --build . --parallel

# Build specific target
cmake --build . --target def_test --parallel
```

Build type is set to `Release` by default in `CMakeLists.txt`. To change it, edit `set(CMAKE_BUILD_TYPE Release)` or pass `-DCMAKE_BUILD_TYPE=Debug` on the cmake configure line.

On Linux, CUDA architecture is hardcoded to `86` (RTX 3090). Change `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` if using a different GPU.

## Running Tests

Test executables are built into `build/test/`. Run them directly:

```bash
./build/test/def_test
./build/test/planar_test
./build/test/bspline_test
./build/test/calib_test
```

Or via CTest from the build directory:

```bash
cd build && ctest
```

## Architecture

The project is a C++17 industrial 3D defect detection library for turbine blade inspection. It processes depth maps (TIFF) into point clouds and runs geometric analysis pipelines.

### Module dependency order

```
utility → geometry (2D/3D) → core → ml → pipeline
```

- `utility/` — logging (spdlog wrapper), file I/O, math helpers, Eigen utilities
- `geometry/2D/` — `Surface`: quadratic surface fitting over height maps
- `geometry/3D/` — `PointCloud`: stores points, normals, colors, intensities, labels, curvatures
- `core/` — `Converter` (TIFF↔PointCloud↔PCL↔CGAL), `Normal`, `Curvature`, `PlaneDetection`, `Cluster`, `Filter`, `Feature`, `Distance`, `Raster`, `CameraCalib`
- `pipeline/` — high-level detection algorithms: `DefectDetection`, `DiskLevelMeasurement`, `GapStepDetection`, `CalibCamera`
- `ml/` — `pointpillar/` and `yolo_3d/` inference via libtorch; CUDA arch 86
- `thirdparty/` — vendored CGAL 5.6.2, libtorch, open3d headers, PoissonRecon, Möller–Trumbore intersection

### Key data flow

1. TIFF depth map → `Converter::tiff2PointCloud()` → `PointCloud`
2. `Normal::compute()` + `Curvature::compute()` on the cloud
3. `DefectDetection` static methods operate on the cloud with KDTree search params
4. Results are clusters/labels on the `PointCloud`

### DefectDetection methods

- `detect_defects()` — convex/concave defect detection using normal deviation
- `detect_pinholes()` / `detect_pinholes_nva()` — pinhole detection (NVA variant uses FPFH features)
- `detect_smooth_surface()` — fits a quadratic `Surface` as reference, detects deviations
- `detect_CSAD()` — CSAD-based detection
- `detect_central_ring()` — ring detection by depth threshold

### Geometry base hierarchy

```
Geometry (abstract, GeometryType enum)
├── Geometry2D → Surface
└── Geometry3D → PointCloud
```

### External dependencies (Linux, installed system-wide or via package manager)

OpenCV 4.8, Boost, spdlog, Eigen3, fmt, PCL, CGAL 5.6.2 (vendored), nanoflann, CUDA Toolkit, libtorch (vendored in `thirdparty/`)

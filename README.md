# HymsonVision3D

Industrial 3D defect detection library for turbine blade inspection. Processes depth maps (TIFF) into point clouds and runs geometric analysis pipelines for surface anomaly detection.

## Architecture

```
utility → geometry (2D/3D) → core → ml → pipeline
```

| Module | Description | Dependencies |
|--------|-------------|-------------|
| `utility/` | Logging (spdlog), file I/O, Eigen helpers, math (RPCA, SVD) | spdlog, fmt, Eigen3, OpenCV, PCL |
| `geometry/` | 2D surface fitting, 3D point cloud/KDTree/mesh data structures | utility, CGAL, nanoflann |
| `core/` | TIFF↔PointCloud conversion, normal/curvature estimation, plane detection, clustering, filtering, FPFH, rasterization, camera calibration | utility, geometry |
| `ml/` | YOLO-3D inference via TensorRT (CUDA), PointPillar via libtorch | CUDA, TensorRT, libtorch |
| `pipeline/` | Defect detection, gap/step measurement, disk level measurement, ML-based segmentation | core, ml |

## Features

- **TIFF to Point Cloud**: Depth map conversion with configurable x/y/z scaling, intensity map support
- **Normal Estimation**: PCA, PCL (OMP), JET, VCM methods; orientation toward positive Z
- **Curvature Computation**: PCA-based, PCL-based, surface variation, triangle normal variation
- **Defect Detection**:
  - Convex/concave defects via normal deviation + DBSCAN + B-Spline + RPCA
  - Pinhole detection via curvature filtering and FPFH + normal vector analysis (NVA)
  - Smooth surface anomaly detection via quadratic surface fitting or normal/curvature thresholding
  - CSAD-based detection
- **Surface Fitting**: Quadratic surface (`z = ax² + by² + cxy + dx + ey + f`) via normal equations or iterative outlier rejection; free-form surface via median blur
- **Clustering**: Region growing (PCL/custom), DBSCAN, planar-constrained clustering
- **Filtering**: Voxel grid, uniform, random downsampling; radius/statistical outlier removal; axis-aligned pass-through
- **ML Inference**: YOLO-3D segmentation via TensorRT engine; PointPillar 3D object detection via libtorch
- **Camera Calibration**: Chessboard, Charuco, circles grid with reprojection error evaluation
- **Plane Detection**: SVD fitting, RANSAC, disk/level measurement
- **Gap & Step Detection**: B-Spline interpolation, line segment extraction, derivative grouping
- **Planarity Measurement**: Signed distance range against fitted plane
- **Rasterization**: Orthographic projection of point cloud to 2D image frame with multiple modes
- **CGAL Integration**: Delaunay triangulation to Surface_mesh
- **OpenPCDet Data Pipeline**: Python tools for converting TIFF→NPY→KITTI bin with label generation

## Build

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --parallel
```

Build type defaults to `Release`. Override with `-DCMAKE_BUILD_TYPE=Debug`.

CUDA architecture is hardcoded to `86` (RTX 3090). Edit `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` for other GPUs.

### libtorch Setup (Linux)

```bash
./env.sh  # downloads libtorch 2.5.1 (CUDA 12.4) to thirdparty/libtorch/
```

## Tests

```bash
./build/test/def_test           # defect detection pipeline
./build/test/planar_test        # plane detection
./build/test/bspline_test       # B-Spline fitting
./build/test/calib_test         # camera calibration
./build/test/test_tiff2ply      # TIFF to PLY conversion
./build/test/test_torch         # libtorch inference
./build/test/test_ml            # ML inference
```

Or from build directory: `cd build && ctest`.

## Data Flow

```
TIFF depth map
    │
    ▼ core::converter::tiff_to_pointcloud()
    │
    ▼ geometry::PointCloud (points, normals, colors, intensities, curvatures, labels)
    │
    ▼ core::Normal::compute() + Curvature::compute()
    │
    ▼ pipeline::DefectDetection::detect_*()
    │
    ▼ Labeled PointCloud with defect clusters → PLY output
```

## Dependencies

| Dependency | Version | Notes |
|------------|---------|-------|
| OpenCV | 4.8 | Image I/O, matrix ops |
| Boost | system | Filesystem, thread |
| spdlog | latest | Logging |
| Eigen3 | latest | Linear algebra |
| fmt | latest | String formatting |
| PCL | latest | Point cloud algorithms |
| CGAL | 5.6.2 | Vendored in `thirdparty/` |
| nanoflann | latest | KD-tree (header-only) |
| CUDA Toolkit | 12.x | GPU inference |
| libtorch | 2.5.1 | Vendored in `thirdparty/` |
| libqhull | latest | Convex hull |
| TensorRT | latest | YOLO-3D inference |
| Open3D | partial | Vendored headers for AABB/OBB |

## Python Data Tools (tools/)

Scripts for preparing OpenPCDet training data from TIFF depth maps:

| Script | Purpose |
|--------|---------|
| `gen_data.py` | TIFF → NPY point cloud conversion |
| `gen_label.py` | Labeled PLY → OpenPCDet label TXT |
| `gen_list.py` | File list generation |
| `convert_bin.py` | NPY → KITTI BIN format |

## Windows Support

The project cross-compiles for Windows via MSVC + vcpkg. See `CMakeLists.txt` for Windows-specific configuration (vcpkg toolchain, DLL export, `/bigobj` flag).

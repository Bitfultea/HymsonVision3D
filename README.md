# HymsonVision3D

Industrial 3D defect detection library for turbine blade inspection. Processes depth maps (TIFF) into point clouds and runs geometric analysis pipelines.

## Features

- **Point Cloud Processing**: TIFF depth map conversion, normal/curvature computation, filtering
- **Defect Detection**: Convex/concave defects, pinholes, smooth surface anomalies, CSAD-based detection
- **Surface Fitting**: Quadratic and free-form surface fitting for reference generation
- **ML Integration**: PointPillar and YOLO-3D inference via libtorch

## Build

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --parallel
```

Build type defaults to `Release`. Change via `CMAKE_BUILD_TYPE` in CMakeLists.txt or `-DCMAKE_BUILD_TYPE=Debug`.

CUDA architecture is hardcoded to `86` (RTX 3090). Modify `CMAKE_CUDA_ARCHITECTURES` for other GPUs.

## Dependencies

- OpenCV 4.8
- Boost
- spdlog
- Eigen3
- fmt
- PCL
- CGAL 5.6.2 (vendored)
- nanoflann
- CUDA Toolkit
- libtorch (vendored in `thirdparty/`)
- libqhull

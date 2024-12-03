#pragma once
#include "3D/Plane.h"
#include "3D/PointCloud.h"
#include "Normal.h"

namespace hymson3d {
namespace core {
class PlaneDetection {
public:
    PlaneDetection() = default;
    ~PlaneDetection() = default;

public:
    // Fitting a plane to many points in 3D
    // From https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
    geometry::Plane::Ptr fit_a_plane(geometry::PointCloud& pointcloud);

public:
    double search_radius = 0.1;
};

}  // namespace core
}  // namespace hymson3d
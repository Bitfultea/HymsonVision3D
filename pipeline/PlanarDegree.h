#pragma once

#include "Normal.h"
#include "PlaneDetection.h"
#include "fmtfallback.h"

namespace hymson3d {
namespace pipeline {
class PlanarDegree {
public:
    PlanarDegree() = default;
    ~PlanarDegree() = default;

public:
    double compute_planar_degree(geometry::PointCloud& pointcloud);
};

}  // namespace pipeline
}  // namespace hymson3d
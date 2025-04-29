#pragma once
#include "fmtfallback.h"
#include "3D/KDtree.h"
#include "3D/Plane.h"
#include "3D/PointCloud.h"
#include "Eigen/Core"

namespace hymson3d {
namespace core {
namespace feature {

Eigen::MatrixXf compute_fpfh(geometry::PointCloud& cloud);

double compute_fpfh_density(int idx,
                            Eigen::MatrixXf& fpfh_data,
                            geometry::KDTree& kdtree,
                            int knn);
}  // namespace feature
}  // namespace core
}  // namespace hymson3d
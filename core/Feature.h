#pragma once
#include "3D/KDtree.h"
#include "3D/Plane.h"
#include "3D/PointCloud.h"
#include "Eigen/Core"
#include "fmtfallback.h"

namespace hymson3d {
namespace core {
namespace feature {

Eigen::MatrixXf compute_fpfh(geometry::PointCloud& cloud);

double compute_fpfh_density(int idx,
                            Eigen::MatrixXf& fpfh_data,
                            geometry::KDTree& kdtree,
                            int knn);

std::pair<bool, cv::Point2f> detect_green_ring(const cv::Mat& img,
                                               bool debug_mode = false);

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
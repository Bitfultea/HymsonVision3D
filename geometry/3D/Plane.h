#pragma once
#include <Eigen/Core>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

#include "Geometry3D.h"

namespace hymson3d {
namespace geometry {

class Plane {
public:
    typedef std::shared_ptr<Plane> Ptr;
    Plane() = default;
    ~Plane() = default;

public:
    void compute_orthogonal_basis();
    void orient_normals_towards_positive_z();

private:
    void project_to_frame();

public:
    Eigen::Vector3d center_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_ = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> inlier_points_;
    std::vector<size_t> inlier_idx_;
    Eigen::Matrix3d orth_basis = Eigen::Matrix3d::Zero();
    Eigen::Vector4d coeff_ = Eigen::Vector4d::Zero();
    cv::Mat plane_frame_;
};

}  // namespace geometry
}  // namespace hymson3d
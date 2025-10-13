#include "PointCloud.h"

namespace hymson3d {
namespace geometry {
PointCloud &PointCloud::Clear() {
    points_.clear();
    normals_.clear();
    colors_.clear();
    intensities_.clear();
    covariances_.clear();
    return *this;
}

bool PointCloud::IsEmpty() const { return !HasPoints(); }

Eigen::Vector3d PointCloud::GetMinBound() const {
    return ComputeMinBound(points_);
}

Eigen::Vector3d PointCloud::GetMaxBound() const {
    return ComputeMaxBound(points_);
}

Eigen::Vector3d PointCloud::GetExtend() const {
    auto max_bound = GetMaxBound();
    auto min_bound = GetMinBound();
    return max_bound - min_bound;
}

Eigen::Vector3d PointCloud::GetCenter() const { return ComputeCenter(points_); }

AABB PointCloud::GetAxisAlignedBoundingBox() const {
    return open3d::geometry::AxisAlignedBoundingBox::CreateFromPoints(points_);
}

OBB PointCloud::GetOrientedBoundingBox(bool robust) const {
    return open3d::geometry::OrientedBoundingBox::CreateFromPoints(points_,
                                                                   robust);
}

OBB PointCloud::GetMinimalOrientedBoundingBox(bool robust) const {
    return open3d::geometry::OrientedBoundingBox::CreateFromPointsMinimal(
            points_, robust);
}

PointCloud &PointCloud::Transform(const Eigen::Matrix4d &transformation) {
    TransformPoints(transformation, points_);
    TransformNormals(transformation, normals_);
    TransformCovariances(transformation, covariances_);
    return *this;
}

PointCloud &PointCloud::Translate(const Eigen::Vector3d &translation,
                                  bool relative) {
    TranslatePoints(translation, points_, relative);
    return *this;
}

PointCloud &PointCloud::Scale(const double scale,
                              const Eigen::Vector3d &center) {
    ScalePoints(scale, points_, center);
    return *this;
}

PointCloud &PointCloud::Rotate(const Eigen::Matrix3d &R,
                               const Eigen::Vector3d &center) {
    RotatePoints(R, points_, center);
    RotateNormals(R, normals_);
    RotateCovariances(R, covariances_);
    return *this;
}

PointCloud &PointCloud::operator+=(const PointCloud &cloud) {
    // We do not use std::vector::insert to combine std::vector because it will
    // crash if the pointcloud is added to itself.
    if (cloud.IsEmpty()) return (*this);
    size_t old_vert_num = points_.size();
    size_t add_vert_num = cloud.points_.size();
    size_t new_vert_num = old_vert_num + add_vert_num;
    if ((!HasPoints() || HasNormals()) && cloud.HasNormals()) {
        normals_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            normals_[old_vert_num + i] = cloud.normals_[i];
    } else {
        normals_.clear();
    }
    if ((!HasPoints() || HasColors()) && cloud.HasColors()) {
        colors_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            colors_[old_vert_num + i] = cloud.colors_[i];
    } else {
        colors_.clear();
    }
    if ((!HasPoints() || HasCovariances()) && cloud.HasCovariances()) {
        covariances_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            covariances_[old_vert_num + i] = cloud.covariances_[i];
    } else {
        covariances_.clear();
    }
    if ((!HasPoints() || HasIntensities()) && cloud.HasIntensities()) {
        intensities_.resize(new_vert_num);
        for (size_t i = 0; i < add_vert_num; i++)
            intensities_[old_vert_num + i] = cloud.intensities_[i];
    } else {
        intensities_.clear();
    }
    points_.resize(new_vert_num);
    for (size_t i = 0; i < add_vert_num; i++)
        points_[old_vert_num + i] = cloud.points_[i];
    return (*this);
}

PointCloud PointCloud::operator+(const PointCloud &cloud) const {
    return (PointCloud(*this) += cloud);
}

}  // namespace geometry
}  // namespace hymson3d
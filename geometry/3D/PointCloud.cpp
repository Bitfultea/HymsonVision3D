#include "PointCloud.h"

namespace hymson3d {
namespace geometry {
PointCloud &PointCloud::Clear() {
    points_.clear();
    normals_.clear();
    colors_.clear();
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

Eigen::Vector3d PointCloud::GetCenter() const { return ComputeCenter(points_); }

// AxisAlignedBoundingBox PointCloud::GetAxisAlignedBoundingBox() const {
//     return AxisAlignedBoundingBox::CreateFromPoints(points_);
// }

// OrientedBoundingBox PointCloud::GetOrientedBoundingBox(bool robust) const {
//     return OrientedBoundingBox::CreateFromPoints(points_, robust);
// }

// OrientedBoundingBox PointCloud::GetMinimalOrientedBoundingBox(
//         bool robust) const {
//     return OrientedBoundingBox::CreateFromPointsMinimal(points_, robust);
// }

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

}  // namespace geometry
}  // namespace hymson3d
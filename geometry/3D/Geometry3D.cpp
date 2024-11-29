#include "Geometry3D.h"

namespace hymson3d {
namespace geometry {

Geometry3D& Geometry3D::Rotate(const Eigen::Matrix3d& R) {
    return Rotate(R, GetCenter());
}

Eigen::Vector3d Geometry3D::ComputeMinBound(
        const std::vector<Eigen::Vector3d>& points) const {
    if (points.empty()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
            points.begin(), points.end(), points[0],
            [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a.array().min(b.array()).matrix();
            });
}

Eigen::Vector3d Geometry3D::ComputeMaxBound(
        const std::vector<Eigen::Vector3d>& points) const {
    if (points.empty()) {
        return Eigen::Vector3d(0.0, 0.0, 0.0);
    }
    return std::accumulate(
            points.begin(), points.end(), points[0],
            [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a.array().max(b.array()).matrix();
            });
}

Eigen::Vector3d Geometry3D::ComputeCenter(
        const std::vector<Eigen::Vector3d>& points) const {
    Eigen::Vector3d center(0, 0, 0);
    if (points.empty()) {
        return center;
    }
    center = std::accumulate(points.begin(), points.end(), center);
    center /= double(points.size());
    return center;
}

void Geometry3D::ResizeAndPaintUniformColor(
        std::vector<Eigen::Vector3d>& colors,
        const size_t size,
        const Eigen::Vector3d& color) const {
    colors.resize(size);
    Eigen::Vector3d clipped_color = color;
    if (color.minCoeff() < 0 || color.maxCoeff() > 1) {
        LOG_WARN("invalid color in PaintUniformColor, clipping to [0, 1]");
        clipped_color = clipped_color.array()
                                .max(Eigen::Vector3d(0, 0, 0).array())
                                .matrix();
        clipped_color = clipped_color.array()
                                .min(Eigen::Vector3d(1, 1, 1).array())
                                .matrix();
    }
    for (size_t i = 0; i < size; i++) {
        colors[i] = clipped_color;
    }
}
void Geometry3D::TransformPoints(const Eigen::Matrix4d& transformation,
                                 std::vector<Eigen::Vector3d>& points) const {
    for (auto& point : points) {
        Eigen::Vector4d new_point =
                transformation *
                Eigen::Vector4d(point(0), point(1), point(2), 1.0);
        point = new_point.head<3>() / new_point(3);
    }
}

void Geometry3D::TransformNormals(const Eigen::Matrix4d& transformation,
                                  std::vector<Eigen::Vector3d>& normals) const {
    for (auto& normal : normals) {
        Eigen::Vector4d new_normal =
                transformation *
                Eigen::Vector4d(normal(0), normal(1), normal(2), 0.0);
        normal = new_normal.head<3>();
    }
}

void Geometry3D::TransformCovariances(
        const Eigen::Matrix4d& transformation,
        std::vector<Eigen::Matrix3d>& covariances) const {
    RotateCovariances(transformation.block<3, 3>(0, 0), covariances);
}

void Geometry3D::TranslatePoints(const Eigen::Vector3d& translation,
                                 std::vector<Eigen::Vector3d>& points,
                                 bool relative) const {
    Eigen::Vector3d transform = translation;
    if (!relative) {
        transform -= ComputeCenter(points);
    }
    for (auto& point : points) {
        point += transform;
    }
}

void Geometry3D::ScalePoints(const double scale,
                             std::vector<Eigen::Vector3d>& points,
                             const Eigen::Vector3d& center) const {
    for (auto& point : points) {
        point = (point - center) * scale + center;
    }
}

void Geometry3D::RotatePoints(const Eigen::Matrix3d& R,
                              std::vector<Eigen::Vector3d>& points,
                              const Eigen::Vector3d& center) const {
    for (auto& point : points) {
        point = R * (point - center) + center;
    }
}

void Geometry3D::RotateNormals(const Eigen::Matrix3d& R,
                               std::vector<Eigen::Vector3d>& normals) const {
    for (auto& normal : normals) {
        normal = R * normal;
    }
}

/// The only part that affects the covariance is the rotation part. For more
/// information on variance propagation please visit:
/// https://en.wikipedia.org/wiki/Propagation_of_uncertainty
void Geometry3D::RotateCovariances(
        const Eigen::Matrix3d& R,
        std::vector<Eigen::Matrix3d>& covariances) const {
    for (auto& covariance : covariances) {
        covariance = R * covariance * R.transpose();
    }
}

}  // namespace geometry
}  // namespace hymson3d
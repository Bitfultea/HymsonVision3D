//
//  this file is part of Open3D: www.open3d.org
//
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <numeric>
#include <vector>

#include "Geometry.h"

namespace hymson3d {
namespace geometry {

class AxisAlignedBoundingBox;
class OrientedBoundingBox;

class Geometry3D : public Geometry {
public:
    ~Geometry3D() override {}

protected:
    Geometry3D(GeometryType type) : Geometry(type, 3) {}

public:
    Geometry3D& Clear() override = 0;
    bool IsEmpty() const override = 0;
    virtual Eigen::Vector3d GetMaxBound() const = 0;
    virtual Eigen::Vector3d GetMinBound() const = 0;
    virtual Eigen::Vector3d GetCenter() const = 0;

    virtual Geometry3D& Transform(const Eigen::Matrix4d& transformation) = 0;
    virtual Geometry3D& Translate(const Eigen::Vector3d& translation,
                                  bool relative = true) = 0;
    virtual Geometry3D& Scale(const double scale,
                              const Eigen::Vector3d& center) = 0;
    virtual Geometry3D& Rotate(const Eigen::Matrix3d& R,
                               const Eigen::Vector3d& center) = 0;
    virtual Geometry3D& Rotate(const Eigen::Matrix3d& R);

    //     virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const = 0;

    //     virtual OrientedBoundingBox GetOrientedBoundingBox(
    //             bool robust = false) const = 0;
    //     virtual OrientedBoundingBox GetMinimalOrientedBoundingBox(
    //             bool robust = false) const = 0;

protected:
    Eigen::Vector3d ComputeMinBound(
            const std::vector<Eigen::Vector3d>& points) const;
    Eigen::Vector3d ComputeMaxBound(
            const std::vector<Eigen::Vector3d>& points) const;
    Eigen::Vector3d ComputeCenter(
            const std::vector<Eigen::Vector3d>& points) const;
    void ResizeAndPaintUniformColor(std::vector<Eigen::Vector3d>& colors,
                                    const size_t size,
                                    const Eigen::Vector3d& color) const;
    void TransformPoints(const Eigen::Matrix4d& transformation,
                         std::vector<Eigen::Vector3d>& points) const;

    /// \brief Transforms the normals with the transformation matrix.
    void TransformNormals(const Eigen::Matrix4d& transformation,
                          std::vector<Eigen::Vector3d>& normals) const;

    /// \brief Transforms all covariance matrices with the transformation.
    void TransformCovariances(const Eigen::Matrix4d& transformation,
                              std::vector<Eigen::Matrix3d>& covariances) const;

    /// \brief Apply translation to the geometry coordinates.
    void TranslatePoints(const Eigen::Vector3d& translation,
                         std::vector<Eigen::Vector3d>& points,
                         bool relative) const;

    /// \brief Scale the coordinates of all points by the scaling factor \p
    /// scale.
    void ScalePoints(const double scale,
                     std::vector<Eigen::Vector3d>& points,
                     const Eigen::Vector3d& center) const;

    /// \brief Rotate all points with the rotation matrix \p R.
    void RotatePoints(const Eigen::Matrix3d& R,
                      std::vector<Eigen::Vector3d>& points,
                      const Eigen::Vector3d& center) const;

    /// \brief Rotate all normals with the rotation matrix \p R.
    void RotateNormals(const Eigen::Matrix3d& R,
                       std::vector<Eigen::Vector3d>& normals) const;

    /// \brief Rotate all covariance matrices with the rotation matrix \p R.
    void RotateCovariances(const Eigen::Matrix3d& R,
                           std::vector<Eigen::Matrix3d>& covariances) const;
};

}  // namespace geometry
}  // namespace hymson3d
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

    virtual AxisAlignedBoundingBox GetAxisAlignedBoundingBox() const = 0;

    virtual OrientedBoundingBox GetOrientedBoundingBox(
            bool robust = false) const = 0;
    virtual OrientedBoundingBox GetMinimalOrientedBoundingBox(
            bool robust = false) const = 0;

protected:
    Eigen::Vector3d ComputeMinBound(
            const std::vector<Eigen::Vector3d>& points) const;
    Eigen::Vector3d ComputeMaxBound(
            const std::vector<Eigen::Vector3d>& points) const;
    Eigen::Vector3d ComputeCenter(
            const std::vector<Eigen::Vector3d>& points) const;
};

}  // namespace geometry
}  // namespace hymson3d
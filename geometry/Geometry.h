#pragma once

#include <string>

#include "Logger.h"

namespace hymson3d {
namespace geometry {

class Geometry {
public:
    enum class GeometryType {
        Unspecified = 0,
        PointCloud = 1,
        Image = 2,
        RGBDImage = 3,
        LineSet = 4,
        Mesh = 5,
        OrientedBoundingBox = 6,
        AxisAlignedBoundingBox = 7,
        Plane = 8
    };

public:
    virtual ~Geometry() {}

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type Specifies the type of geometry of the object constructed.
    /// \param dimension Specifies whether the dimension is 2D or 3D.
    Geometry(GeometryType type, int dimension)
        : geometry_type_(type), dimension_(dimension) {}

public:
    /// Clear all elements in the geometry.
    virtual Geometry& Clear() = 0;
    /// Returns `true` iff the geometry is empty.
    virtual bool IsEmpty() const = 0;
    /// Returns one of registered geometry types.
    GeometryType GetGeometryType() const { return geometry_type_; }
    /// Returns whether the geometry is 2D or 3D.
    int Dimension() const { return dimension_; }

    std::string GetName() const { return name_; }
    void SetName(const std::string& name) { name_ = name; }

private:
    /// Type of geometry from GeometryType.
    GeometryType geometry_type_ = GeometryType::Unspecified;
    /// Number of dimensions of the geometry.
    int dimension_ = 3;
    std::string name_;
};
}  // namespace geometry
}  // namespace hymson3d
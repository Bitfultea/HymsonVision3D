// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <Eigen/Core>

#include "Geometry.h"

namespace hymson3d {
namespace geometry {

/// \class Geometry2D
///
/// \brief The base geometry class for 2D geometries.
///
/// Main class for 2D geometries, Derives all data from Geometry Base class.
class Geometry2D : public Geometry {
public:
    ~Geometry2D() override {}

protected:
    /// \brief Parameterized Constructor.
    ///
    /// \param type  type of object based on GeometryType
    Geometry2D(GeometryType type) : Geometry(type, 2) {}

public:
    Geometry& Clear() override = 0;
    bool IsEmpty() const override = 0;
    /// Returns min bounds for geometry coordinates.
    virtual Eigen::Vector2d GetMinBound() const = 0;
    /// Returns max bounds for geometry coordinates.
    virtual Eigen::Vector2d GetMaxBound() const = 0;

public:
    /// Width of the image.
    int width_ = 0;
    /// Height of the image.
    int height_ = 0;
    /// Number of channels in the image.
    int num_of_channels_ = 0;
    /// Number of bytes per channel.
    int bytes_per_channel_ = 0;
    /// Image storage buffer.
    std::vector<uint8_t> data_;
};

}  // namespace geometry
}  // namespace hymson3d

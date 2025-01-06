#pragma once
#include <Eigen/Core>
#include <memory>
#include <tuple>
#include <vector>

#include "BoundingBox3D.h"
#include "Geometry3D.h"

namespace hymson3d {
namespace geometry {
struct curvature {
    double mean_curvature;
    double gaussian_curvature;
    double total_curvature;
};

class PointCloud : public Geometry3D {
public:
    typedef std::shared_ptr<PointCloud> Ptr;
    /// \brief Default Constructor.
    PointCloud() : Geometry3D(Geometry::GeometryType::PointCloud) {}
    /// \brief Parameterized Constructor.
    ///
    /// \param points Points coordinates.
    PointCloud(const std::vector<Eigen::Vector3d> &points)
        : Geometry3D(Geometry::GeometryType::PointCloud), points_(points) {}
    ~PointCloud() override {}

public:
    PointCloud &Clear() override;
    bool IsEmpty() const override;
    Eigen::Vector3d GetMinBound() const override;
    Eigen::Vector3d GetExtend() const;
    Eigen::Vector3d GetMaxBound() const override;
    Eigen::Vector3d GetCenter() const override;

    AABB GetAxisAlignedBoundingBox() const;
    OBB GetOrientedBoundingBox(bool robust = false) const;
    OBB GetMinimalOrientedBoundingBox(bool robust = false) const;

    PointCloud &Transform(const Eigen::Matrix4d &transformation) override;
    PointCloud &Translate(const Eigen::Vector3d &translation,
                          bool relative = true) override;
    PointCloud &Scale(const double scale,
                      const Eigen::Vector3d &center) override;
    PointCloud &Rotate(const Eigen::Matrix3d &R,
                       const Eigen::Vector3d &center) override;

    PointCloud &operator+=(const PointCloud &cloud);
    PointCloud operator+(const PointCloud &cloud) const;

    /// Returns 'true' if the point cloud contains points.
    bool HasPoints() const { return points_.size() > 0; }

    /// Returns `true` if the point cloud contains point normals.
    bool HasNormals() const {
        return points_.size() > 0 && normals_.size() == points_.size();
    }

    /// Returns `true` if the point cloud contains point colors.
    bool HasColors() const {
        return points_.size() > 0 && colors_.size() == points_.size();
    }

    /// Returns 'true' if the point cloud contains per-point covariance matrix.
    bool HasCovariances() const {
        return !points_.empty() && covariances_.size() == points_.size();
    }
    bool HasIntensities() const {
        return points_.size() > 0 && intensities_.size() == points_.size();
    }

    bool HasLabels() const { return labels_.size() == points_.size(); }

    bool HasCurvatures() const { return curvatures_.size() == points_.size(); }

    /// Normalize point normals to length 1.
    PointCloud &NormalizeNormals() {
        for (size_t i = 0; i < normals_.size(); i++) {
            normals_[i].normalize();
        }
        return *this;
    }

    /// Assigns each point in the PointCloud the same color.
    PointCloud &PaintUniformColor(const Eigen::Vector3d &color) {
        ResizeAndPaintUniformColor(colors_, points_.size(), color);
        return *this;
    }

    std::shared_ptr<PointCloud> Crop(const AxisAlignedBoundingBox &bbox,
                                     bool invert = false) const;

    std::shared_ptr<PointCloud> Crop(const OrientedBoundingBox &bbox,
                                     bool invert = false) const;
    // Eigen::Vector3d GenerateRandomColor() const;

public:
    /// Points coordinates.
    std::vector<Eigen::Vector3d> points_;
    /// Points normals.
    std::vector<Eigen::Vector3d> normals_;
    /// RGB colors of points.
    std::vector<Eigen::Vector3d> colors_;
    /// Covariance Matrix for each point
    std::vector<Eigen::Matrix3d> covariances_;
    /// Points Intensities
    std::vector<float> intensities_;
    /// Labels for each point
    std::vector<int> labels_;
    /// Curvature of each point
    std::vector<curvature *> curvatures_;
    // y_slices
    std::vector<std::vector<Eigen::Vector2d>> y_slices_;
    std::vector<std::vector<Eigen::Vector3d>> ny_slices_;
    std::vector<std::vector<size_t>> y_slice_idxs;
    std::vector<std::vector<Eigen::Vector2d>> x_slices_;
    std::vector<std::vector<Eigen::Vector3d>> nx_slices_;
    std::vector<std::vector<size_t>> x_slice_idxs;

    // for tiff conversion
    size_t width_;
    size_t height_;
};
}  // namespace geometry
}  // namespace hymson3d
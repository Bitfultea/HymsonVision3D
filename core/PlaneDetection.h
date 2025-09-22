#pragma once
#include "2D/Curve.h"
#include "3D/Plane.h"
#include "3D/PointCloud.h"
#include "Normal.h"
#include "fmtfallback.h"

namespace hymson3d {
namespace core {
class PlaneDetection {
public:
    PlaneDetection() = default;
    ~PlaneDetection() = default;

public:
    // Fitting a plane to many points in 3D
    // From https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
    geometry::Plane::Ptr fit_a_plane(geometry::PointCloud& pointcloud);

    geometry::Plane::Ptr fit_a_plane_ransc(geometry::PointCloud& cloud);

    // use bspline to fit a curve
    Eigen::VectorXd fit_a_curve(std::vector<Eigen::Vector2d> control_pts,
                                int sampled_pts,
                                int plot_id = 0,
                                bool debug_mode = true);

    std::vector<Eigen::Vector2d> resample_a_curve(
            std::vector<Eigen::Vector2d> control_pts,
            int sampled_pts,
            int plot_id = 0,
            bool debug_mode = true);

    std::shared_ptr<geometry::BSpline> generate_a_curve(
            std::vector<Eigen::Vector2d> control_pts);

public:
    double search_radius = 0.1;

private:
    void plot_curve(std::vector<double> x_vec,
                    std::vector<double> y_vec,
                    std::vector<double> spline_x_vec,
                    std::vector<double> spline_y_vec,
                    int plot_id = 0);

    void plot_curve(std::vector<double> x_vec,
                    std::vector<double> y_vec,
                    std::vector<double> u_vec,
                    std::vector<double> v_vec,
                    std::vector<double> spline_x_vec,
                    std::vector<double> spline_y_vec,
                    int plot_id = 0);
};

}  // namespace core
}  // namespace hymson3d
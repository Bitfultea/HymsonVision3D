#pragma once
#include "2D/Curve.h"
#include "3D/Plane.h"
#include "3D/PointCloud.h"
#include "Normal.h"
#include "fmtfallback.h"
#include "open3d/Random.h"

namespace hymson3d {
namespace core {

struct ransac_result {
    double fitness_;
    double inlier_rmse_;
};

/// \class RandomSampler
///
/// \brief Helper class for random sampling
template <typename T>
class RandomSampler {
public:
    explicit RandomSampler(const size_t total_size) : total_size_(total_size) {}

    std::vector<T> operator()(size_t sample_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<T> samples;
        samples.reserve(sample_size);

        size_t valid_sample = 0;
        std::unordered_set<int> sample_set;
        while (valid_sample < sample_size) {
            const size_t idx =
                    open3d::utility::random::RandUint32() % total_size_;
            // Well, this is slow. But typically the sample_size is small.
            // if (std::find(samples.begin(), samples.end(), idx) ==
            //     samples.end()) {
            //     samples.push_back(idx);
            //     valid_sample++;
            // }
            if (sample_set.find(idx) == sample_set.end()) {
                sample_set.insert(idx);
                samples.push_back(idx);
                valid_sample++;
            }
        }

        return samples;
    }

private:
    size_t total_size_;
    std::mutex mutex_;
};

class PlaneDetection {
public:
    PlaneDetection() = default;
    ~PlaneDetection() = default;

public:
    // Fitting a plane to many points in 3D
    // From https://www.ilikebigbits.com/2015_03_04_plane_from_points.html
    geometry::Plane::Ptr fit_a_plane(geometry::PointCloud& pointcloud);
    geometry::Plane::Ptr fit_a_plane(std::vector<Eigen::Vector3d>& points);

    geometry::Plane::Ptr fit_a_plane_ransc(
            geometry::PointCloud& cloud,
            const double distance_threshold = 0.01,
            const int ransac_n = 3,
            const int num_iterations = 100,
            const double probability = 0.99999999);

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

    // evaluation of Ransac result
    // Calculates the number of inliers given a list of points and a plane
    // model,
    // and the total squared point-to-plane distance.
    // These numbers are then used to evaluate how well the plane model fits the
    // given points.
    ransac_result evaluate_RANSAC_baseon_distance(
            const std::vector<Eigen::Vector3d>& points,
            const Eigen::Vector4d plane_model,
            std::vector<size_t>& inliers,
            double distance_threshold);
};

}  // namespace core
}  // namespace hymson3d
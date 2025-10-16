#include "Distance.h"

#include "3D/KDtree.h"
namespace hymson3d {
namespace core {
namespace feature {

double point_to_plane_distance(const Eigen::Vector3d& point,
                               const geometry::Plane& plane) {
    double normal_mag = plane.normal_.norm();
    double dist =
            std::abs(plane.normal_.dot(point) + plane.coeff_[3]) / normal_mag;
    return dist;
}

double point_to_plane_signed_distance(const Eigen::Vector3d& point,
                                      const geometry::Plane& plane) {
    double normal_mag = plane.normal_.norm();
    double dist = (plane.normal_.dot(point) + plane.coeff_[3]) / normal_mag;
    return dist;
}

double cloud_to_cloud_distance(const geometry::PointCloud& cloud,
                               const geometry::PointCloud& cloud_target) {
    std::vector<double> distances(cloud.points_.size());
    geometry::KDTree kdtree;
    kdtree.SetData(cloud_target);
#pragma omp parallel for
    for (int i = 0; i < cloud.points_.size(); i++) {
        std::vector<int> indices(1);
        std::vector<double> dists(1);
        if (kdtree.SearchKNN(cloud.points_[i], 1, indices, dists) == 0) {
            distances[i] = 0;
        } else {
            distances[i] = std::sqrt(dists[0]);
        }
    }
    // 找到最小距离
    if (!distances.empty()) {
        return *std::min_element(distances.begin(), distances.end());
    }
    return 0.0;
}

double cloud_to_cloud_distance(
        const std::vector<Eigen::Vector3d>& points,
        const std::vector<Eigen::Vector3d>& points_target) {
    std::vector<double> distances(points.size());
    geometry::KDTree kdtree;
    geometry::PointCloud* cloud_target =
            new geometry::PointCloud(points_target);
    kdtree.SetData(*cloud_target);
#pragma omp parallel for
    for (int i = 0; i < points.size(); i++) {
        std::vector<int> indices(1);
        std::vector<double> dists(1);
        if (kdtree.SearchKNN(points[i], 1, indices, dists) == 0) {
            distances[i] = 0;
        } else {
            distances[i] = std::sqrt(dists[0]);
        }
    }
    // 找到最小距离
    if (!distances.empty()) {
        return *std::min_element(distances.begin(), distances.end());
    }
    return 0.0;
}

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
#include "Distance.h"

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

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
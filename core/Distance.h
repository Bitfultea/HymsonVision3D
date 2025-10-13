#include "3D/KDtree.h"
#include "3D/Plane.h"
#include "3D/PointCloud.h"
#include "fmtfallback.h"

namespace hymson3d {
namespace core {
namespace feature {
double point_to_plane_distance(const Eigen::Vector3d& point,
                               const geometry::Plane& plane);

double point_to_plane_signed_distance(const Eigen::Vector3d& point,
                                      const geometry::Plane& plane);

// compute the distance between two pointclouds which is the distance of two
// closest points in both clouds
double cloud_to_cloud_distance(const geometry::PointCloud& cloud,
                               const geometry::PointCloud& cloud_target);

double cloud_to_cloud_distance(
        const std::vector<Eigen::Vector3d>& points,
        const std::vector<Eigen::Vector3d>& points_target);
}  // namespace feature
}  // namespace core
}  // namespace hymson3d
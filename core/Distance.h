#include "fmtfallback.h"
#include "3D/KDtree.h"
#include "3D/Plane.h"
#include "3D/PointCloud.h"

namespace hymson3d {
namespace core {
namespace feature {
double point_to_plane_distance(const Eigen::Vector3d& point,
                               const geometry::Plane& plane);

double point_to_plane_signed_distance(const Eigen::Vector3d& point,
                                      const geometry::Plane& plane);

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
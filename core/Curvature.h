#include "3D/KDtree.h"
#include "3D/PointCloud.h"

namespace hymson3d {
namespace core {
namespace feature {
void ComputeCurvature_PCL(geometry::PointCloud& cloud,
                          geometry::KDTreeSearchParam& param);

// TODO:: implement fast curvature computation
void ComputeCurvature(geometry::PointCloud& cloud,
                      geometry::KDTreeSearchParam& param);

Eigen::Vector3d color_with_curvature(double curvature,
                                     double min_val,
                                     double max_val);

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
#include "3D/PointCloud.h"

namespace hymson3d {
namespace core {
namespace feature {

void ComputeNormals_PCA(geometry::PointCloud& cloud);

void ComputeNormals_PCL(geometry::PointCloud& cloud, float radius);

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
#include "3D/KDtree.h"
#include "3D/PointCloud.h"

namespace hymson3d {
namespace core {
namespace feature {

void ComputeNormals_PCA(geometry::PointCloud& cloud,
                        geometry::KDTreeSearchParam& param);

// implemetation using PCL lib with PCA method
void ComputeNormals_PCL(geometry::PointCloud& cloud,
                        geometry::KDTreeSearchParam& param);

// fitting a jet surface over its nearest neighbors. The default jet is a
// quadric surface. This algorithm is well suited to point sets scattered over
// curved surfaces.
// TODO::Implement this function
void ComputeNormals_JET(geometry::PointCloud& cloud,
                        geometry::KDTreeSearchParam& param);

// using the Voronoi Covariance Measure of the point set. This algorithm is more
// complex and slower than the previous algorithms.
// See https://inria.hal.science/inria-00406575v2/document
// TODO::Implement this function
void ComputerNormals_VCM(geometry::PointCloud& cloud);

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
#include "3D/Plane.h"

namespace hymson3d {
namespace geometry {

void Plane::compute_orthogonal_basis() {
    static constexpr double tol = 1e-3;
    if ((Eigen::Vector3d(0, 1, 1) - normal_).squaredNorm() > tol) {
        // construct x-vec by cross(normal, [0;1;1])
        orth_basis.col(0) = Eigen::Vector3d(normal_.y() - normal_.z(),
                                            -normal_.x(), normal_.x())
                                    .normalized();
    } else {
        // construct x-vec by cross(normal, [1;0;1])
        orth_basis.col(0) =
                Eigen::Vector3d(normal_.y(), normal_.z() - normal_.x(),
                                -normal_.y())
                        .normalized();
    }
    orth_basis.col(1) = normal_.cross(orth_basis.col(0)).normalized();
    orth_basis.col(2) = normal_;
}

}  // namespace geometry
}  // namespace hymson3d
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

void Plane::orient_normals_towards_positive_z() {
    const Eigen::Vector3d z_reference = Eigen::Vector3d(0.0, 0.0, 1.0);
    if (normal_.norm() == 0.0) {
        LOG_ERROR(
                "Plane's normal is not computed yet. Compute the planar "
                "first.");
        return;
    }
    if (normal_.dot(z_reference) < 0.0) {
        normal_ *= -1.0;  // flip the normal
        coeff_[3] *= -1.0; //filp D
    }
}

}  // namespace geometry
}  // namespace hymson3d
#include "Eigen.h"
namespace hymson3d {
namespace utility {
namespace mathtool {

Eigen::Vector3d ComputeEigenvector0(const Eigen::Matrix3d &A, double eval0);

Eigen::Vector3d ComputeEigenvector1(const Eigen::Matrix3d &A,
                                    const Eigen::Vector3d &evec0,
                                    double eval1);

Eigen::Vector3d FastEigen3x3(const Eigen::Matrix3d &covariance);

}  // namespace mathtool
}  // namespace utility
}  // namespace hymson3d
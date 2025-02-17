#include <tuple>

#include "Eigen.h"

namespace hymson3d {
namespace utility {
namespace mathtool {

Eigen::Vector3d ComputeEigenvector0(const Eigen::Matrix3d &A, double eval0);

Eigen::Vector3d ComputeEigenvector1(const Eigen::Matrix3d &A,
                                    const Eigen::Vector3d &evec0,
                                    double eval1);

std::tuple<Eigen::Vector3d, std::vector<double>> FastEigen3x3(
        const Eigen::Matrix3d &covariance);

template <typename T>
T GetMedian(std::vector<T> buffer) {
    const size_t N = buffer.size();
    std::nth_element(buffer.begin(), buffer.begin() + N / 2,
                     buffer.begin() + N);
    return buffer[N / 2];
}

// Calculate the Median Absolute Deviation statistic
template <typename T>
T GetMAD(const std::vector<T> &buffer, T median) {
    const size_t N = buffer.size();
    std::vector<T> shifted(N);
    for (size_t i = 0; i < N; i++) {
        shifted[i] = std::abs(buffer[i] - median);
    }
    std::nth_element(shifted.begin(), shifted.begin() + N / 2,
                     shifted.begin() + N);
    static constexpr double k = 1.4826;  // assumes normally distributed data
    return k * shifted[N / 2];
}

}  // namespace mathtool
}  // namespace utility
}  // namespace hymson3d
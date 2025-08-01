#include "MathTool.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>

#include "Logger.h"
namespace hymson3d {
namespace utility {
namespace mathtool {

Eigen::Vector3d ComputeEigenvector0(const Eigen::Matrix3d &A, double eval0) {
    Eigen::Vector3d row0(A(0, 0) - eval0, A(0, 1), A(0, 2));
    Eigen::Vector3d row1(A(0, 1), A(1, 1) - eval0, A(1, 2));
    Eigen::Vector3d row2(A(0, 2), A(1, 2), A(2, 2) - eval0);
    Eigen::Vector3d r0xr1 = row0.cross(row1);
    Eigen::Vector3d r0xr2 = row0.cross(row2);
    Eigen::Vector3d r1xr2 = row1.cross(row2);
    double d0 = r0xr1.dot(r0xr1);
    double d1 = r0xr2.dot(r0xr2);
    double d2 = r1xr2.dot(r1xr2);

    double dmax = d0;
    int imax = 0;
    if (d1 > dmax) {
        dmax = d1;
        imax = 1;
    }
    if (d2 > dmax) {
        imax = 2;
    }

    if (imax == 0) {
        return r0xr1 / std::sqrt(d0);
    } else if (imax == 1) {
        return r0xr2 / std::sqrt(d1);
    } else {
        return r1xr2 / std::sqrt(d2);
    }
}

Eigen::Vector3d ComputeEigenvector1(const Eigen::Matrix3d &A,
                                    const Eigen::Vector3d &evec0,
                                    double eval1) {
    Eigen::Vector3d U, V;
    if (std::abs(evec0(0)) > std::abs(evec0(1))) {
        double inv_length =
                1 / std::sqrt(evec0(0) * evec0(0) + evec0(2) * evec0(2));
        U << -evec0(2) * inv_length, 0, evec0(0) * inv_length;
    } else {
        double inv_length =
                1 / std::sqrt(evec0(1) * evec0(1) + evec0(2) * evec0(2));
        U << 0, evec0(2) * inv_length, -evec0(1) * inv_length;
    }
    V = evec0.cross(U);

    Eigen::Vector3d AU(A(0, 0) * U(0) + A(0, 1) * U(1) + A(0, 2) * U(2),
                       A(0, 1) * U(0) + A(1, 1) * U(1) + A(1, 2) * U(2),
                       A(0, 2) * U(0) + A(1, 2) * U(1) + A(2, 2) * U(2));

    Eigen::Vector3d AV = {A(0, 0) * V(0) + A(0, 1) * V(1) + A(0, 2) * V(2),
                          A(0, 1) * V(0) + A(1, 1) * V(1) + A(1, 2) * V(2),
                          A(0, 2) * V(0) + A(1, 2) * V(1) + A(2, 2) * V(2)};

    double m00 = U(0) * AU(0) + U(1) * AU(1) + U(2) * AU(2) - eval1;
    double m01 = U(0) * AV(0) + U(1) * AV(1) + U(2) * AV(2);
    double m11 = V(0) * AV(0) + V(1) * AV(1) + V(2) * AV(2) - eval1;

    double absM00 = std::abs(m00);
    double absM01 = std::abs(m01);
    double absM11 = std::abs(m11);
    double max_abs_comp;
    if (absM00 >= absM11) {
        max_abs_comp = std::max(absM00, absM01);
        if (max_abs_comp > 0) {
            if (absM00 >= absM01) {
                m01 /= m00;
                m00 = 1 / std::sqrt(1 + m01 * m01);
                m01 *= m00;
            } else {
                m00 /= m01;
                m01 = 1 / std::sqrt(1 + m00 * m00);
                m00 *= m01;
            }
            return m01 * U - m00 * V;
        } else {
            return U;
        }
    } else {
        max_abs_comp = std::max(absM11, absM01);
        if (max_abs_comp > 0) {
            if (absM11 >= absM01) {
                m01 /= m11;
                m11 = 1 / std::sqrt(1 + m01 * m01);
                m01 *= m11;
            } else {
                m11 /= m01;
                m01 = 1 / std::sqrt(1 + m11 * m11);
                m11 *= m01;
            }
            return m11 * U - m01 * V;
        } else {
            return U;
        }
    }
}

std::tuple<Eigen::Vector3d, std::vector<double>> FastEigen3x3(
        const Eigen::Matrix3d &covariance) {
    // Current version based on
    // https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    // which handles edge cases like points on a plane

    Eigen::Matrix3d A = covariance;
    double max_coeff = A.maxCoeff();
    if (max_coeff == 0) {
        return {Eigen::Vector3d::Zero(), {0, 0, 0}};
    }
    A /= max_coeff;

    double norm = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    if (norm > 0) {
        Eigen::Vector3d eval;
        Eigen::Vector3d evec0;
        Eigen::Vector3d evec1;
        Eigen::Vector3d evec2;

        double q = (A(0, 0) + A(1, 1) + A(2, 2)) / 3;

        double b00 = A(0, 0) - q;
        double b11 = A(1, 1) - q;
        double b22 = A(2, 2) - q;

        double p =
                std::sqrt((b00 * b00 + b11 * b11 + b22 * b22 + norm * 2) / 6);

        double c00 = b11 * b22 - A(1, 2) * A(1, 2);
        double c01 = A(0, 1) * b22 - A(1, 2) * A(0, 2);
        double c02 = A(0, 1) * A(1, 2) - b11 * A(0, 2);
        double det = (b00 * c00 - A(0, 1) * c01 + A(0, 2) * c02) / (p * p * p);

        double half_det = det * 0.5;
        half_det = std::min(std::max(half_det, -1.0), 1.0);

        double angle = std::acos(half_det) / (double)3;
        double const two_thirds_pi = 2.09439510239319549;
        double beta2 = std::cos(angle) * 2;
        double beta0 = std::cos(angle + two_thirds_pi) * 2;
        double beta1 = -(beta0 + beta2);

        eval(0) = q + p * beta0;
        eval(1) = q + p * beta1;
        eval(2) = q + p * beta2;

        if (half_det >= 0) {
            evec2 = ComputeEigenvector0(A, eval(2));
            if (eval(2) < (0) && eval(2) < eval(1)) {
                A *= max_coeff;
                std::vector<double> evals = {eval(2),
                                             std::min(eval(1), eval(0)),
                                             std::max(eval(1), eval(0))};
                return std::make_tuple(evec2, evals);
            }
            evec1 = ComputeEigenvector1(A, evec2, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                std::vector<double> evals = {eval(1),
                                             std::min(eval(0), eval(2)),
                                             std::max(eval(0), eval(2))};
                return std::make_tuple(evec1, evals);
            }
            evec0 = evec1.cross(evec2);
            std::vector<double> evals = {eval(0), std::min(eval(1), eval(2)),
                                         std::max(eval(1), eval(2))};
            return std::make_tuple(evec0, evals);
        } else {
            evec0 = ComputeEigenvector0(A, eval(0));
            if (eval(0) < eval(1) && eval(0) < eval(2)) {
                A *= max_coeff;
                std::vector<double> evals = {eval(0),
                                             std::min(eval(1), eval(2)),
                                             std::max(eval(1), eval(2))};
                return std::make_tuple(evec0, evals);
            }
            evec1 = ComputeEigenvector1(A, evec0, eval(1));
            A *= max_coeff;
            if (eval(1) < eval(0) && eval(1) < eval(2)) {
                std::vector<double> evals = {eval(1),
                                             std::min(eval(0), eval(2)),
                                             std::max(eval(0), eval(2))};
                return std::make_tuple(evec1, evals);
            }
            evec2 = evec0.cross(evec1);
            std::vector<double> evals = {eval(2), std::min(eval(1), eval(0)),
                                         std::max(eval(1), eval(0))};
            return std::make_tuple(evec2, evals);
        }
    } else {
        A *= max_coeff;
        if (A(0, 0) < A(1, 1) && A(0, 0) < A(2, 2)) {
            return std::make_tuple(Eigen::Vector3d(1, 0, 0),
                                   std::vector<double>{1, 0, 0});
        } else if (A(1, 1) < A(0, 0) && A(1, 1) < A(2, 2)) {
            return std::make_tuple(Eigen::Vector3d(0, 1, 0),
                                   std::vector<double>{0, 1, 0});
        } else {
            return std::make_tuple(Eigen::Vector3d(0, 0, 1),
                                   std::vector<double>{0, 1, 0});
        }
    }
}

// SVD软阈值处理
Eigen::MatrixXd singularValueThresholding(const Eigen::MatrixXd &D,
                                          double tau) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            D, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd singularValues = svd.singularValues();

    // thresholding
    for (int i = 0; i < singularValues.size(); ++i) {
        singularValues(i) = std::max(singularValues(i) - tau, 0.0);
    }

    // compute lowrank matrix L
    return svd.matrixU() * singularValues.asDiagonal() *
           svd.matrixV().transpose();
}

Eigen::MatrixXd softThresholding(const Eigen::MatrixXd &X, double lambda) {
    return (X.array().abs() > lambda)
            .select(X.array() - lambda * X.array().sign(), 0);
}

// ALM method based
void RPCA(const Eigen::MatrixXd &M,
          Eigen::MatrixXd &L,
          Eigen::MatrixXd &S,
          double lambda,
          double tol,
          int maxIter) {
    int m = M.rows();
    int n = M.cols();

    if (lambda < 0) {
        lambda = 1.0 / sqrt(std::max(m, n));
    }

    // 初始化
    Eigen::MatrixXd Y = M / std::max(M.norm(), 1.0);
    L = Eigen::MatrixXd::Zero(m, n);
    S = Eigen::MatrixXd::Zero(m, n);
    double mu = 1.25 / L.norm();
    double mu_bar = mu * 1e7;
    double rho = 1.5;

    for (int iter = 0; iter < maxIter; ++iter) {
        // 更新 L（低秩部分）
        L = singularValueThresholding(M - S + (1.0 / mu) * Y, 1.0 / mu);

        // 更新 S（稀疏部分）
        S = softThresholding(M - L + (1.0 / mu) * Y, lambda / mu);

        // 更新拉格朗日乘子
        Eigen::MatrixXd Z = M - L - S;
        Y = Y + mu * Z;

        // 更新 mu
        mu = std::min(mu * rho, mu_bar);

        // 收敛条件
        std::cout << Z.norm() << "; " << M.norm() << std::endl;
        if (Z.norm() / M.norm() < tol) {
            LOG_INFO("Converged in {} iterations.", iter + 1);
            break;
        }
    }
}

}  // namespace mathtool
}  // namespace utility
}  // namespace hymson3d
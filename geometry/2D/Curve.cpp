#include "Curve.h"

#include <algorithm>
#include <stdexcept>

#include "Logger.h"

namespace hymson3d {
namespace geometry {
// 构造函数
BSpline::BSpline(int degree) : degree_(degree) {}

// 设置节点
void BSpline::setKnots(const Eigen::VectorXd& knots) { knots_ = knots; }

// 设置控制点
void BSpline::setControlPoints(const Eigen::VectorXd& x,
                               const Eigen::VectorXd& y) {
    x_ = x;
    y_ = y;
    if (x_.size() != y_.size()) {
        throw std::invalid_argument("x and y must have the same length.");
    }
}

// 生成节点向量
Eigen::VectorXd BSpline::generateKnots(int N,
                                       double a,
                                       double b,
                                       int method) const {
    int K = degree_ + 1;
    if (N < K) {
        throw std::invalid_argument("N must be greater than or equal to K.");
    }

    if (method == 1) {
        return Eigen::VectorXd::LinSpaced(N + K, a, b);  // 均匀样条
    } else {
        Eigen::VectorXd knots(N + K);
        knots.head(K - 1).setConstant(a);
        knots.tail(K - 1).setConstant(b);
        knots.segment(K - 1, N - K + 2) =
                Eigen::VectorXd::LinSpaced(N - K + 2, a, b);
        return knots;  // 准均匀样条
    }
}

Eigen::MatrixXd BSpline::generateSplineBasis(const Eigen::VectorXd& pts) const {
    int N = pts.size();     // 输入点数量
    int M = knots_.size();  // 节点数量

    // 打印调试信息
    // std::cout << "Number of points (N): " << N << std::endl;
    // std::cout << "Number of knots (M): " << M << std::endl;

    // 确保节点的数量至少大于1，否则无法生成样条基
    if (M <= 1) {
        LOG_ERROR("Error: Number of knots must be greater than 1.");
        // std::cout << "Error: Number of knots must be greater than 1."
        //           << std::endl;
        return Eigen::MatrixXd();
    }

    // 初始化样条基函数矩阵
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N, M - 1);

    // 打印 B 矩阵的大小
    // std::cout << "Matrix B size: " << B.rows() << " x " << B.cols() <<
    // std::endl;

    // 初始化 1 阶样条基
    for (int i = 0; i < M - 1; ++i) {
        for (int j = 0; j < N; ++j) {
            if (pts[j] >= knots_[i] && pts[j] < knots_[i + 1]) {
                B(j, i) = 1.0;
            }
        }
    }

    // 确保最后一个基函数正确
    if (N > 0 && M > 1) {
        B(N - 1, M - 2) = 1.0;
    }

    // 打印基函数矩阵更新后的状态
    // std::cout << "Updated B matrix after initialization:" << std::endl;
    // std::cout << B << std::endl;

    // 递归计算 k 阶样条基
    int K = degree_ + 1;  // 样条的阶数
    for (int k_ = 2; k_ <= K; ++k_) {
        for (int i = 0; i < M - k_; ++i) {
            for (int j = 0; j < N; ++j) {
                double c1 = (knots_[i + k_ - 1] != knots_[i])
                                    ? (pts[j] - knots_[i]) /
                                              (knots_[i + k_ - 1] - knots_[i])
                                    : 0.0;
                double c2 = (knots_[i + k_] != knots_[i + 1])
                                    ? (knots_[i + k_] - pts[j]) /
                                              (knots_[i + k_] - knots_[i + 1])
                                    : 0.0;
                // 更新样条基函数矩阵
                B(j, i) = c1 * B(j, i) + c2 * B(j, i + 1);
            }
        }
    }

    // 打印递归计算后的 B 矩阵
    // std::cout << "Updated B matrix after recursion:" << std::endl;
    // std::cout << B << std::endl;

    // 返回前 M-K 列的矩阵（去掉多余的列）
    return B.leftCols(M - K);
}

// 获取样条曲线的 x 坐标
Eigen::VectorXd BSpline::getSplineX(const Eigen::VectorXd& pts) {
    Eigen::MatrixXd B = generateSplineBasis(pts);
    return B * x_;
}

// 获取样条曲线的 y 坐标
Eigen::VectorXd BSpline::getSplineY(const Eigen::VectorXd& pts) {
    Eigen::MatrixXd B = generateSplineBasis(pts);
    return B * y_;
}

// 插值函数
Eigen::VectorXd BSpline::interpolate(const Eigen::VectorXd& u) {
    Eigen::VectorXd xx = getSplineX(u);
    Eigen::VectorXd yy = getSplineY(u);
    Eigen::VectorXd v(u.size());

    for (int i = 0; i < u.size(); ++i) {
        // 找到 u[i] 在 xx 中的位置，保证 idx 不会越界
        int idx = (std::upper_bound(xx.data(), xx.data() + xx.size(), u[i]) -
                   xx.data()) -
                  1;

        // 确保 idx 在合法范围内
        if (idx < 0) {
            idx = 0;  // 如果 u[i] 小于 xx[0]，则使用最左边的值
        }
        if (idx >= xx.size() - 1) {
            // 如果 u[i] 大于 xx[xx.size() - 1]，则使用最右边的值
            idx = xx.size() - 2;
        }

        // 计算 t 并检查是否有除零错误
        double denominator = xx[idx + 1] - xx[idx];
        double t = 0.0;  // 必须在这里声明并初始化 t
        if (denominator != 0) {
            t = (u[i] - xx[idx]) / denominator;  // 如果分母不为零，计算 t
        }

        // 计算插值
        v[i] = (1 - t) * yy[idx] + t * yy[idx + 1];
    }

    return v;
}

}  // namespace geometry
}  // namespace hymson3d
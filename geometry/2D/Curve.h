#pragma once

#include <Eigen/Core>

namespace hymson3d {
namespace geometry {
class BSpline {
public:
    // 构造函数：初始化 B-Spline 对象
    BSpline(int degree = 3);

    // 设置节点向量
    void setKnots(const Eigen::VectorXd &knots);

    // 设置控制点
    void setControlPoints(const Eigen::VectorXd &x, const Eigen::VectorXd &y);

    // 根据给定的参数生成节点向量
    // void generateKnots(int n);
    Eigen::VectorXd generateKnots(int N, double a, double b, int method) const;

    // 生成样条基函数矩阵
    // void generateSplineBasis();
    Eigen::MatrixXd generateSplineBasis(const Eigen::VectorXd &pts) const;

    // 根据基函数进行插值
    // void interpolate();
    Eigen::VectorXd interpolate(const Eigen::VectorXd &u);

    // 获取插值后的 x 坐标
    // std::vector<double> getSplineX() const;
    Eigen::VectorXd getSplineX(const Eigen::VectorXd &pts);

    // 获取插值后的 y 坐标
    // std::vector<double> getSplineY() const;
    Eigen::VectorXd getSplineY(const Eigen::VectorXd &pts);

private:
    int degree_;                                   // B-Spline 曲线的度数
    Eigen::VectorXd knots_;                        // 节点向量
    Eigen::VectorXd x_;                            // x 控制点
    Eigen::VectorXd y_;                            // y 控制点
    std::vector<std::vector<double>> splineBasis;  // 样条基函数矩阵
    std::vector<double> splineX;                   // 插值后的 x 坐标
    std::vector<double> splineY;                   // 插值后的 y 坐标
};
}  // namespace geometry
}  // namespace hymson3d
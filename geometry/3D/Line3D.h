#pragma once

#include "open3d/Line3D.h"

namespace hymson3d {
namespace geometry {

class Line3D : public open3d::geometry::Line3D {};

class BSpline {
public:
    // 构造函数：初始化 B-Spline 对象
    BSpline(int degree = 3);

    // 设置节点向量
    void setKnots(const Eigen::VectorXd &knots);

    // 设置控制点
    void setControlPoints(const Eigen::VectorXd &x, const Eigen::VectorXd &y);

    // 根据给定的参数生成节点向量
    void generateKnots(int n);

    // 生成样条基函数矩阵
    void generateSplineBasis();

    // 根据基函数进行插值
    void interpolate();

    // 获取插值后的 x 坐标
    std::vector<double> getSplineX() const;

    // 获取插值后的 y 坐标
    std::vector<double> getSplineY() const;

private:
    int degree;                                    // B-Spline 曲线的度数
    Eigen::VectorXd knots;                         // 节点向量
    Eigen::VectorXd xControlPoints;                // x 控制点
    Eigen::VectorXd yControlPoints;                // y 控制点
    std::vector<std::vector<double>> splineBasis;  // 样条基函数矩阵
    std::vector<double> splineX;                   // 插值后的 x 坐标
    std::vector<double> splineY;                   // 插值后的 y 坐标
};

}  // namespace geometry
}  // namespace hymson3d
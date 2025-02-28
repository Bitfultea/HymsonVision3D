#include "PlaneDetection.h"

#include <opencv2/opencv.hpp>

#include "2D/Curve.h"

namespace hymson3d {
namespace core {

geometry::Plane::Ptr PlaneDetection::fit_a_plane(
        geometry::PointCloud& pointcloud) {
    Eigen::Vector3d centroid = pointcloud.GetCenter();

    // Calculate full 3x3 covariance matrix, excluding symmetries:
    double xx = 0, xy = 0, xz = 0;
    double yy = 0, yz = 0, zz = 0;

    for (auto pt : pointcloud.points_) {
        Eigen::Vector3d r = pt - centroid;
        xx += r.x() * r.x();
        xy += r.x() * r.y();
        xz += r.x() * r.z();
        yy += r.y() * r.y();
        yz += r.y() * r.z();
        zz += r.z() * r.z();
    }

    // TODO::Experiemnt with following
    // from https://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
    size_t n = pointcloud.points_.size();
    xx /= n;
    xy /= n;
    xz /= n;
    yy /= n;
    yz /= n;
    zz /= n;

    Eigen::Vector3d weighted_dir(0, 0, 0);
    // x direction
    double det_x = yy * zz - yz * yz;
    Eigen::Vector3d axis_dir_x(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
    double weight_x = det_x * det_x;
    if (weighted_dir.dot(axis_dir_x) < 0.0) weight_x = -weight_x;
    weighted_dir += axis_dir_x * weight_x;

    // y direction
    double det_y = xx * zz - xz * xz;
    Eigen::Vector3d axis_dir_y(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
    double weight_y = det_y * det_y;
    if (weighted_dir.dot(axis_dir_y) < 0.0) weight_y = -weight_y;
    weighted_dir += axis_dir_y * weight_y;

    // z direction
    double det_z = xx * yy - xy * xy;
    Eigen::Vector3d axis_dir_z(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
    double weight_z = det_z * det_z;
    if (weighted_dir.dot(axis_dir_z) < 0.0) weight_z = -weight_z;
    weighted_dir += axis_dir_z * weight_z;

    double norm = weighted_dir.norm();
    if (norm == 0) {
        LOG_ERROR("Invalid Plane Normal Detected!");
    }
    weighted_dir /= weighted_dir.norm();  // normaliszed
    double d = -weighted_dir.dot(centroid);

    Eigen::Vector4d plane_coeff(weighted_dir.x(), weighted_dir.y(),
                                weighted_dir.z(), d);
    geometry::Plane::Ptr plane = std::make_shared<geometry::Plane>();
    plane->coeff_ = plane_coeff;
    plane->normal_ = weighted_dir;
    return plane;
}

std::shared_ptr<geometry::BSpline> PlaneDetection::generate_a_curve(
        std::vector<Eigen::Vector2d> control_pts) {
    // 控制点 (x 和 y)
    int n = control_pts.size();
    Eigen::VectorXd x(n);
    Eigen::VectorXd y(n);

    for (int i = 0; i < control_pts.size(); ++i) {
        x(i) = control_pts[i](0);
        y(i) = control_pts[i](1);
    }

    // 创建样条对象
    std::shared_ptr<geometry::BSpline> spline =
            std::make_shared<geometry::BSpline>(3);  // 默认三次样条
    spline->setControlPoints(x, y);

    // 生成节点
    Eigen::VectorXd knots =
            spline->generateKnots(x.size(), x(0), x(x.size() - 1), 2);

    spline->setKnots(knots);

    // 生成样条基
    Eigen::VectorXd pts =
            Eigen::VectorXd::LinSpaced(100, x(0), x(x.size() - 1));
    Eigen::MatrixXd B = spline->generateSplineBasis(pts);

    return spline;
}

Eigen::VectorXd PlaneDetection::fit_a_curve(
        std::vector<Eigen::Vector2d> control_pts,
        int sampled_pts,
        int plot_id,
        bool debug_mode) {
    // 控制点 (x 和 y)
    int n = control_pts.size();
    Eigen::VectorXd x(n);
    Eigen::VectorXd y(n);

    for (int i = 0; i < control_pts.size(); ++i) {
        x(i) = control_pts[i](0);
        y(i) = control_pts[i](1);
    }

    // 打印控制点的维度x
    // std::cout << "Control points x size: " << x.size() << std::endl;
    // std::cout << "Control points y size: " << y.size() << std::endl;

    // 创建样条对象
    geometry::BSpline spline(3);  // 默认三次样条
    spline.setControlPoints(x, y);

    // 生成节点
    Eigen::VectorXd knots =
            spline.generateKnots(x.size(), x(0), x(x.size() - 1), 2);

    // 打印生成的节点
    // std::cout << "Generated knots size: " << knots.size() << std::endl;
    spline.setKnots(knots);

    // 生成样条基
    Eigen::VectorXd pts =
            Eigen::VectorXd::LinSpaced(100, x(0), x(x.size() - 1));
    Eigen::MatrixXd B = spline.generateSplineBasis(pts);

    // 打印样条基矩阵的维度
    // if (debug_mode)
    //     std::cout << "Spline basis B size: " << B.rows() << "x" << B.cols()
    //               << std::endl;

    std::vector<double> x_vec(x.data(), x.data() + x.size());
    std::vector<double> y_vec(y.data(), y.data() + y.size());

    // 绘制样条曲线
    Eigen::VectorXd spline_x = spline.getSplineX(pts);
    Eigen::VectorXd spline_y = spline.getSplineY(pts);
    // std::cout << "Spline X values: " << spline_x.transpose() << std::endl;
    // std::cout << "Spline Y values: " << spline_y.transpose() << std::endl;

    // 差值新的点
    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(
            sampled_pts, std::max(x.minCoeff(), 0.0), x.maxCoeff());
    Eigen::VectorXd v = spline.interpolate(u);
    // std::cout << x << std::endl;
    // std::cout << "-------------------------" << std::endl;
    // std::cout << v.transpose() << std::endl;
    // std::cout << u.transpose() << std::endl;
    std::vector<double> u_vec(u.data(), u.data() + u.size());
    std::vector<double> v_vec(v.data(), v.data() + v.size());

    std::vector<double> spline_x_vec(spline_x.data(),
                                     spline_x.data() + spline_x.size());
    std::vector<double> spline_y_vec(spline_y.data(),
                                     spline_y.data() + spline_y.size());

    if (debug_mode)
        plot_curve(x_vec, y_vec, u_vec, v_vec, spline_x_vec, spline_y_vec,
                   plot_id);

    return v;
}

void PlaneDetection::plot_curve(std::vector<double> x_vec,
                                std::vector<double> y_vec,
                                std::vector<double> spline_x_vec,
                                std::vector<double> spline_y_vec,
                                int plot_id) {
    // 使用 OpenCV 绘制曲线和插值点
    cv::Mat image = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
    image.setTo(cv::Scalar(255, 255, 255));

    // 将坐标缩放到图像大小
    double x_min = *std::min_element(x_vec.begin(), x_vec.end());
    double x_max = *std::max_element(x_vec.begin(), x_vec.end());
    double y_min = *std::min_element(y_vec.begin(), y_vec.end());
    double y_max = *std::max_element(y_vec.begin(), y_vec.end());

    for (size_t i = 0; i < x_vec.size(); ++i) {
        int x = static_cast<int>((x_vec[i] - x_min) / (x_max - x_min) * 800);
        int y = static_cast<int>(500 -
                                 (y_vec[i] - y_min) / (y_max - y_min) * 500);
        cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
    }

    // 绘制曲线
    for (size_t i = 0; i < spline_x_vec.size() - 1; ++i) {
        int x1 = static_cast<int>((spline_x_vec[i] - x_min) / (x_max - x_min) *
                                  800);
        int y1 = static_cast<int>(500 - (spline_y_vec[i] - y_min) /
                                                (y_max - y_min) * 500);
        int x2 = static_cast<int>((spline_x_vec[i + 1] - x_min) /
                                  (x_max - x_min) * 800);
        int y2 = static_cast<int>(500 - (spline_y_vec[i + 1] - y_min) /
                                                (y_max - y_min) * 500);
        cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2),
                 cv::Scalar(255, 0, 0), 1);
    }

    // 显示图像
    cv::imwrite("./bspline/Spline_Curve.jpg", image);
}

void PlaneDetection::plot_curve(std::vector<double> x_vec,
                                std::vector<double> y_vec,
                                std::vector<double> u_vec,
                                std::vector<double> v_vec,
                                std::vector<double> spline_x_vec,
                                std::vector<double> spline_y_vec,
                                int plot_id) {
    // 使用 OpenCV 绘制曲线和插值点
    cv::Mat image = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
    image.setTo(cv::Scalar(255, 255, 255));

    // 将坐标缩放到图像大小
    double x_min = *std::min_element(x_vec.begin(), x_vec.end());
    double x_max = *std::max_element(x_vec.begin(), x_vec.end());
    double y_min = *std::min_element(y_vec.begin(), y_vec.end());
    double y_max = *std::max_element(y_vec.begin(), y_vec.end());

    for (size_t i = 0; i < x_vec.size(); ++i) {
        int x = static_cast<int>((x_vec[i] - x_min) / (x_max - x_min) * 800);
        int y = static_cast<int>(500 -
                                 (y_vec[i] - y_min) / (y_max - y_min) * 500);
        cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
    }

    for (size_t i = 0; i < u_vec.size(); ++i) {
        int x = static_cast<int>((u_vec[i] - x_min) / (x_max - x_min) * 800);
        int y = static_cast<int>(500 -
                                 (v_vec[i] - y_min) / (y_max - y_min) * 500);
        cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
    }

    // 绘制曲线
    for (size_t i = 0; i < spline_x_vec.size() - 1; ++i) {
        int x1 = static_cast<int>((spline_x_vec[i] - x_min) / (x_max - x_min) *
                                  800);
        int y1 = static_cast<int>(500 - (spline_y_vec[i] - y_min) /
                                                (y_max - y_min) * 500);
        int x2 = static_cast<int>((spline_x_vec[i + 1] - x_min) /
                                  (x_max - x_min) * 800);
        int y2 = static_cast<int>(500 - (spline_y_vec[i + 1] - y_min) /
                                                (y_max - y_min) * 500);
        // std::cout << cv::Point(x1, y1) << " " << cv::Point(x2, y2) <<
        // std::endl;
        cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2),
                 cv::Scalar(255, 0, 0), 1);
    }

    // 显示图像
    cv::imwrite("./bspline/Spline_Curve_int" + std::to_string(plot_id) + ".jpg",
                image);
}

}  // namespace core
}  // namespace hymson3d
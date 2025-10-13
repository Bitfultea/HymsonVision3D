#include "PlaneDetection.h"

#include <opencv2/opencv.hpp>

#include "2D/Curve.h"
#include "Filter.h"
#include "open3d/TriangleMesh.h"

namespace hymson3d {
namespace core {

// Find the plane such that the summed squared distance from the
// plane to all points is minimized.
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

geometry::Plane::Ptr PlaneDetection::fit_a_plane(
        std::vector<Eigen::Vector3d>& points) {
    Eigen::Vector3d centroid(0, 0, 0);
    centroid = std::accumulate(points.begin(), points.end(), centroid);
    centroid /= double(points.size());

    // Calculate full 3x3 covariance matrix, excluding symmetries:
    double xx = 0, xy = 0, xz = 0;
    double yy = 0, yz = 0, zz = 0;

    for (auto pt : points) {
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
    size_t n = points.size();
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
    plane->inlier_points_ = points;
    plane->center_ = centroid;
    return plane;
}

geometry::Plane::Ptr PlaneDetection::fit_a_plane_ransc(
        geometry::PointCloud& cloud,
        const double distance_threshold /* = 0.01 */,
        const int ransac_n /* = 3 */,
        const int num_iterations /* = 100 */,
        const double probability /* = 0.99999999 */) {
    if (probability <= 0 || probability > 1) {
        LOG_ERROR("Probability must be > 0 and <= 1.0");
    }

    ransac_result ransac_result;
    geometry::Plane::Ptr best_plane = std::make_shared<geometry::Plane>();
    best_plane->coeff_ = Eigen::Vector4d::Zero();

    size_t num_points = cloud.points_.size();
    RandomSampler<size_t> sampler(num_points);

    // Pre-generate all random samples before entering the parallel region
    std::vector<std::vector<size_t>> all_sampled_indices;
    all_sampled_indices.reserve(num_iterations);
    for (int i = 0; i < num_iterations; i++) {
        all_sampled_indices.push_back(sampler(ransac_n));
    }

    // Return if ransac_n is less than the required plane model parameters.
    if (ransac_n < 3) {
        LOG_ERROR("ransac_n should be set to higher than or equal to 3.");
        // best_plane->inlier_idx_ = {};
        return best_plane;
    }
    if (num_points < size_t(ransac_n)) {
        LOG_ERROR("There must be at least 'ransac_n' points.");
        return best_plane;
    }

    // Use size_t here to avoid large integer which acceed max of int.
    size_t break_iteration = std::numeric_limits<size_t>::max();
    int iteration_count = 0;

#pragma omp parallel for
    for (int itr = 0; itr < num_iterations; itr++) {
        if ((size_t)iteration_count > break_iteration) {
            continue;
        }

        // Access the pre-generated sampled indices
        std::vector<size_t> inliers = all_sampled_indices[itr];

        // Fit model to num_model_parameters randomly selected points among the
        // inliers.
        Eigen::Vector4d plane_model;
        if (ransac_n == 3) {
            plane_model = open3d::geometry::TriangleMesh::ComputeTrianglePlane(
                    cloud.points_[inliers[0]], cloud.points_[inliers[1]],
                    cloud.points_[inliers[2]]);
        } else {
            core::Filter filter;
            std::shared_ptr<geometry::PointCloud> cloud_ptr =
                    std::make_shared<geometry::PointCloud>(cloud);
            auto pcd = filter.IndexDownSample(cloud_ptr, inliers);
            plane_model = fit_a_plane(*pcd)->coeff_;
        }

        if (plane_model.isZero(0)) {
            continue;
        }

        inliers.clear();
        auto this_result = evaluate_RANSAC_baseon_distance(
                cloud.points_, plane_model, inliers, distance_threshold);

#pragma omp critical
        {
            if (this_result.fitness_ > ransac_result.fitness_ ||
                (this_result.fitness_ == ransac_result.fitness_ &&
                 this_result.inlier_rmse_ < ransac_result.inlier_rmse_)) {
                ransac_result = this_result;
                best_plane->coeff_ = plane_model;
                if (ransac_result.fitness_ < 1.0) {
                    break_iteration =
                            std::min(log(1 - probability) /
                                             log(1 - pow(ransac_result.fitness_,
                                                         ransac_n)),
                                     (double)num_iterations);
                } else {
                    // Set break_iteration to 0 to force to break the loop.
                    break_iteration = 0;
                }
            }
            iteration_count++;
        }
    }

    // Find the final inliers using best_plane_model.
    std::vector<size_t> final_inliers;
    if (!best_plane->coeff_.isZero(0)) {
        for (size_t idx = 0; idx < cloud.points_.size(); ++idx) {
            Eigen::Vector4d point(cloud.points_[idx](0), cloud.points_[idx](1),
                                  cloud.points_[idx](2), 1);
            double distance = std::abs(best_plane->coeff_.dot(point));

            if (distance < distance_threshold) {
                final_inliers.emplace_back(idx);
            }
        }
    }

    // Improve best_plane_model using the final inliers.
    best_plane = fit_a_plane(cloud);
    best_plane->inlier_idx_ = final_inliers;

    LOG_DEBUG(
            "RANSAC | Inliers: {:d}, Fitness: {:e}, RMSE: {:e}, Iteration: "
            "{:d}",
            final_inliers.size(), ransac_result.fitness_,
            ransac_result.inlier_rmse_, iteration_count);

    return best_plane;
}

ransac_result PlaneDetection::evaluate_RANSAC_baseon_distance(
        const std::vector<Eigen::Vector3d>& points,
        const Eigen::Vector4d plane_model,
        std::vector<size_t>& inliers,
        double distance_threshold) {
    ransac_result result;

    double error = 0;
    for (size_t idx = 0; idx < points.size(); ++idx) {
        Eigen::Vector4d point(points[idx](0), points[idx](1), points[idx](2),
                              1);
        double distance = std::abs(plane_model.dot(point));

        if (distance < distance_threshold) {
            error += distance * distance;
            inliers.emplace_back(idx);
        }
    }

    size_t inlier_num = inliers.size();
    if (inlier_num == 0) {
        result.fitness_ = 0;
        result.inlier_rmse_ = 0;
    } else {
        result.fitness_ = (double)inlier_num / (double)points.size();
        result.inlier_rmse_ = std::sqrt(error / (double)inlier_num);
    }
    return result;
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
    //     std::cout << "Spline basis B size: " << B.rows() << "x" <<
    //     B.cols()
    //               << std::endl;

    std::vector<double> x_vec(x.data(), x.data() + x.size());
    std::vector<double> y_vec(y.data(), y.data() + y.size());

    // 绘制样条曲线
    Eigen::VectorXd spline_x = spline.getSplineX(pts);
    Eigen::VectorXd spline_y = spline.getSplineY(pts);
    // std::cout << "Spline X values: " << spline_x.transpose() <<
    // std::endl; std::cout << "Spline Y values: " << spline_y.transpose()
    // << std::endl;

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

std::vector<Eigen::Vector2d> PlaneDetection::resample_a_curve(
        std::vector<Eigen::Vector2d> control_pts,
        int sampled_pts,
        int plot_id,
        bool debug_mode) {
    int n = control_pts.size();
    Eigen::VectorXd x(n);
    Eigen::VectorXd y(n);

    for (int i = 0; i < control_pts.size(); ++i) {
        x(i) = control_pts[i](0);
        y(i) = control_pts[i](1);
    }

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
    //     std::cout << "Spline basis B size: " << B.rows() << "x" <<
    //     B.cols()
    //               << std::endl;

    std::vector<double> x_vec(x.data(), x.data() + x.size());
    std::vector<double> y_vec(y.data(), y.data() + y.size());

    // 绘制样条曲线
    Eigen::VectorXd spline_x = spline.getSplineX(pts);
    Eigen::VectorXd spline_y = spline.getSplineY(pts);

    Eigen::VectorXd u = Eigen::VectorXd::LinSpaced(
            sampled_pts, std::max(x.minCoeff(), 0.0), x.maxCoeff());
    Eigen::VectorXd v = spline.interpolate(u);

    std::vector<double> u_vec(u.data(), u.data() + u.size());  // x
    std::vector<double> v_vec(v.data(), v.data() + v.size());  // z

    std::vector<Eigen::Vector2d> resample_pts;
    resample_pts.resize(u_vec.size());
#pragma omp parallel
    for (int i = 0; i < u_vec.size(); i++) {
        // std::cout << "u= " << u_vec[i] << " v= " << v_vec[i] <<
        // std::endl;
        resample_pts[i] = Eigen::Vector2d(u_vec[i], v_vec[i]);
    }

    if (debug_mode) {
        // 将坐标缩放到图像大小
        double x_min = *std::min_element(x_vec.begin(), x_vec.end());
        double x_max = *std::max_element(x_vec.begin(), x_vec.end());
        double y_min = *std::min_element(y_vec.begin(), y_vec.end());
        double y_max = *std::max_element(y_vec.begin(), y_vec.end());
        cv::Mat image = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
        image.setTo(cv::Scalar(255, 255, 255));
        for (size_t i = 0; i < u_vec.size(); ++i) {
            int x = static_cast<int>((u_vec[i] - x_min) / (x_max - x_min) *
                                     800);
            int y = static_cast<int>(500 - (v_vec[i] - y_min) /
                                                   (y_max - y_min) * 500);
            cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
        }
        cv::imwrite(
                "./bspline/Spline_Curve_int" + std::to_string(plot_id) + ".jpg",
                image);
    }

    return resample_pts;
}

void PlaneDetection::plot_curve(std::vector<double> x_vec,
                                std::vector<double> y_vec,
                                std::vector<double> spline_x_vec,
                                std::vector<double> spline_y_vec,
                                int plot_id) {
    // 使用 OpenCV 绘制曲线和插值点
    cv::Mat image = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
    image.setTo(cv::Scalar(255, 255, 255));

    cv::Mat bg = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
    bg.setTo(cv::Scalar(255, 255, 255));

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

        if (i > 0 && i < spline_x_vec.size() - 1) {
            double derivative = (spline_y_vec[i + 1] - spline_y_vec[i]) /
                                (spline_x_vec[i + 1] - spline_x_vec[i]);
            double dy = spline_y_vec[i + 1] - spline_y_vec[i - 1];
            double dx = spline_x_vec[i + 1] - spline_x_vec[i - 1];
            double q = spline_x_vec[i] - 0.25 * dx;
            double p = spline_x_vec[i] + 0.25 * dx;
            double m = spline_x_vec[i] - 0.25 * dy;
            double n = spline_x_vec[i] + 0.25 * dy;

            int x3 = static_cast<int>((q - x_min) / (x_max - x_min) * 800);
            int y3 = 10 + static_cast<int>(500 -
                                           (p - y_min) / (y_max - y_min) * 500);
            int x4 = static_cast<int>((m - x_min) / (x_max - x_min) * 800);
            int y4 = 10 + static_cast<int>(500 -
                                           (n - y_min) / (y_max - y_min) * 500);
            cv::arrowedLine(image, cv::Point(x3, y3), cv::Point(x4, y4),
                            cv::Scalar(0, 0, 0), 2);
        }
    }

    // 显示图像
    cv::imwrite("./bspline/Spline_Curve.jpg", image);
    // cv::imwrite("./bspline/derivative.jpg", bg);
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

    cv::Mat bg = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
    bg.setTo(cv::Scalar(255, 255, 255));

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
        // if (i < u_vec.size() - 1) {
        //     double derivative =
        //             (v_vec[i + 1] - v_vec[i]) / (u_vec[i + 1] -
        //             u_vec[i]);
        //     int d = static_cast<int>(500 -
        //                              (derivative - (-10)) / (10 - (-10))
        //                              * 500);
        //     cv::circle(bg, cv::Point(x1, 1e4 * derivative), 2,
        //                cv::Scalar(0, 0, 0), -1);
        // }

        if (i > 0 && i < u_vec.size() - 1) {
            double derivative =
                    (v_vec[i + 1] - v_vec[i]) / (u_vec[i + 1] - u_vec[i]);
            double dy = v_vec[i + 1] - v_vec[i - 1];
            double dx = u_vec[i + 1] - u_vec[i - 1];
            double q = u_vec[i] - 0.25 * dx;
            double p = u_vec[i] + 0.25 * dx;
            double m = u_vec[i] - 0.25 * dy;
            double n = u_vec[i] + 0.25 * dy;

            int x3 = static_cast<int>((q - x_min) / (x_max - x_min) * 800);
            int y3 =
                    static_cast<int>(500 - (p - y_min) / (y_max - y_min) * 500);
            int x4 = static_cast<int>((m - x_min) / (x_max - x_min) * 800);
            int y4 =
                    static_cast<int>(500 - (n - y_min) / (y_max - y_min) * 500);
            cv::arrowedLine(image, cv::Point(x3, y3), cv::Point(x4, y4),
                            cv::Scalar(0, 0, 0), 2);
        }
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
    cv::line(bg, cv::Point(0, 250), cv::Point(800, 250), cv::Scalar(255, 0, 0),
             1);

    // 显示图像
    cv::imwrite("./bspline/Spline_Curve_int" + std::to_string(plot_id) + ".jpg",
                image);
    cv::imwrite("./bspline/Spline_Dev_int" + std::to_string(plot_id) + ".jpg",
                bg);
}

}  // namespace core
}  // namespace hymson3d
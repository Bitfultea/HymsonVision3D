#include "Feature.h"

#include <pcl/features/feature.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>  // 法线
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>

#include "Converter.h"

namespace hymson3d {
namespace core {
namespace feature {

typedef pcl::PointCloud<pcl::FPFHSignature33> fpfhFeature;

Eigen::MatrixXf compute_fpfh(geometry::PointCloud& cloud) {
    LOG_DEBUG("Start Computing FPFH feature");
    Eigen::MatrixXf fpfh_data;
    fpfh_data.resize(cloud.points_.size(), 33);
    fpfh_data.setZero();

    // Convert to PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZ>);

    if (!cloud.HasNormals()) {
        converter::to_pcl_pointcloud(cloud, pcl_cloud);
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
        normal_estimator.setSearchMethod(tree);
        normal_estimator.setViewPoint(0, 0, std::numeric_limits<float>::max());
        normal_estimator.setInputCloud(pcl_cloud);
        normal_estimator.setKSearch(30);
        // normal_estimator.setRadiusSearch(30);
        normal_estimator.compute(*normals);
    } else {
        LOG_DEBUG("Already have normals. Use previsous computed normals");
        converter::to_pcl_pointcloud(cloud, pcl_cloud, normals);
    }

    // Compute FPFH feature
    fpfhFeature::Ptr fpfh(new fpfhFeature);
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>
            est_fpfh;
    est_fpfh.setNumberOfThreads(8);
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree4 (new
    // pcl::search::KdTree<pcl::PointXYZ> ());
    est_fpfh.setInputCloud(pcl_cloud);
    est_fpfh.setInputNormals(normals);
    est_fpfh.setSearchMethod(tree);
    // est_fpfh.setKSearch(12);
    est_fpfh.setRadiusSearch(0.05);
    est_fpfh.compute(*fpfh);

#pragma omp parallel for
    for (int i = 0; i < cloud.points_.size(); i++) {
        for (int j = 0; j < 33; ++j) {
            fpfh_data(i, j) = fpfh->points[i].histogram[j];
        }
    }

    LOG_DEBUG("Completing Computing FPFH feature");
    return fpfh_data;
}

/**
 * @brief 检测图像中的绿色圆环并返回其中心坐标
 *
 * 该函数通过在HSV色彩空间中识别绿色区域来检测图像中的绿色圆环。
 * 它会找到最大的绿色轮廓，并拟合一个椭圆来确定圆环的位置。
 *
 * @param img 输入的彩色图像(BGR格式)
 * @param debug_mode 是否启用调试模式，启用时会在原图上绘制检测结果并保存到文件
 * @return std::pair<bool, cv::Point2f> 第一个元素表示是否成功检测到圆环，
 *         第二个元素是圆环中心点的坐标。如果未检测到圆环，则返回(false, (0,0))
 */
std::pair<bool, cv::Point2f> detect_green_ring(const cv::Mat& img,
                                               bool debug_mode) {
    if (img.empty()) {
        LOG_ERROR("Image is empty");
        return {false, cv::Point2f(0, 0)};
    }

    // H(色相) : 35 - 85(涵盖浅绿到深绿)
    // S(饱和度) : 43 - 255(只要不是太灰)
    // V(亮度) : 46 - 255(只要不是太黑)
    cv::Scalar lower_green(cv::Scalar(35, 43, 46));
    cv::Scalar upper_green(cv::Scalar(85, 255, 255));
    cv::Mat kernel(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, lower_green, upper_green, mask);

    // denoise
    cv::Mat maskClean;
    cv::morphologyEx(mask, maskClean, cv::MORPH_OPEN, kernel);
    // cv::imwrite("mask.jpg", maskClean);

    // find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(maskClean, contours, hierarchy, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    if (!contours.empty()) {
        std::vector<cv::Point> best_contour;
        std::sort(contours.begin(), contours.end(),
                  [](const std::vector<cv::Point>& a,
                     const std::vector<cv::Point>& b) {
                      return a.size() > b.size();
                  });

        cv::RotatedRect best_ellipse;
        bool found = false;
        for (const auto& contour : contours) {
            cv::RotatedRect ellipse = cv::fitEllipse(contour);
            double ratio = ellipse.size.width / ellipse.size.height;
            // std::cout << "ratio: " << ratio << std::endl;
            if (ratio > 0.75 && ratio < 1.3) {
                best_ellipse = ellipse;
                found = true;
                break;
            }
        }
        if (found) best_ellipse = cv::fitEllipse(contours[0]);

        cv::Point2f center = best_ellipse.center;

        // draw the ellipse
        if (debug_mode) {
            cv::ellipse(img, best_ellipse, cv::Scalar(0, 255, 0),
                        2);  // ellipse
            cv::circle(img, center, 5, cv::Scalar(0, 0, 255),
                       -1);  // centre
            cv::imwrite("circle_location.jpg", img);
        }
        LOG_INFO("圆环中心坐标: ({},{})", static_cast<int>(center.x),
                 static_cast<int>(center.y));
        LOG_INFO("圆环半径: {}", best_ellipse.size.width / 2);

        return {true, center};
    }

    return {false, cv::Point2f(0, 0)};
}

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
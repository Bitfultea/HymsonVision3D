#include "GapStepDetection.h"

#include <math.h>

#include <opencv2/opencv.hpp>

#include "Converter.h"
#include "Curvature.h"
#include "Feature.h"
#include "FileSystem.h"
#include "MathTool.h"
#include "PlaneDetection.h"

namespace hymson3d {
namespace pipeline {

void GapStepDetection::detect_gap_step(
        std::shared_ptr<geometry::PointCloud> cloud,
        Eigen::Vector3d transformation_matrix,
        bool debug_mode) {
    // debug mode
    if (debug_mode) {
        utility::filesystem::MakeDirectory("./bspline");
    }

    // LOG_DEBUG("Slice along Y-axis");
    //  slice along y axis
    slice_along_y(cloud, transformation_matrix);

    // bspline interpolation
    double height_threshold = 0.01;
    lineSegments corners;
    bspline_interpolation(cloud, height_threshold, corners, debug_mode);

    // calculate the gap step result
    double gap_step = 0.0, step_width = 0.0;
    calculate_gap_step(corners, gap_step, step_width);
}

void GapStepDetection::detect_gap_step_dll(
        std::shared_ptr<geometry::PointCloud> cloud,
        Eigen::Vector3d transformation_matrix,
        double& gap_step,
        double& step_width,
        bool debug_mode) {
    // debug mode
    if (debug_mode) {
        utility::filesystem::MakeDirectory("./bspline");
    }
    // std::cout << "1" << std::endl;

    // LOG_DEBUG("Slice along Y-axis");
    //  slice along y axis
    slice_along_y(cloud, transformation_matrix);
    // std::cout << "2" << std::endl;

    // bspline interpolation
    double height_threshold = 0.01;
    lineSegments corners;
    bspline_interpolation(cloud, height_threshold, corners, debug_mode);
    // std::cout << "3" << std::endl;

    // calculate the gap step result
    // double gap_step = 0.0, step_width = 0.0;
    calculate_gap_step(corners, gap_step, step_width);
    // std::cout << "4" << std::endl;
}

void GapStepDetection::detect_gap_step_dll_plot(
        std::shared_ptr<geometry::PointCloud> cloud,
        Eigen::Vector3d transformation_matrix,
        double& gap_step,
        double& step_width,
        double& height_threshold,
        std::vector<std::vector<double>>& temp_res,
        std::string& debug_path,
        bool debug_mode) {
    // debug mode
    if (debug_mode) {
        utility::filesystem::MakeDirectory_dll(
                debug_path);  //"C:\\Users\\Administrator\\Desktop\\res\\bspline"
    }
    // std::cout << "1" << std::endl;

    // LOG_DEBUG("Slice along Y-axis");
    //  slice along y axis
    slice_along_y(cloud, transformation_matrix);
    // std::cout << "2" << std::endl;

    // bspline interpolation
    // double height_threshold = 0.01;
    lineSegments corners;
    // bspline_interpolation(cloud, height_threshold, corners, debug_mode);
    // std::cout << "3" << std::endl;
    bspline_interpolation_dll(cloud, height_threshold, corners, debug_path,
                              debug_mode);
    // std::cout << "3" << std::endl;

    // calculate the gap step result
    // double gap_step = 0.0, step_width = 0.0;
    calculate_gap_step_dll_plot(corners, gap_step, step_width, temp_res);
    // std::cout << "4" << std::endl;
}

void GapStepDetection::slice_along_y(geometry::PointCloud::Ptr cloud,
                                     Eigen::Vector3d transformation_matrix) {
    bool has_normals = cloud->HasNormals();
    // std::cout << transformation_matrix << std::endl;
    // std::cout << "has points: " << cloud->points_.size() << std::endl;
    if (has_normals) {
        Eigen::Vector3d min_bound = cloud->GetMinBound();
        Eigen::Vector3d max_bound = cloud->GetMaxBound();
        int num_slice = (int)(((max_bound.y() - min_bound.y()) /
                               transformation_matrix.y()) +
                              1);
        std::vector<double> y_slice_peaks(num_slice, 0);
        cloud->y_slice_peaks = y_slice_peaks;
        cloud->y_slices_.resize(num_slice);
        cloud->ny_slices_.resize(num_slice);
        cloud->y_slice_idxs.resize(num_slice);
        for (size_t j = 0; j < cloud->points_.size(); j++) {
            auto pt = cloud->points_[j];
            auto n = cloud->normals_[j];
            int slice_idx =
                    (int)((pt.y() - min_bound.y()) / transformation_matrix.y());
            if (cloud->y_slice_peaks[slice_idx] <= pt.z()) {
                cloud->y_slice_peaks[slice_idx] = pt.z();
            }
            cloud->y_slices_[slice_idx].emplace_back(
                    Eigen::Vector2d(pt.x(), pt.z()));
            cloud->ny_slices_[slice_idx].emplace_back(
                    Eigen::Vector3d(n.x(), n.y(), n.z()));
            cloud->y_slice_idxs[slice_idx].emplace_back(j);
        }
    } else {
        Eigen::Vector3d min_bound = cloud->GetMinBound();
        Eigen::Vector3d max_bound = cloud->GetMaxBound();
        int num_slice = (int)((max_bound.y() / transformation_matrix.y()) -
                              (min_bound.y()) / transformation_matrix.y() + 1);

        std::vector<double> y_slice_peaks(num_slice, 0);
        cloud->y_slice_peaks = y_slice_peaks;
        cloud->y_slices_.resize(num_slice);
        cloud->y_slice_idxs.reserve(num_slice);
        auto pre_y = cloud->points_[0].y();
        int slice_idx = 0;

        for (size_t j = 0; j < cloud->points_.size(); j++) {
            auto pt = cloud->points_[j];
            if (pt.y() != pre_y) {
                slice_idx += 1;
                pre_y = pt.y();
            }
            if (cloud->y_slice_peaks[slice_idx] <= pt.z()) {
                cloud->y_slice_peaks[slice_idx] = pt.z();
            }
            cloud->y_slices_[slice_idx].emplace_back(
                    Eigen::Vector2d(pt.x(), pt.z()));
        }
    }
}

void GapStepDetection::bspline_interpolation(geometry::PointCloud::Ptr cloud,
                                             double height_threshold,
                                             lineSegments& corners,
                                             bool debug_mode) {
    // use common part to fit a curve
    core::PlaneDetection plane_detector;
    int sampled_pts = 100;
    std::vector<double> step_height;
    step_height.resize(cloud->y_slices_.size());
    corners.resize(cloud->y_slices_.size());

#pragma omp parallel for
    for (int i = 0; i < cloud->y_slices_.size(); i++) {
        if (cloud->y_slices_[i].size() == 0) continue;
        std::vector<Eigen::Vector2d> resampled_pts =
                plane_detector.resample_a_curve(cloud->y_slices_[i],
                                                sampled_pts, i, false);
        // compute the derivative
        std::vector<std::vector<Eigen::Vector2d>> groups =
                group_by_derivative(resampled_pts);

        std::vector<std::vector<Eigen::Vector2d>> filter_groups =
                statistics_filter(groups);

        lineSegments lines = line_segment(filter_groups);
        step_height[i] = std::abs(lines[0].first.y() - lines[1].first.y());

        std::vector<std::vector<Eigen::Vector2d>> intersections;
        compute_step_width(resampled_pts, lines, intersections,
                           height_threshold);
        // put two corners corresponding to the slice to container
        corners[i] = std::make_pair(intersections[2][0], intersections[2][1]);

        if (debug_mode) {
            plot_clusters(resampled_pts, filter_groups, lines, intersections,
                          i);
        }
    }

    // return sampled_map;
}

void GapStepDetection::bspline_interpolation_dll(
        geometry::PointCloud::Ptr cloud,
        double height_threshold,
        lineSegments& corners,
        std::string& debug_path,
        bool debug_mode) {
    // use common part to fit a curve
    core::PlaneDetection plane_detector;
    int sampled_pts = 100;
    std::vector<double> step_height;
    step_height.resize(cloud->y_slices_.size());
    corners.resize(cloud->y_slices_.size());

#pragma omp parallel for
    for (int i = 0; i < cloud->y_slices_.size(); i++) {
        if (cloud->y_slices_[i].size() == 0) continue;
        std::vector<Eigen::Vector2d> resampled_pts =
                plane_detector.resample_a_curve(cloud->y_slices_[i],
                                                sampled_pts, i, false);
        // compute the derivative
        std::vector<std::vector<Eigen::Vector2d>> groups =
                group_by_derivative(resampled_pts);

        std::vector<std::vector<Eigen::Vector2d>> filter_groups =
                statistics_filter(groups);

        lineSegments lines = line_segment(filter_groups);
        step_height[i] = std::abs(lines[0].first.y() - lines[1].first.y());

        std::vector<std::vector<Eigen::Vector2d>> intersections;
        compute_step_width(resampled_pts, lines, intersections,
                           height_threshold);
        // put two corners corresponding to the slice to container
        corners[i] = std::make_pair(intersections[2][0], intersections[2][1]);

        if (debug_mode) {
            plot_clusters_dll(resampled_pts, filter_groups, lines,
                              intersections, debug_path, i);
        }
    }

    // return sampled_map;
}

std::vector<std::vector<Eigen::Vector2d>> GapStepDetection::group_by_derivative(
        std::vector<Eigen::Vector2d>& sampled_pts) {
    std::vector<Eigen::Vector2d> horiz_pts;
    for (int i = 0; i < sampled_pts.size(); i++) {
        double derivative;
        if (i == 0) {
            derivative = (sampled_pts[i + 1](1) - sampled_pts[i](1)) /
                         (sampled_pts[i + 1](0) - sampled_pts[i](0));
        } else if (i == sampled_pts.size() - 1) {
            derivative = (sampled_pts[i](1) - sampled_pts[i - 1](1)) /
                         (sampled_pts[i](0) - sampled_pts[i - 1](0));
        } else {
            derivative = (sampled_pts[i + 1](1) - sampled_pts[i - 1](1)) /
                         (sampled_pts[i + 1](0) - sampled_pts[i - 1](0));
        }
        if (derivative > -0.15 && derivative < 0.15) {
            horiz_pts.emplace_back(sampled_pts[i]);
        }
    }
    // apply k-mean
    cv::Mat data(horiz_pts.size(), 2, CV_32F);
    for (int i = 0; i < horiz_pts.size(); ++i) {
        data.at<float>(i, 0) = horiz_pts[i](0);
        data.at<float>(i, 1) = horiz_pts[i](1);
    }

    // Set K-means random seed
    cv::theRNG().state = 42;
    // Perform K-means clustering
    cv::Mat labels;
    cv::Mat centers;
    cv::kmeans(data, 2, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                100, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    // Collect points in each cluster
    std::vector<std::vector<Eigen::Vector2d>> clusters(2);
    for (int i = 0; i < horiz_pts.size(); ++i) {
        int cluster_idx = labels.at<int>(i);
        clusters[cluster_idx].emplace_back(horiz_pts[i]);
    }

    // statistics filter
    return clusters;
}

std::vector<std::vector<Eigen::Vector2d>> GapStepDetection::statistics_filter(
        std::vector<std::vector<Eigen::Vector2d>>& clusters) {
    std::vector<std::vector<Eigen::Vector2d>> filter_group_pts;
    for (int i = 0; i < clusters.size(); i++) {
        Eigen::Vector2d mean(0, 0);
        mean = std::accumulate(clusters[i].begin(), clusters[i].end(),
                               Eigen::Vector2d(0, 0));
        mean /= double(clusters[i].size());
        double sum_sq_diff_x = 0.0;
        double sum_sq_diff_y = 0.0;
        for (const auto& pt : clusters[i]) {
            sum_sq_diff_x += (pt.x() - mean.x()) * (pt.x() - mean.x());
            sum_sq_diff_y += (pt.y() - mean.y()) * (pt.y() - mean.y());
        }
        double std_dev_x = std::sqrt(sum_sq_diff_x / clusters[i].size());
        double std_dev_y = std::sqrt(sum_sq_diff_y / clusters[i].size());
        double threshold_x = 1.5 * std_dev_x;
        double threshold_y = 1.2 * std_dev_y;

        // Filter points
        std::vector<Eigen::Vector2d> filtered_cluster;
        for (const auto& pt : clusters[i]) {
            if (std::abs(pt.x() - mean.x()) <= threshold_x &&
                std::abs(pt.y() - mean.y()) <= threshold_y) {
                filtered_cluster.push_back(pt);
            }
        }
        filter_group_pts.push_back(filtered_cluster);
    }
    return filter_group_pts;
}

void GapStepDetection::plot_clusters(
        std::vector<Eigen::Vector2d>& resampled_pts,
        std::vector<std::vector<Eigen::Vector2d>>& clusters,
        lineSegments& line_segs,
        std::vector<std::vector<Eigen::Vector2d>> intersections,
        int img_id) {
    cv::Mat bg = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
    bg.setTo(cv::Scalar(255, 255, 255));
    std::vector<double> x_vec;
    std::vector<double> y_vec;

    for (int i = 0; i < resampled_pts.size(); i++) {
        x_vec.push_back(resampled_pts[i].x());
        y_vec.push_back(resampled_pts[i].y());
    }

    double x_min = *std::min_element(x_vec.begin(), x_vec.end());
    double x_max = *std::max_element(x_vec.begin(), x_vec.end());
    double y_min = *std::min_element(y_vec.begin(), y_vec.end());
    double y_max = *std::max_element(y_vec.begin(), y_vec.end());

    for (size_t i = 0; i < x_vec.size(); ++i) {
        int x = static_cast<int>((x_vec[i] - x_min) / (x_max - x_min) * 800);
        int y = static_cast<int>(500 -
                                 (y_vec[i] - y_min) / (y_max - y_min) * 500);
        cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
    }

    for (int i = 0; i < clusters.size(); i++) {
        if (i == 0) {
            double left_x = line_segs[i].first.x();
            double right_x = line_segs[i].second.x();
            double mean_z = line_segs[i].first.y();
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
                // std::cout << pt.x() << " " << pt.y() << std::endl;
                // std::cout << x << " " << y << std::endl;
            }
            int line_x_left =
                    static_cast<int>((left_x - x_min) / (x_max - x_min) * 800);
            int line_x_right =
                    static_cast<int>((right_x - x_min) / (x_max - x_min) * 800);
            int line_z = static_cast<int>(500 - (mean_z - y_min) /
                                                        (y_max - y_min) * 500);
            cv::line(bg, cv::Point(line_x_left, line_z),
                     cv::Point(line_x_right, line_z), cv::Scalar(0, 0, 255), 1);
        } else {
            // i == 1
            double left_x = line_segs[i].first.x();
            double right_x = line_segs[i].second.x();
            double mean_z = line_segs[i].first.y();
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), -1);
            }
            int line_x_left =
                    static_cast<int>((left_x - x_min) / (x_max - x_min) * 800);
            int line_x_right =
                    static_cast<int>((right_x - x_min) / (x_max - x_min) * 800);
            int line_z = static_cast<int>(500 - (mean_z - y_min) /
                                                        (y_max - y_min) * 500);
            cv::line(bg, cv::Point(line_x_left, line_z),
                     cv::Point(line_x_right, line_z), cv::Scalar(255, 0, 0), 1);
        }
    }

    for (int i = 0; i < intersections.size(); i++) {
        if (i == 0) {
            int tmp_y = 0;
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(0, 0, 255),
                               cv::MARKER_STAR, 10);
                tmp_y = y;
                // std::cout << "Line: " << std::endl;
                // std::cout << pt.x() << " " << pt.y() << std::endl;
                // std::cout << x << " " << y << std::endl;
            }
            if (intersections[0].size() > 0)
                cv::line(bg, cv::Point(0, tmp_y), cv::Point(799, tmp_y),
                         cv::Scalar(0, 0, 255), 1);

        } else if (i == 1) {
            int tmp_y = 0;
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(255, 0, 0),
                               cv::MARKER_STAR, 10);
                tmp_y = y;
            }
            if (intersections[1].size() > 0)
                cv::line(bg, cv::Point(0, tmp_y), cv::Point(799, tmp_y),
                         cv::Scalar(255, 0, 0), 1);
        } else {
            // egde points i== 2
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(255, 0, 0),
                               cv::MARKER_DIAMOND, 10);
            }
        }
    }
    cv::imwrite("./bspline/group_pts" + std::to_string(img_id) + ".jpg", bg);
}

void GapStepDetection::plot_clusters_dll(
        std::vector<Eigen::Vector2d>& resampled_pts,
        std::vector<std::vector<Eigen::Vector2d>>& clusters,
        lineSegments& line_segs,
        std::vector<std::vector<Eigen::Vector2d>> intersections,
        std::string& debug_path,
        int img_id) {
    cv::Mat bg = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
    bg.setTo(cv::Scalar(255, 255, 255));
    std::vector<double> x_vec;
    std::vector<double> y_vec;

    for (int i = 0; i < resampled_pts.size(); i++) {
        x_vec.push_back(resampled_pts[i].x());
        y_vec.push_back(resampled_pts[i].y());
    }

    double x_min = *std::min_element(x_vec.begin(), x_vec.end());
    double x_max = *std::max_element(x_vec.begin(), x_vec.end());
    double y_min = *std::min_element(y_vec.begin(), y_vec.end());
    double y_max = *std::max_element(y_vec.begin(), y_vec.end());

    for (size_t i = 0; i < x_vec.size(); ++i) {
        int x = static_cast<int>((x_vec[i] - x_min) / (x_max - x_min) * 800);
        int y = static_cast<int>(500 -
                                 (y_vec[i] - y_min) / (y_max - y_min) * 500);
        cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
    }

    for (int i = 0; i < clusters.size(); i++) {
        if (i == 0) {
            double left_x = line_segs[i].first.x();
            double right_x = line_segs[i].second.x();
            double mean_z = line_segs[i].first.y();
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
                // std::cout << pt.x() << " " << pt.y() << std::endl;
                // std::cout << x << " " << y << std::endl;
            }
            int line_x_left =
                    static_cast<int>((left_x - x_min) / (x_max - x_min) * 800);
            int line_x_right =
                    static_cast<int>((right_x - x_min) / (x_max - x_min) * 800);
            int line_z = static_cast<int>(500 - (mean_z - y_min) /
                                                        (y_max - y_min) * 500);
            cv::line(bg, cv::Point(line_x_left, line_z),
                     cv::Point(line_x_right, line_z), cv::Scalar(0, 0, 255), 1);
        } else {
            // i == 1
            double left_x = line_segs[i].first.x();
            double right_x = line_segs[i].second.x();
            double mean_z = line_segs[i].first.y();
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), -1);
            }
            int line_x_left =
                    static_cast<int>((left_x - x_min) / (x_max - x_min) * 800);
            int line_x_right =
                    static_cast<int>((right_x - x_min) / (x_max - x_min) * 800);
            int line_z = static_cast<int>(500 - (mean_z - y_min) /
                                                        (y_max - y_min) * 500);
            cv::line(bg, cv::Point(line_x_left, line_z),
                     cv::Point(line_x_right, line_z), cv::Scalar(255, 0, 0), 1);
        }
    }

    for (int i = 0; i < intersections.size(); i++) {
        if (i == 0) {
            int tmp_y = 0;
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(0, 0, 255),
                               cv::MARKER_STAR, 10);
                tmp_y = y;
                // std::cout << "Line: " << std::endl;
                // std::cout << pt.x() << " " << pt.y() << std::endl;
                // std::cout << x << " " << y << std::endl;
            }
            if (intersections[0].size() > 0)
                cv::line(bg, cv::Point(0, tmp_y), cv::Point(799, tmp_y),
                         cv::Scalar(0, 0, 255), 1);

        } else if (i == 1) {
            int tmp_y = 0;
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(255, 0, 0),
                               cv::MARKER_STAR, 10);
                tmp_y = y;
            }
            if (intersections[1].size() > 0)
                cv::line(bg, cv::Point(0, tmp_y), cv::Point(799, tmp_y),
                         cv::Scalar(255, 0, 0), 1);
        } else {
            // egde points i== 2
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(255, 0, 0),
                               cv::MARKER_DIAMOND, 10);
            }
        }
    }
    // std::cout << "debug_path:"
    //           << debug_path + "group_pts" + std::to_string(img_id) + ".jpg"
    //           << std::endl;
    cv::imwrite(debug_path + "group_pts" + std::to_string(img_id) + ".jpg", bg);
}

GapStepDetection::lineSegments GapStepDetection::line_segment(
        std::vector<std::vector<Eigen::Vector2d>>& pt_groups) {
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> res;
    for (int i = 0; i < pt_groups.size(); i++) {
        double left_x = std::numeric_limits<double>::max();
        double right_x = 0;
        double mean_z = 0;
        for (auto pt : pt_groups[i]) {
            if (pt.x() < left_x) left_x = pt.x();
            if (pt.x() > right_x) right_x = pt.x();
            mean_z += pt.y();
        }
        mean_z = mean_z / pt_groups[i].size();
        res.push_back(std::make_pair(Eigen::Vector2d(left_x, mean_z),
                                     Eigen::Vector2d(right_x, mean_z)));
    }
    if (res[0].second.x() > res[1].first.x()) std::swap(res[1], res[0]);
    return res;
}

void GapStepDetection::compute_step_width(
        std::vector<Eigen::Vector2d>& resampled_pts,
        lineSegments& line_segs,
        std::vector<std::vector<Eigen::Vector2d>>& intersections,
        double height_threshold) {
    std::pair<Eigen::Vector2d, Eigen::Vector2d> left_line = line_segs[0];
    std::pair<Eigen::Vector2d, Eigen::Vector2d> right_line = line_segs[1];

    double left_height = left_line.first.y() - height_threshold;
    double right_height = right_line.first.y() - height_threshold;
    std::vector<Eigen::Vector2d> left_intersections;
    std::vector<Eigen::Vector2d> right_intersections;
    std::vector<std::vector<Eigen::Vector2d>> res;
    int lowest_idx = 0;
    double upper_bound = (-1) * std::numeric_limits<double>::max();
    double lower_bound = std::numeric_limits<double>::max();

    for (int i = 0; i < resampled_pts.size() - 1; i++) {
        auto pt = resampled_pts[i];
        auto next_pt = resampled_pts[i + 1];
        if (pt.y() > upper_bound) upper_bound = pt.y();
        if (pt.y() < lower_bound) lower_bound = pt.y();
        if ((pt.y() - left_height) * (next_pt.y() - left_height) < 0) {
            double denominator = next_pt.y() - pt.y();
            double t = 0.0;
            if (denominator != 0) {
                t = (left_height - pt.y()) / denominator;
            }
            double u = pt.x() + t * (next_pt.x() - pt.x());
            left_intersections.push_back(Eigen::Vector2d(u, left_height));
        }
        if ((pt.y() - right_height) * (next_pt.y() - right_height) < 0) {
            double denominator = next_pt.y() - pt.y();
            double t = 0.0;
            if (denominator != 0) {
                t = (right_height - pt.y()) / denominator;
            }
            double u = pt.x() + t * (next_pt.x() - pt.x());
            right_intersections.push_back(Eigen::Vector2d(u, right_height));
        }
        if (pt.y() < resampled_pts[lowest_idx].y()) lowest_idx = i;
    }
    // std::cout << upper_bound << " " << lower_bound << std::endl;
    if (left_intersections.size() == 0) {
        // not intersections
        auto u = left_line.second;
        left_intersections.push_back(u);
    }
    if (right_intersections.size() == 0) {
        // not intersections
        auto u = right_line.first;
        right_intersections.push_back(u);
    }
    intersections.emplace_back(left_intersections);
    intersections.emplace_back(right_intersections);

    Eigen::Vector2d mid_low = resampled_pts[lowest_idx];

    Eigen::Vector2d left_pt{0, 0};
    double left_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < left_intersections.size(); i++) {
        double dist = mid_low.x() - left_intersections[i].x();
        if (dist < 0) continue;
        if (left_intersections[i].x() > left_pt.x() && dist < left_dist) {
            left_pt = left_intersections[i];
            left_dist = dist;
        }
    }

    Eigen::Vector2d right_pt{std::numeric_limits<double>::max(),
                             std::numeric_limits<double>::max()};
    double right_dist = std::numeric_limits<double>::max();
    for (int i = 0; i < right_intersections.size(); i++) {
        double dist = right_intersections[i].x() - mid_low.x();
        if (dist < 0) continue;
        if (right_intersections[i].x() < right_pt.x() && dist < right_dist) {
            right_pt = right_intersections[i];
            right_dist = dist;
        }
    }
    std::vector<Eigen::Vector2d> edge_pts;
    edge_pts.emplace_back(left_pt);
    edge_pts.emplace_back(right_pt);
    intersections.emplace_back(edge_pts);
}

void GapStepDetection::calculate_gap_step(lineSegments& corners,
                                          double& gap_step,
                                          double& step_width) {
    double sum_width = 0.0;
    double sum_height = 0.0;
    int exception_count = 0;
    for (int i = 0; i < corners.size(); i++) {
        double temp_x = abs(corners[i].second.x() - corners[i].first.x());
        double temp_y = abs(corners[i].second.y() - corners[i].first.y());
        // exception
        if (temp_x > 100 || temp_y > 100) {
            exception_count++;
            continue;
        }

        sum_width += temp_x;
        sum_height += temp_y;
        // std::cout << i <<" " << corners[i].second.x() - corners[i].first.x()
        // << " "<<corners[i].second.y() - corners[i].first.y()<<std::endl;
    }
    gap_step = sum_width / (corners.size() - exception_count);
    step_width = sum_height / (corners.size() - exception_count);
    // std::cout<< corners.size()<<std::endl;
    // std::cout<<sum_width<<std::endl;
    // std::cout<<sum_height<<std::endl;
    LOG_INFO("gap step: {} step width: {}", gap_step, step_width);
}

void GapStepDetection::calculate_gap_step_dll_plot(lineSegments& corners,
                                          double& gap_step,
                                          double& step_width, 
                                          std::vector<std::vector<double>>& temp_res) {
    //temp_res.resize(2);
    double sum_width = 0.0;
    double sum_height = 0.0;
    int exception_count = 0;
    for (int i = 0; i < corners.size(); i++) {
        double temp_x = abs(corners[i].second.x() - corners[i].first.x());
        double temp_y = abs(corners[i].second.y() - corners[i].first.y());
        // exception
        if (temp_x > 100 || temp_y > 100) {
            exception_count++;
            continue;
        }
        temp_res[0].emplace_back(temp_x);
        temp_res[1].emplace_back(temp_y);
        sum_width += temp_x;
        sum_height += temp_y;
        // std::cout << i <<" " << corners[i].second.x() - corners[i].first.x()
        // << " "<<corners[i].second.y() - corners[i].first.y()<<std::endl;
    }
    //LOG_INFO("exception count: {}, valid corners: {}", exception_count,
    //         corners.size() - exception_count);
    gap_step = sum_height / (corners.size() - exception_count);
    step_width = sum_width / (corners.size() - exception_count);
    // std::cout<< corners.size()<<std::endl;
    // std::cout<<sum_width<<std::endl;
    // std::cout<<sum_height<<std::endl;
    LOG_INFO("gap step: {} step width: {}", gap_step, step_width);
}

}  // namespace pipeline
}  // namespace hymson3d
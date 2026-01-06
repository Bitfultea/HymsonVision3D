#include "DiskLevelMeasurement.h"

#include "3D/Mesh.h"
#include "Cluster.h"
#include "Curvature.h"
#include "Distance.h"
#include "Feature.h"
#include "FileSystem.h"
#include "FileTool.h"
#include "Filter.h"
#include "Logger.h"
#include "MathTool.h"
#include "Normal.h"
#include "Raster.h"

namespace hymson3d {
namespace pipeline {
std::vector<float> statistical_filter(const std::vector<float>& data,
                                      const float& n) {
    if (data.empty()) return {};
    int size = data.size();
    // 1. 计算均值
    float mean = std::accumulate(data.begin(), data.end(), 0.0) / size;

    // 2. 计算标准差
    float sum_squared_diff = 0.0;
    for (const auto& val : data) {
        sum_squared_diff += (val - mean) * (val - mean);
    }
    float std_dev = std::sqrt(sum_squared_diff / data.size());

    // 3. 过滤超出 mean ± n*σ 的值
    std::vector<float> filtered;

    for (int i = 0; i < size; ++i) {
        if (std::abs(data[i] - mean) <= n * std_dev) {
            filtered.push_back(data[i]);
        }
    }

    return filtered;
}
void fixCorner(cv::Mat& img,
               const cv::Mat& abnormalMask,
               int y,
               int x,
               int ks,
               float delta) {
    // 如果角点不是异常，直接返回
    if (abnormalMask.at<uchar>(y, x) == 0) return;
    // float sum = 0.f;
    // int cnt = 0;
    int step = 0;
    if ((ks - 1) % 2 != 0 || (ks <= 1)) {
        LOG_ERROR("Illegal input ks parameters.");
        return;
    } else {
        step = (ks - 1) / 2;
    }
    const int rows = img.rows;
    const int cols = img.cols;

    // ===== 根据角点位置裁剪邻域范围 =====
    const int y0 = std::max(0, y - step);
    const int y1 = std::min(rows - 1, y + step);
    const int x0 = std::max(0, x - step);
    const int x1 = std::min(cols - 1, x + step);
    // ===== 指针访问 + 无边界判断内层循环 =====
    //增加均值和标准差滤波
    std::vector<float> temp;
    for (int yy = y0; yy <= y1; ++yy) {
        const float* imgPtr = img.ptr<float>(yy);
        const uchar* maskPtr = abnormalMask.ptr<uchar>(yy);

        for (int xx = x0; xx <= x1; ++xx) {
            if (maskPtr[xx] == 0) {
                temp.emplace_back(imgPtr[xx]);
                // sum += imgPtr[xx];
                //++cnt;
            }
        }
    }
    // float delta = 1.5;
    std::vector<float> filtered = statistical_filter(temp, delta);
    if (filtered.size() > 0) {
        float filtered_mean =
                std::accumulate(filtered.begin(), filtered.end(), 0.0) /
                filtered.size();
        img.at<float>(y, x) = filtered_mean;
    } else {
        LOG_ERROR("No available data for use.");
    }
}

    bool DiskLevelMeasurement::preprocess_img(cv::Mat& tiff_image,
    int ks, float delta) {
        try {
            //异常角点检测
            double minVal, maxVal;
            cv::minMaxLoc(tiff_image, &minVal, &maxVal);
            if ((maxVal - minVal) > 500) {
                // std::cout << "minVal:" << minVal << "maxVal:" << maxVal <<
                // std::endl;
                cv::Mat mask1 =
                        ((tiff_image - minVal) >=
                         0.5 * (maxVal - minVal));  //异常值在大部分数值下方
                cv::Mat mask2 =
                        ((tiff_image - minVal) < 0.5 * (maxVal - minVal));
                int count1 = cv::countNonZero(mask1);
                int count2 = cv::countNonZero(mask2);
                //int ks = 101;
                //float delta = 1.5;
                if (count1 >= count2) {
                    //判断角点是否是异常值
                    fixCorner(tiff_image, mask2, 0, 0, ks, delta);
                    fixCorner(tiff_image, mask2, 0, tiff_image.cols - 1, ks,
                              delta);
                    fixCorner(tiff_image, mask2, tiff_image.rows - 1, 0, ks,
                              delta);
                    fixCorner(tiff_image, mask2, tiff_image.rows - 1,
                              tiff_image.cols - 1, ks, delta);
                } else {
                    fixCorner(tiff_image, mask1, 0, 0, ks, delta);
                    fixCorner(tiff_image, mask1, 0, tiff_image.cols - 1, ks,
                              delta);
                    fixCorner(tiff_image, mask1, tiff_image.rows - 1, 0, ks,
                              delta);
                    fixCorner(tiff_image, mask1, tiff_image.rows - 1,
                              tiff_image.cols - 1, ks, delta);
                }
            }
            return true;
        } catch (...) {
            LOG_ERROR("Failed to perform preprocess_img.");
            return false;
        }


    }

bool DiskLevelMeasurement::perform_measurement(
        std::shared_ptr<geometry::PointCloud> cloud,
        geometry::KDTreeSearchParamRadius param,
        DiskLevelMeasurementResult* result,
        std::pair<bool, cv::Point2f> disk_centre,
        float central_plane_size,
        float normal_angle_threshold,
        float distance_threshold,
        int min_planar_points,
        int method,
        bool debug_mode) {
    bool status;
    if (method == 0) {
        status = measure_pindisk_heightlevel_auto(
                cloud, param, result, disk_centre, central_plane_size,
                normal_angle_threshold, distance_threshold, min_planar_points,
                debug_mode);
        if (!status) {
            LOG_ERROR("Failed to perform measurement");
            return false;
        }
    } else if (method == 1) {
        status = measure_pindisk_heightlevel_region(
                cloud, param, result, disk_centre, central_plane_size,
                normal_angle_threshold, distance_threshold, min_planar_points,
                debug_mode);
        if (!status) {
            LOG_ERROR("Failed to perform measurement");
            return false;
        }
    } else {
        LOG_ERROR("Invalid method");
        return false;
    }
    return true;
}

// slow but fully automatic
bool DiskLevelMeasurement::measure_pindisk_heightlevel_auto(
        std::shared_ptr<geometry::PointCloud> cloud,
        geometry::KDTreeSearchParamRadius param,
        DiskLevelMeasurementResult* result,
        std::pair<bool, cv::Point2f> disk_centre,
        float central_plane_size,
        float normal_angle_threshold,
        float distance_threshold,
        int min_planar_points,
        bool debug_mode) {
    // get planar points
    std::vector<geometry::Plane::Ptr> planes;
    segment_plane_instances(cloud, param, planes, normal_angle_threshold,
                            min_planar_points, debug_mode);
    if (debug_mode) {
        cloud->PaintUniformColor(Eigen::Vector3d(1, 1, 1));
        for (int i = 0; i < planes.size(); i++) {
            Eigen::Vector3d color = core::Cluster::GenerateRandomColor();
            for (int j = 0; j < planes[i]->inlier_idx_.size(); j++) {
                cloud->colors_[planes[i]->inlier_idx_[j]] = color;
            }
        }
        utility::write_ply("planar_cluster_0.ply", cloud,
                           utility::FileFormat::BINARY);
    }

    // merge planes points
    float closet_plane_distance = 20;
    merge_plane_instances(cloud, planes, closet_plane_distance);
    if (debug_mode) {
        cloud->PaintUniformColor(Eigen::Vector3d(1, 1, 1));
        for (int i = 0; i < planes.size(); i++) {
            Eigen::Vector3d color = core::Cluster::GenerateRandomColor();
            for (int j = 0; j < planes[i]->inlier_idx_.size(); j++) {
                cloud->colors_[planes[i]->inlier_idx_[j]] = color;
            }
        }
        utility::write_ply("planar_cluster_1.ply", cloud,
                           utility::FileFormat::BINARY);
    }

    // identify central plane and bottom plane
    bool detect_bottom_plane = true;
    std::pair<geometry::Plane::Ptr, geometry::Plane::Ptr> plane_pair;
    plane_pair = identify_plane_instances(cloud, planes, central_plane_size,
                                          detect_bottom_plane, debug_mode);
    if (debug_mode) {
        cloud->PaintUniformColor(Eigen::Vector3d(1, 1, 1));
        Eigen::Vector3d color_centre = Eigen::Vector3d(0, 1, 0);
        for (int i = 0; i < plane_pair.first->inlier_idx_.size(); i++) {
            cloud->colors_[plane_pair.first->inlier_idx_[i]] = color_centre;
        }
        Eigen::Vector3d color_bot = Eigen::Vector3d(1, 0, 0);
        for (int i = 0; i < plane_pair.second->inlier_idx_.size(); i++) {
            cloud->colors_[plane_pair.second->inlier_idx_[i]] = color_bot;
        }
        utility::write_ply("planar_cluster_2.ply", cloud,
                           utility::FileFormat::BINARY);
    }

    // compute the desire data of planar clusters
    *result = DiskLevelMeasurement::calculate_planes_figure(plane_pair);
    return true;
}

// need to
bool DiskLevelMeasurement::measure_pindisk_heightlevel_region(
        std::shared_ptr<geometry::PointCloud> cloud,
        geometry::KDTreeSearchParamRadius param,
        DiskLevelMeasurementResult* result,
        std::pair<bool, cv::Point2f> disk_centre,
        float central_plane_size,
        float normal_angle_threshold,
        float distance_threshold,
        int min_planar_points,
        bool debug_mode) {
    int row = cloud->height_;
    int col = cloud->width_;
    std::vector<Eigen::Vector3d> bottom_points;
    // get the corner points of the tiff
    for (int i = 0; i < 4; i++) {
        if (i == 0) bottom_points.emplace_back(cloud->points_[0]);
        if (i == 1) bottom_points.emplace_back(cloud->points_[col - 1]);
        if (i == 2) bottom_points.emplace_back(cloud->points_[col * (row - 1)]);
        if (i == 3) bottom_points.emplace_back(cloud->points_[(col * row) - 1]);
    }
    // for (auto pt : bottom_points) {
    //     std::cout << pt << std::endl;
    // }
    geometry::PointCloud::Ptr bot_cloud =
            std::make_shared<geometry::PointCloud>();
    bot_cloud->points_ = bottom_points;
    core::PlaneDetection pd;
    // geometry::Plane::Ptr bot_plane =
    //         pd.fit_a_plane_ransc(*bot_cloud, 2, 4, 10, 0.9999999);
    geometry::Plane::Ptr bot_plane = pd.fit_a_plane(*bot_cloud);
    if (debug_mode) {
        LOG_DEBUG(
                "Bottom plane info: \n Center: {}, {}, {}; \n Normal: {}, {}, "
                "{};",
                bot_plane->center_.x(), bot_plane->center_.y(),
                bot_plane->center_.z(), bot_plane->normal_.x(),
                bot_plane->normal_.y(), bot_plane->normal_.z());
        utility::write_plane_mesh_ply(
                "bottom_plane.ply", *bot_plane, bot_cloud->GetMinBound().x(),
                bot_cloud->GetMaxBound().x(), bot_cloud->GetMinBound().y(),
                bot_cloud->GetMaxBound().y(), 100);
    }

    bool use_ransc = true;
    geometry::Plane::Ptr central_plane =
            get_plane_in_range(cloud, param, disk_centre, central_plane_size,
                               normal_angle_threshold, distance_threshold,
                               min_planar_points, use_ransc, debug_mode);
    if (central_plane == nullptr) {
        LOG_ERROR("Cannot detect central plane.");
        return false;
    }

    if (debug_mode) {
        utility::write_plane_mesh_ply(
                "central_plane.ply", *central_plane,
                central_plane->center_.x() - central_plane_size / 2,
                central_plane->center_.x() + central_plane_size / 2,
                central_plane->center_.y() - central_plane_size / 2,
                central_plane->center_.y() + central_plane_size / 2, 100);
    }

    std::pair<geometry::Plane::Ptr, geometry::Plane::Ptr> plane_pair;
    plane_pair.first = central_plane;
    plane_pair.second = bot_plane;
    *result = DiskLevelMeasurement::calculate_planes_figure(plane_pair);
    return true;
}

bool DiskLevelMeasurement::measure_pindisk_heightlevel_region_dll(
        std::shared_ptr<geometry::PointCloud> bottom_cloud,
        std::shared_ptr<geometry::PointCloud> central_cloud,
        DiskLevelMeasurementResult* result,
        geometry::KDTreeSearchParamRadius param,
        float normal_angle_threshold,
        float distance_threshold,
        int min_planar_points,
        bool debug_mode) {
    core::PlaneDetection pd;
    // geometry::Plane::Ptr bot_plane =
    //         pd.fit_a_plane_ransc(*bot_cloud, 2, 4, 10, 0.9999999);
    geometry::Plane::Ptr bot_plane = pd.fit_a_plane(*bottom_cloud);
    if (debug_mode) {
        LOG_DEBUG(
                "Bottom plane info: \n Center: {}, {}, {}; \n Normal: {}, {}, "
                "{};",
                bot_plane->center_.x(), bot_plane->center_.y(),
                bot_plane->center_.z(), bot_plane->normal_.x(),
                bot_plane->normal_.y(), bot_plane->normal_.z());
        utility::write_plane_mesh_ply("bottom_plane.ply", *bot_plane,
                                      bottom_cloud->GetMinBound().x(),
                                      bottom_cloud->GetMaxBound().x(),
                                      bottom_cloud->GetMinBound().y(),
                                      bottom_cloud->GetMaxBound().y(), 100);
    }

    bool use_ransc = true;
    geometry::Plane::Ptr central_plane = get_plane_in_range_all(
            central_cloud, param, normal_angle_threshold, distance_threshold,
            min_planar_points, use_ransc, debug_mode);
    if (central_plane == nullptr) {
        LOG_ERROR("Cannot detect central plane.");
        return false;
    }

    if (debug_mode) {
        LOG_DEBUG(
                "Central plane info: \n Center: {}, {}, {}; \n Normal: {}, {}, "
                "{};",
                central_plane->center_.x(), central_plane->center_.y(),
                central_plane->center_.z(), central_plane->normal_.x(),
                central_plane->normal_.y(), central_plane->normal_.z());
        utility::write_plane_mesh_ply("central_plane.ply", *central_plane,
                                      central_cloud->GetMinBound().x(),
                                      central_cloud->GetMaxBound().x(),
                                      central_cloud->GetMinBound().y(),
                                      central_cloud->GetMaxBound().y(), 100);
    }

    std::pair<geometry::Plane::Ptr, geometry::Plane::Ptr> plane_pair;
    plane_pair.first = central_plane;
    plane_pair.second = bot_plane;
    *result = DiskLevelMeasurement::calculate_planes_figure(plane_pair);
    return true;
}

void DiskLevelMeasurement::measure_pindisk_heightlevel_region_dllv2(
        std::vector<Eigen::Vector3d> bottom_points,
        std::shared_ptr<geometry::PointCloud> central_cloud,
        DiskLevelMeasurementResult* result,
        geometry::KDTreeSearchParamRadius param,
        float normal_angle_threshold,
        float distance_threshold,
        int min_planar_points,
        bool debug_mode) {
    geometry::PointCloud::Ptr bottom_cloud =
            std::make_shared<geometry::PointCloud>();
    bottom_cloud->points_ = bottom_points;
    core::PlaneDetection pd;
    // geometry::Plane::Ptr bot_plane =
    //         pd.fit_a_plane_ransc(*bot_cloud, 2, 4, 10, 0.9999999);
    geometry::Plane::Ptr bot_plane = pd.fit_a_plane(*bottom_cloud);
    if (debug_mode) {
        LOG_DEBUG(
                "Bottom plane info: \n Center: {}, {}, {}; \n Normal: {}, {}, "
                "{};",
                bot_plane->center_.x(), bot_plane->center_.y(),
                bot_plane->center_.z(), bot_plane->normal_.x(),
                bot_plane->normal_.y(), bot_plane->normal_.z());
        utility::write_plane_mesh_ply("bottom_plane.ply", *bot_plane,
                                      bottom_cloud->GetMinBound().x(),
                                      bottom_cloud->GetMaxBound().x(),
                                      bottom_cloud->GetMinBound().y(),
                                      bottom_cloud->GetMaxBound().y(), 100);
    }

    bool use_ransc = true;
    geometry::Plane::Ptr central_plane = get_plane_in_range_all(
            central_cloud, param, normal_angle_threshold, distance_threshold,
            min_planar_points, use_ransc, debug_mode);

    if (debug_mode) {
        LOG_DEBUG(
                "Central plane info: \n Center: {}, {}, {}; \n Normal: {}, {}, "
                "{};",
                central_plane->center_.x(), central_plane->center_.y(),
                central_plane->center_.z(), central_plane->normal_.x(),
                central_plane->normal_.y(), central_plane->normal_.z());
        utility::write_plane_mesh_ply("central_plane.ply", *central_plane,
                                      central_cloud->GetMinBound().x(),
                                      central_cloud->GetMaxBound().x(),
                                      central_cloud->GetMinBound().y(),
                                      central_cloud->GetMaxBound().y(), 100);
    }

    std::pair<geometry::Plane::Ptr, geometry::Plane::Ptr> plane_pair;
    plane_pair.first = central_plane;
    plane_pair.second = bot_plane;
    *result = DiskLevelMeasurement::calculate_planes_figure(plane_pair);
}
void DiskLevelMeasurement::measure_pindisk_heightlevel(
        std::shared_ptr<geometry::PointCloud> bottom_cloud,
        std::shared_ptr<geometry::PointCloud> central_cloud,
        DiskLevelMeasurementResult* result,
        bool debug_mode) {
    core::PlaneDetection pd;
    // geometry::Plane::Ptr bot_plane =
    //         pd.fit_a_plane_ransc(*bot_cloud, 2, 4, 10, 0.9999999);
    geometry::Plane::Ptr bot_plane = pd.fit_a_plane(*bottom_cloud);
    if (debug_mode) {
        LOG_DEBUG(
                "Bottom plane info: \n Center: {}, {}, {}; \n Normal: {}, {}, "
                "{};",
                bot_plane->center_.x(), bot_plane->center_.y(),
                bot_plane->center_.z(), bot_plane->normal_.x(),
                bot_plane->normal_.y(), bot_plane->normal_.z());
        utility::write_plane_mesh_ply("bottom_plane.ply", *bot_plane,
                                      bottom_cloud->GetMinBound().x(),
                                      bottom_cloud->GetMaxBound().x(),
                                      bottom_cloud->GetMinBound().y(),
                                      bottom_cloud->GetMaxBound().y(), 100);
    }
    geometry::Plane::Ptr central_plane = pd.fit_a_plane(*central_cloud);
    if (debug_mode) {
        LOG_DEBUG(
                "Central plane info: \n Center: {}, {}, {}; \n Normal: {}, {}, "
                "{};",
                central_plane->center_.x(), central_plane->center_.y(),
                central_plane->center_.z(), central_plane->normal_.x(),
                central_plane->normal_.y(), central_plane->normal_.z());
        utility::write_plane_mesh_ply("central_plane.ply", *central_plane,
                                      central_cloud->GetMinBound().x(),
                                      central_cloud->GetMaxBound().x(),
                                      central_cloud->GetMinBound().y(),
                                      central_cloud->GetMaxBound().y(), 100);
    }

    std::pair<geometry::Plane::Ptr, geometry::Plane::Ptr> plane_pair;
    plane_pair.first = central_plane;
    plane_pair.second = bot_plane;
    *result = DiskLevelMeasurement::calculate_planes_figure(plane_pair);
}

void DiskLevelMeasurement::segment_plane_instances(
        std::shared_ptr<geometry::PointCloud> cloud,
        geometry::KDTreeSearchParamRadius param,
        std::vector<geometry::Plane::Ptr>& planes,
        float normal_angle_threshold,
        int min_planar_points,
        bool debug_mode) {
    // preform plane instance segmentation
    float radius = param.radius_;
    float curvature_threshold = 0.0f;
    int planar_proposal_size = 100;
    int num_plane_candicates = core::Cluster::PlanarCluster(
            *cloud, radius, normal_angle_threshold, curvature_threshold, false,
            planar_proposal_size, debug_mode);

    // extract plane
    for (int i = 0; i < num_plane_candicates; i++) {
        geometry::Plane::Ptr plane = std::make_shared<geometry::Plane>();
        planes.emplace_back(plane);
    }
    for (int i = 0; i < cloud->points_.size(); i++) {
        if (cloud->labels_[i] >= 0) {
            planes[cloud->labels_[i]]->inlier_points_.emplace_back(
                    cloud->points_[i]);
            planes[cloud->labels_[i]]->inlier_idx_.emplace_back(i);
        }
    }
    // std::vector<geometry::PointCloud> planes_cloud;
    // for (int i = 0; i < num_plane_candicates; i++) {
    //     geometry::PointCloud::Ptr pcd =
    //             std::make_shared<geometry::PointCloud>();
    //     pcd->points_ = planes[i]->inlier_points_;
    //     planes_cloud.emplace_back(*pcd);
    // }

    // get rid off small planes
    std::vector<geometry::Plane::Ptr> filtered_planes;
    for (int i = 0; i < planes.size(); i++) {
        if (planes[i]->inlier_points_.size() > min_planar_points) {
            filtered_planes.emplace_back(planes[i]);
        } else {
            LOG_DEBUG("Plane {} is too small({} points). Removed.", i,
                      planes[i]->inlier_points_.size());
        }
    }
    planes = filtered_planes;
}

void DiskLevelMeasurement::merge_plane_instances(
        std::shared_ptr<geometry::PointCloud> cloud,
        std::vector<geometry::Plane::Ptr>& planes,
        float plane_distance_threshold) {
    std::set<int> merged_idx;
    std::vector<geometry::Plane::Ptr> mergered_planes;
    if (planes.size() == 1) {
        LOG_DEBUG("Only one plane detected. No need to merge.");
        return;
    }
    for (int i = 0; i < planes.size(); i++) {
        if (merged_idx.count(i)) {  // alreadt merged
            continue;
        }
        for (int j = i + 1; j < planes.size(); j++) {
            // calcaulte the distance between two planes/clouds
            if (hymson3d::core::feature::cloud_to_cloud_distance(
                        planes[i]->inlier_points_, planes[j]->inlier_points_) <
                plane_distance_threshold) {
                planes[i]->inlier_points_.insert(
                        planes[i]->inlier_points_.end(),
                        planes[j]->inlier_points_.begin(),
                        planes[j]->inlier_points_.end());
                planes[i]->inlier_idx_.insert(planes[i]->inlier_idx_.end(),
                                              planes[j]->inlier_idx_.begin(),
                                              planes[j]->inlier_idx_.end());
                merged_idx.insert(j);
            }
        }
        mergered_planes.emplace_back(planes[i]);
    }
    planes = mergered_planes;
}

std::pair<geometry::Plane::Ptr, geometry::Plane::Ptr>
DiskLevelMeasurement::identify_plane_instances(
        std::shared_ptr<geometry::PointCloud> cloud,
        std::vector<geometry::Plane::Ptr>& planes,
        float central_plane_size,
        bool detect_bottom_plane,
        bool debug_mode) {
    Eigen::Vector3d centre_pt = cloud->GetCenter();
    geometry::Plane::Ptr bottom_plane = std::make_shared<geometry::Plane>();
    bottom_plane = nullptr;
    geometry::Plane::Ptr central_plane = std::make_shared<geometry::Plane>();
    central_plane = nullptr;
    if (!detect_bottom_plane) {
        geometry::Plane::Ptr bottom_plane = std::make_shared<geometry::Plane>();
        bottom_plane->coeff_ = Eigen::Vector4d(0, 0, 1, 0);
        double closest_distance = std::numeric_limits<double>::max();
        for (int i = 0; i < planes.size(); i++) {
            std::shared_ptr<geometry::PointCloud> plane_cloud =
                    std::make_shared<geometry::PointCloud>();
            plane_cloud->points_ = planes[i]->inlier_points_;
            Eigen::Vector3d plane_center = plane_cloud->GetCenter();
            float distance = (plane_center - centre_pt).norm();
            if (distance < closest_distance) {
                closest_distance = distance;
                central_plane = planes[i];
            }
        }
    } else {
        int max_pts = 0;

        for (int i = 0; i < planes.size(); i++) {
            std::shared_ptr<geometry::PointCloud> plane_cloud =
                    std::make_shared<geometry::PointCloud>();
            plane_cloud->points_ = planes[i]->inlier_points_;
            Eigen::Vector3d plane_center = plane_cloud->GetCenter();
            float distance = sqrt(pow((plane_center.x() - centre_pt.x()), 2) +
                                  pow((plane_center.x() - centre_pt.x()), 2));
            Eigen::Vector3d dims = plane_cloud->GetExtend();
            bool size_ok = false;

            if (dims.x() > 0.8 * central_plane_size &&
                dims.x() < 1.2 * central_plane_size) {
                if (dims.y() > 0.8 * central_plane_size &&
                    dims.y() < 1.2 * central_plane_size) {
                    size_ok = true;
                }
            }
            // double ratio = dims.x() / dims.y();
            std::cout << size_ok << " " << distance << std::endl;
            if ((distance < central_plane_size / 2) && size_ok) {
                central_plane = planes[i];
                continue;
            }
            if (planes[i]->inlier_points_.size() > max_pts) {
                bottom_plane = planes[i];
                max_pts = planes[i]->inlier_points_.size();
            }
        }
    }
    if (central_plane == nullptr) {
        LOG_ERROR("No central plane found");
    }
    if (bottom_plane == nullptr) {
        LOG_ERROR("No bottom plane found");
    }
    return std::make_pair(central_plane, bottom_plane);
}

DiskLevelMeasurementResult DiskLevelMeasurement::calculate_planes_figure(
        std::pair<geometry::Plane::Ptr, geometry::Plane::Ptr> input_planes) {
    core::PlaneDetection plane_detector;
    DiskLevelMeasurementResult result;
    geometry::Plane::Ptr central_plane;
    geometry::Plane::Ptr bottom_plane;
    if (input_planes.first->normal_.norm() == 0.0) {
        central_plane =
                plane_detector.fit_a_plane(input_planes.first->inlier_points_);
    } else {
        central_plane = input_planes.first;
    }
    if (input_planes.second->normal_.norm() == 0.0) {
        bottom_plane =
                plane_detector.fit_a_plane(input_planes.second->inlier_points_);
    } else {
        bottom_plane = input_planes.second;
    }
    // std::cout << "central_plane normal: " <<
    // central_plane->normal_.transpose()
    //           << std::endl;
    // std::cout << "bottom_plane normal: " <<
    // bottom_plane->normal_.transpose()
    //           << std::endl;
    central_plane->orient_normals_towards_positive_z();
    bottom_plane->orient_normals_towards_positive_z();
    double angle =
            (central_plane->normal_.dot(bottom_plane->normal_)) /
            (central_plane->normal_.norm() * bottom_plane->normal_.norm());
    result.plane_angle = std::abs(std::acos(angle) * (180.0 / M_PI));
    result.plane_height_gap = core::feature::point_to_plane_distance(
            central_plane->center_, *bottom_plane);
    return result;
}

geometry::Plane::Ptr DiskLevelMeasurement::get_plane_in_range(
        geometry::PointCloud::Ptr cloud,
        geometry::KDTreeSearchParamRadius param,
        std::pair<bool, cv::Point2f> disk_centre,
        float central_plane_size,
        float normal_angle_threshold,
        float distance_threshold,
        int min_planar_points,
        bool use_ransac,
        bool debug_mode) {
    Eigen::Vector3d centroid;
    if (disk_centre.first) {
        centroid =
                Eigen::Vector3d(disk_centre.second.x, disk_centre.second.y, 0);
    } else {
        centroid = cloud->GetCenter();
    }
    double x_lower_bound = centroid[0] - central_plane_size / 2.0;
    double x_upper_bound = centroid[0] + central_plane_size / 2.0;
    double y_lower_bound = centroid[1] - central_plane_size / 2.0;
    double y_upper_bound = centroid[1] + central_plane_size / 2.0;
    std::vector<Eigen::Vector3d> points;
    for (size_t i = 0; i < cloud->points_.size(); i++) {
        Eigen::Vector3d point = cloud->points_[i];
        if (point[0] >= x_lower_bound && point[0] <= x_upper_bound &&
            point[1] >= y_lower_bound && point[1] <= y_upper_bound) {
            points.emplace_back(point);
        }
    }
    LOG_DEBUG("Inner rangepoints size: {}", points.size());
    std::shared_ptr<geometry::PointCloud> region_cloud =
            std::make_shared<geometry::PointCloud>();
    region_cloud->points_ = points;

    std::vector<geometry::Plane::Ptr> planes;
    segment_plane_instances(region_cloud, param, planes, normal_angle_threshold,
                            min_planar_points, debug_mode);
    float closet_plane_distance = 20;
    merge_plane_instances(region_cloud, planes, closet_plane_distance);

    LOG_DEBUG("Plane instances size: {}", planes.size());
    if (planes.size() == 0) {
        LOG_ERROR("No plane found in the region");
        return nullptr;
    }

    geometry ::Plane::Ptr max_plane;
    int max_pts = 0;
    for (auto p : planes) {
        if (p->inlier_points_.size() > max_pts) {
            max_plane = p;
            max_pts = p->inlier_points_.size();
        }
    }

    if (debug_mode) {
        utility::write_ply("centroid_pts.ply", max_plane->inlier_points_);
    }

    core::PlaneDetection plane_detector;
    std::shared_ptr<geometry::Plane> output_plane;
    if (!use_ransac) {
        output_plane = plane_detector.fit_a_plane(max_plane->inlier_points_);
    } else {
        std::shared_ptr<geometry::PointCloud> central_cloud =
                std::make_shared<geometry::PointCloud>();
        central_cloud->points_ = max_plane->inlier_points_;
        output_plane = plane_detector.fit_a_plane_ransc(*central_cloud, 2, 4,
                                                        100, 0.9999999);
    }

    return output_plane;
}
geometry::Plane::Ptr DiskLevelMeasurement::get_plane_in_range_all(
        std::shared_ptr<geometry::PointCloud> region_cloud,
        geometry::KDTreeSearchParamRadius param,
        float normal_angle_threshold,
        float distance_threshold,
        int min_planar_points,
        bool use_ransac,
        bool debug_mode) {
    std::vector<geometry::Plane::Ptr> planes;
    segment_plane_instances(region_cloud, param, planes, normal_angle_threshold,
                            min_planar_points, debug_mode);
    float closet_plane_distance = 20;
    merge_plane_instances(region_cloud, planes, closet_plane_distance);

    LOG_DEBUG("Plane instances size: {}", planes.size());
    if (planes.size() == 0) {
        LOG_ERROR("No plane found in the region");
        return nullptr;
    }

    geometry ::Plane::Ptr max_plane;
    int max_pts = 0;
    for (auto p : planes) {
        if (p->inlier_points_.size() > max_pts) {
            max_plane = p;
            max_pts = p->inlier_points_.size();
        }
    }

    if (debug_mode) {
        utility::write_ply("centroid_pts.ply", max_plane->inlier_points_);
    }

    core::PlaneDetection plane_detector;
    std::shared_ptr<geometry::Plane> output_plane;
    if (!use_ransac) {
        output_plane = plane_detector.fit_a_plane(max_plane->inlier_points_);
    } else {
        std::shared_ptr<geometry::PointCloud> central_cloud =
                std::make_shared<geometry::PointCloud>();
        central_cloud->points_ = max_plane->inlier_points_;
        output_plane = plane_detector.fit_a_plane_ransc(*central_cloud, 2, 4,
                                                        100, 0.9999999);
    }

    return output_plane;
}
}  // namespace pipeline
}  // namespace hymson3d
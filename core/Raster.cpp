#include "Raster.h"
namespace hymson3d {
namespace core {

void PointCloudRaster::set_frame_size(int frame_width, int frame_height) {
    frame_width_ = frame_width;
    frame_height_ = frame_height;
}

/// @brief 计算正交投影矩阵的内参
/// K = | fx 0 cx |
///     | 0 fy cy |
///     | f 0 1  |
void PointCloudRaster::cal_ortho_proj_intrinsics(
        geometry::PointCloud::Ptr pointcloud) {
    Eigen::Vector3d min_bound = pointcloud->GetMinBound();
    Eigen::Vector3d max_bound = pointcloud->GetMaxBound();
    Eigen::Vector3d extent = max_bound - min_bound;
    Eigen::Vector3d center = (max_bound + min_bound) / 2.0;

    // fx,fy as the scale ratio of a pxiel in x,y direction respectively
    double dist_per_piexel_x = extent.x() / frame_width_;
    double dist_per_piexel_y = extent.y() / frame_height_;
    LOG_INFO("Project resolution: {} x {}", dist_per_piexel_x,
             dist_per_piexel_y);

    double fx = 1.0 / dist_per_piexel_x;
    double fy = 1.0 / dist_per_piexel_y;
    // cx,cy ceneter the frame coord as the camera
    double cx = frame_width_ / 2.0;
    double cy = frame_height_ / 2.0;

    Eigen::Matrix3d K;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    camera_->set_intrinsics(K);
}

cv::Mat PointCloudRaster::project_to_frame(geometry::PointCloud::Ptr pointcloud,
                                           bool use_z_buffer) {
    cal_ortho_proj_intrinsics(pointcloud);  // set ortho projection intrinsics

    cv::Mat depthImage = cv::Mat::zeros(frame_height_, frame_width_, CV_32FC1);

    if (!use_z_buffer) {
#pragma omp parallel for
        // no z buffer normally for z upper rthographic projection
        for (int i = 0; i < pointcloud->points_.size(); ++i) {
            // use homogeneous coordinate
            Eigen::Vector4d point3D(pointcloud->points_[i].x(),
                                    pointcloud->points_[i].y(),
                                    pointcloud->points_[i].z(), 1.0);
            // world to camera coordinate
            Eigen::Vector4d pointCamera = camera_->get_extrinsic() * point3D;
            // camera to frame plane
            Eigen::Vector3d pointFrame =
                    camera_->get_intrinsics() * pointCamera.head<3>();
            // pixel position
            int u = static_cast<int>(pointFrame.x());
            int v = static_cast<int>(pointFrame.y());
            // outside of frame
            if (u < 0 || u >= frame_width_ || v < 0 || v >= frame_height_)
                continue;
            // use point z-value as depth. ignore multiple points in one pixel
            depthImage.at<float>(v, u) = pointcloud->points_[i].z();
        }
    } else {
#pragma omp parallel for
        // use z buffer but still only consider orthographic projection
        for (int i = 0; i < pointcloud->points_.size(); ++i) {
            Eigen::Vector4d point3D(pointcloud->points_[i].x(),
                                    pointcloud->points_[i].y(),
                                    pointcloud->points_[i].z(), 1.0);
            Eigen::Vector4d pointCamera = camera_->get_extrinsic() * point3D;
            Eigen::Vector3d pointFrame =
                    camera_->get_intrinsics() * pointCamera.head<3>();
            float depth = static_cast<float>(pointCamera.z());
            int u = std::round(pointFrame.x());
            int v = std::round(pointFrame.y());
            if (u < 0 || u >= frame_width_ || v < 0 || v >= frame_height_)
                continue;
            // update with the closest point to camera
            if (depthImage.at<float>(v, u) == 0 ||
                depth < depthImage.at<float>(v, u)) {
                depthImage.at<float>(v, u) = depth;
            }
        }
    }

    return depthImage;
}

cv::Mat PointCloudRaster::project_to_frame(geometry::PointCloud::Ptr pointcloud,
                                           Mode mode) {
    cal_ortho_proj_intrinsics(pointcloud);  // set ortho projection intrinsics

    cv::Mat depthImage = cv::Mat::zeros(frame_height_, frame_width_, CV_32FC1);
    cv::Mat hitcount = cv::Mat::zeros(frame_height_, frame_width_, CV_8UC1);

    switch (mode) {
        case Mode::AVE:
#pragma omp parallel for
            // use z buffer but still only consider orthographic projection
            for (int i = 0; i < pointcloud->points_.size(); ++i) {
                Eigen::Vector4d point3D(pointcloud->points_[i].x(),
                                        pointcloud->points_[i].y(),
                                        pointcloud->points_[i].z(), 1.0);
                Eigen::Vector4d pointCamera =
                        camera_->get_extrinsic() * point3D;
                Eigen::Vector3d pointFrame =
                        camera_->get_intrinsics() * pointCamera.head<3>();
                float depth = static_cast<float>(pointCamera.z());
                int u = static_cast<int>(pointFrame.x());
                int v = static_cast<int>(pointFrame.y());
                if (u < 0 || u >= frame_width_ || v < 0 || v >= frame_height_)
                    continue;
                depthImage.at<float>(v, u) += depth;
                hitcount.at<unsigned char>(v, u)++;
            }
            depthImage = depthImage / hitcount;
            break;
        case Mode::FAR:
#pragma omp parallel for
            // use z buffer but still only consider orthographic projection
            for (int i = 0; i < pointcloud->points_.size(); ++i) {
                Eigen::Vector4d point3D(pointcloud->points_[i].x(),
                                        pointcloud->points_[i].y(),
                                        pointcloud->points_[i].z(), 1.0);
                Eigen::Vector4d pointCamera =
                        camera_->get_extrinsic() * point3D;
                Eigen::Vector3d pointFrame =
                        camera_->get_intrinsics() * pointCamera.head<3>();
                float depth = static_cast<float>(pointCamera.z());
                int u = std::round(pointFrame.x());
                int v = std::round(pointFrame.y());
                if (u < 0 || u >= frame_width_ || v < 0 || v >= frame_height_)
                    continue;
                if (depthImage.at<float>(v, u) == 0 ||
                    depth > depthImage.at<float>(v, u)) {
                    depthImage.at<float>(v, u) = depth;
                }
            }
            break;
        case Mode::NEAREST:
#pragma omp parallel for
            // use z buffer but still only consider orthographic projection
            for (int i = 0; i < pointcloud->points_.size(); ++i) {
                Eigen::Vector4d point3D(pointcloud->points_[i].x(),
                                        pointcloud->points_[i].y(),
                                        pointcloud->points_[i].z(), 1.0);
                Eigen::Vector4d pointCamera =
                        camera_->get_extrinsic() * point3D;
                Eigen::Vector3d pointFrame =
                        camera_->get_intrinsics() * pointCamera.head<3>();
                float depth = static_cast<float>(pointCamera.z());
                int u = std::round(pointFrame.x());
                int v = std::round(pointFrame.y());
                if (u < 0 || u >= frame_width_ || v < 0 || v >= frame_height_)
                    continue;
                if (depthImage.at<float>(v, u) == 0 ||
                    depth < depthImage.at<float>(v, u)) {
                    depthImage.at<float>(v, u) = depth;
                }
            }
            break;
        case Mode::SUM:
#pragma omp parallel for
            // use z buffer but still only consider orthographic projection
            for (int i = 0; i < pointcloud->points_.size(); ++i) {
                Eigen::Vector4d point3D(pointcloud->points_[i].x(),
                                        pointcloud->points_[i].y(),
                                        pointcloud->points_[i].z(), 1.0);
                Eigen::Vector4d pointCamera =
                        camera_->get_extrinsic() * point3D;
                Eigen::Vector3d pointFrame =
                        camera_->get_intrinsics() * pointCamera.head<3>();
                float depth = static_cast<float>(pointCamera.z());
                int u = std::round(pointFrame.x());
                int v = std::round(pointFrame.y());
                if (u < 0 || u >= frame_width_ || v < 0 || v >= frame_height_)
                    continue;
                depthImage.at<float>(v, u) += depth;
            }
            break;
    }

    return depthImage;
}
cv::Mat PointCloudRaster::robust_normalize(const cv::Mat& src,
                                           double lower_percent,
                                           double upper_percent) {
    cv::Mat flat;
    src.reshape(1, 1).convertTo(flat, CV_32F);
    std::vector<float> pixels = flat;

    std::sort(pixels.begin(), pixels.end());
    float lower = pixels[static_cast<int>(pixels.size() * lower_percent)];
    float upper = pixels[static_cast<int>(pixels.size() * upper_percent)];

    cv::Mat clipped;
    cv::threshold(src, clipped, upper, upper, cv::THRESH_TRUNC);
    cv::max(clipped, lower, clipped);

    cv::Mat norm_img;
    if (upper - lower > 1e-6) {
        clipped.convertTo(norm_img, CV_8U, 255.0 / (upper - lower),
                          -lower * 255.0 / (upper - lower));
    } else {
        norm_img = cv::Mat::zeros(src.size(), CV_8U);
    }
    return norm_img;
}

cv::Mat PointCloudRaster::project_to_feature_frame(const cv::Mat& tiff_image) {
    cv::Mat raw_f;
    tiff_image.convertTo(raw_f, CV_32F);

    // Ch1: Height
    cv::Mat ch_height = robust_normalize(raw_f);

    // Ch2: Gradient
    cv::Mat gx, gy, grad;
    cv::Sobel(raw_f, gx, CV_32F, 1, 0, 3);
    cv::Sobel(raw_f, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, grad);
    cv::Mat ch_grad = robust_normalize(grad, 0.0, 0.98);

    // Ch3: Texture (CLAHE)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat ch_texture;
    clahe->apply(ch_height, ch_texture);

    // Merge
    cv::Mat merged;
    std::vector<cv::Mat> channels = {ch_height, ch_grad, ch_texture};
    cv::merge(channels, merged);  // BGR for OpenCV

    return merged;
}

// set camera at z-positive direction above the center of pointcloud
cv::Mat PointCloudRaster::simple_projection(
        geometry::PointCloud::Ptr pointcloud) {
    cv::Mat depthImage = cv::Mat::zeros(frame_height_, frame_width_, CV_32FC1);
    Eigen::Vector3d min_bound = pointcloud->GetMinBound();
    Eigen::Vector3d max_bound = pointcloud->GetMaxBound();
    Eigen::Vector3d extent = max_bound - min_bound;

#pragma omp parallel for
    for (int i = 0; i < pointcloud->points_.size(); ++i) {
        Eigen::Vector3d pt = pointcloud->points_[i] - min_bound;
        int u = std::round(pt.x() / extent.x() * frame_width_);
        int v = std::round(pt.y() / extent.y() * frame_height_);
        if (u >= 0 && u < frame_width_ && v >= 0 && v < frame_height_) {
            depthImage.at<float>(v, u) = pt.z();
        }
    }
    return depthImage;
}

cv::Mat PointCloudRaster::simple_projection_x(
        geometry::PointCloud::Ptr pointcloud) {
    cv::Mat depthImage = cv::Mat::zeros(frame_height_, frame_width_, CV_32FC1);
    Eigen::Vector3d min_bound = pointcloud->GetMinBound();
    Eigen::Vector3d max_bound = pointcloud->GetMaxBound();
    Eigen::Vector3d extent = max_bound - min_bound;

#pragma omp parallel for
    for (int i = 0; i < pointcloud->points_.size(); ++i) {
        Eigen::Vector3d pt = pointcloud->points_[i] - min_bound;
        int u = std::round(pt.y() / extent.y() * frame_width_);
        int v = std::round(pt.z() / extent.z() * frame_height_);
        if (u >= 0 && u < frame_width_ && v >= 0 && v < frame_height_) {
            depthImage.at<float>(v, u) = pt.x();
        }
    }
    return depthImage;
}

}  // namespace core
}  // namespace hymson3d
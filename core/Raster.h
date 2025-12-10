#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "3D/PointCloud.h"
#include "Camera.h"
#include "fmtfallback.h"

namespace hymson3d {
namespace core {

class PointCloudRaster {
public:
    PointCloudRaster() = default;
    PointCloudRaster(int frame_width, int frame_height, Camera::Ptr camera)
        : frame_width_(frame_width),
          frame_height_(frame_height),
          camera_(camera) {}
    enum Mode { AVE = 0, FARR = 1, NEAREST = 2, SUM = 3 };

public:
    void set_camera(Camera::Ptr camera) { camera_ = camera; }
    void set_frame_size(int frame_width, int frame_height);
    // ignore z_buffer in camera_coord for orthographic projection
    cv::Mat project_to_frame(geometry::PointCloud::Ptr pointcloud,
                             bool use_z_buffer = false);
    cv::Mat project_to_frame(geometry::PointCloud::Ptr pointcloud,
                             Mode mode = Mode::AVE);
    cv::Mat project_to_feature_frame(const cv::Mat& tiff_image);
    cv::Mat simple_projection(geometry::PointCloud::Ptr pointcloud);
    cv::Mat simple_projection_x(geometry::PointCloud::Ptr pointcloud);

private:
    void cal_ortho_proj_intrinsics(geometry::PointCloud::Ptr pointcloud);
    cv::Mat robust_normalize(const cv::Mat& src,
                             double lower_percent = 0.005,
                             double upper_percent = 0.995);

public:
    int frame_width_;
    int frame_height_;
    Camera::Ptr camera_;
};

}  // namespace core
}  // namespace hymson3d
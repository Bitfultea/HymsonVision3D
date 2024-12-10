#include "CalibCamera.h"

namespace hymson3d {
namespace pipeline {

calibration_data* CalibCamera::calib(const std::string config_file,
                                     bool debug_mode) {
    core::CameraCalib::Ptr camera_calibration =
            std::make_shared<core::CameraCalib>(debug_mode);
    camera_calibration->init(config_file);
    camera_calibration->process();

    calibration_data* calib_data = new calibration_data();
    calib_data->camera_matrix = camera_calibration->get_camera_matrix();
    calib_data->dist_coeffs = camera_calibration->get_dist_coeffs();
    return calib_data;
}

cv::Mat CalibCamera::undisort(const std::string config_file,
                              cv::Mat& img,
                              bool debug_mode) {
    core::CameraCalib::Ptr camera_calibration =
            std::make_shared<core::CameraCalib>(debug_mode);
    cv::Mat undistorted_img;
    camera_calibration->undistort_image(config_file, img, undistorted_img);
    return undistorted_img;
}

}  // namespace pipeline
}  // namespace hymson3d
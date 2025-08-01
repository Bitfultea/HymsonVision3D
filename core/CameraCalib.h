#pragma once
#include <opencv2/core.hpp>

#include "Camera.h"

namespace hymson3d {
namespace core {

class CameraCalib {
public:
    CameraCalib(bool debug_mode) { m_debug_mode = debug_mode; }
    ~CameraCalib() = default;

public:
    enum Pattern {
        NOT_EXISTING,
        CHESSBOARD,
        CHARUCOBOARD,
        CIRCLES_GRID,
        ASYMMETRIC_CIRCLES_GRID
    };
    typedef std::shared_ptr<CameraCalib> Ptr;
    bool init(std::string config_file);
    bool process();
    double cal_reprojection_errors();
    void cal_camera_extersics();
    void undistort_image(std::string config_file,
                         cv::Mat& distorted_img,
                         cv::Mat& undistorted_img);
    void undistort_image(cv::Mat& distorted_img, cv::Mat& undistorted_img);
    std::vector<std::string> get_img_list() { return m_img_list; }
    cv::Mat get_camera_matrix() { return m_camera_matrix; }
    cv::Mat get_dist_coeffs() { return m_dist_coeffs; }

private:
    void read_calib_param();
    void write_calib_param();
    void detect_corner();
    void compute_3d_points();
    void calibrate_camera();

public:
    Camera::Ptr m_camera;
    std::string m_calib_img_dir;
    std::string m_calib_file;
    Pattern m_pattern;
    bool m_debug_mode;
    /*winSize is used to control the side length of the search window. */
    int m_win_size;
    std::vector<cv::Mat> RTvecs;

private:
    double total_error;
    int valid_img_num;
    std::vector<std::string> m_img_list;
    cv::Size m_boardsize;
    cv::Size m_image_size;
    float marker_size;
    // marker size defined differently for CharucoBoard. We use square size
    // equaivalent to the marker size here
    std::vector<std::vector<cv::Point2f>> m_point2d_all_images;
    std::vector<std::vector<cv::Point3f>> m_point3d_all_images;

    cv::Mat m_camera_matrix;  // intrinsics
    cv::Mat m_dist_coeffs;    // distortion coefficents
    // rotation and translation
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
};

}  // namespace core
}  // namespace hymson3d
#include "CameraCalib.h"

#include <opencv2/opencv.hpp>

#include "FileSystem.h"
#include "Logger.h"

namespace hymson3d {
namespace core {

bool CameraCalib::init(std::string img_dir) {
    m_calib_img_dir = img_dir;
    return utility::filesystem::ListFilesInDirectory(img_dir, m_img_list);
}

void CameraCalib::write_calib_param() {
    cv::FileStorage fs(m_calib_file, cv::FileStorage::WRITE);
    fs << "calib_image_dir" << m_calib_img_dir;
    fs << "calib_param";
    fs << "{" << "distortion" << m_dist_coeffs;
    fs << "intrinsic" << m_camera_matrix << "}";
    fs.release();
}

void CameraCalib::read_calib_param() {
    cv::FileStorage fs;
    fs.open(m_calib_file, cv::FileStorage::READ);
    fs["calib_image_dir"] >> m_calib_img_dir;
    fs["calib_pattern"] >> m_pattern;
    cv::FileNode param = fs["calib_param"];
    param["distortion"] >> m_dist_coeffs;
    param["intrinsic"] >> m_camera_matrix;
    fs.release();
}

void CameraCalib::detect_corner() {
    valid_img_num = 0;
    for (auto img_path : m_img_list) {
        cv::Mat img = cv::imread(img_path);
        cv::Mat image_gray;
        cvtColor(img, image_gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Point2f> corners;
        // print debug info
        if (valid_img_num == 0) {
            LOG_DEBUG("image raw info:");
            LOG_DEBUG("channels = {}", img.channels());
            LOG_DEBUG("image type = {}", img.type());
            m_image_size.width = img.cols;
            m_image_size.height = img.rows;
            LOG_DEBUG("image width = {}", m_image_size.width);
            LOG_DEBUG("image height = {}", m_image_size.height);
        }

        bool found;
        switch (m_pattern) {
            case CHESSBOARD:
                found = findChessboardCorners(image_gray, m_boardsize, corners);
                break;
            case CHARUCOBOARD:
                LOG_WARN("CHARUCOBOARD pattern not supported yet.");
                break;
            case CIRCLES_GRID:
                found = findCirclesGrid(image_gray, m_boardsize, corners);
                break;
            case ASYMMETRIC_CIRCLES_GRID:
                found = findCirclesGrid(image_gray, m_boardsize, corners,
                                        cv::CALIB_CB_ASYMMETRIC_GRID);
                break;
            default:
                found = false;
                break;
        }
        if (found) {
            cv::TermCriteria criteria(
                    cv::TermCriteria::EPS | cv::TermCriteria::Type::MAX_ITER,
                    30, 0.001);
            cv::cornerSubPix(image_gray, corners,
                             cv::Size(m_win_size, m_win_size), cv::Size(-1, -1),
                             criteria);
            m_point2d_all_images.emplace_back(corners);
        } else {
            LOG_WARN("Corners not found in {}", img_path);
        }
        if (m_debug_mode) {
            drawChessboardCorners(img, m_boardsize, cv::Mat(corners), found);
        }
        valid_img_num++;
    }
}

void CameraCalib::compute_3d_points() {
    std::vector<cv::Point3f> corners_3d;
    switch (m_pattern) {
        case CHESSBOARD:
            for (int i = 0; i < m_boardsize.height; i++) {
                for (int j = 0; j < m_boardsize.width; j++) {
                    corners_3d.push_back(
                            cv::Point3f(j * marker_size, i * marker_size, 0));
                }
            }
            break;
        case CIRCLES_GRID:
            for (int i = 0; i < m_boardsize.height; ++i) {
                for (int j = 0; j < m_boardsize.width; ++j) {
                    corners_3d.push_back(
                            cv::Point3f(j * marker_size, i * marker_size, 0));
                }
            }
            break;
        case CHARUCOBOARD:
            LOG_WARN("CHARUCOBOARD pattern not supported yet.");
            break;
        case ASYMMETRIC_CIRCLES_GRID:
            for (int i = 0; i < m_boardsize.height; i++) {
                for (int j = 0; j < m_boardsize.width; j++) {
                    corners_3d.push_back(cv::Point3f(
                            (2 * j + i % 2) * marker_size, i * marker_size, 0));
                }
            }
            break;
        default:
            break;
    }
    std::vector<std::vector<cv::Point3f>> points3D_all_images(valid_img_num,
                                                              corners_3d);
    m_point3d_all_images = points3D_all_images;
}

void CameraCalib::calibrate_camera() {
    m_camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    m_dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);  // both rad and tangent
    cv::calibrateCamera(m_point3d_all_images, m_point2d_all_images,
                        m_image_size, m_camera_matrix, m_dist_coeffs, rvecs,
                        tvecs);
}

double CameraCalib::cal_reprojection_errors() {
    double err = 0.0;
    std::vector<float> per_image_err;
    per_image_err.resize(valid_img_num);
    total_error = 0;
    size_t total_num_points = 0;
    std::vector<cv::Point2f> points_reproject;
    LOG_INFO("start compute reprojection errors for each image");
    std::vector<cv::Point2f> points_per_image;
    std::vector<cv::Point3f> points3D_per_image;
    for (int i = 0; i < valid_img_num; i++) {
        points_per_image = m_point2d_all_images[i];
        points3D_per_image = m_point3d_all_images[i];
        cv::projectPoints(points3D_per_image, rvecs[i], tvecs[i],
                          m_camera_matrix, m_dist_coeffs, points_reproject);
        // L2 norm of n points
        err = norm(points_per_image, points_reproject, cv::NormTypes::NORM_L2);
        size_t num_points = points3D_per_image.size();
        per_image_err[i] = (float)std::sqrt(err * err / num_points);

        total_error += err * err;
        total_num_points += num_points;
        LOG_INFO("{} th image has average error of {} pixels", i + 1,
                 per_image_err[i]);
    }
    LOG_INFO("Average error: {} pixels",
             std::sqrt(total_error / total_num_points));

    return std::sqrt(total_error / total_num_points);
}

void CameraCalib::cal_camera_extersics() {
    cv::Mat rotate_Mat = cv::Mat::zeros(3, 3, CV_64F);
    cv::Mat rt_quad = cv::Mat::eye(4, 4, CV_64F);
    RTvecs.resize(valid_img_num);
    // TODO:: write r and t to 4x4 matrix
    for (int i = 0; i < valid_img_num; i++) {
        cv::Rodrigues(rvecs[i], rotate_Mat);
    }
}

}  // namespace core
}  // namespace hymson3d
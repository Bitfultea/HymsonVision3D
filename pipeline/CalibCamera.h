#include <opencv2/opencv.hpp>

#include "CameraCalib.h"
#include "FileSystem.h"
#include "fmtfallback.h"

namespace hymson3d {
namespace pipeline {

struct calibration_data {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
};

class CalibCamera {
public:
    CalibCamera() {}
    ~CalibCamera() {}
    static calibration_data* calib(const std::string config_file,
                                   bool debug_mode = false);
    static cv::Mat undisort(const std::string config_file,
                            cv::Mat& img,
                            bool debug_mode = false);

    // static core::CameraCalib::Ptr camera_calibration;
    static calibration_data calib_data;
};
}  // namespace pipeline
}  // namespace hymson3d
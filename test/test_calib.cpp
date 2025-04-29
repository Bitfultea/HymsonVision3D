#include <opencv2/opencv.hpp>
#include <string>
#include "fmtfallback.h"
#include "CalibCamera.h"
#include "CameraCalib.h"
#include "FileSystem.h"
using namespace hymson3d;
using namespace pipeline;
int main() {
    bool debug_mode = true;
    core::CameraCalib calib(debug_mode);
    std::string config_file =
            "/home/charles/Data/Repo/HymsonVision3D/test/test_calib.yaml";
    calib.init(config_file);
    calib.process();
    // calib undistort
    utility::filesystem::MakeDirectory("./undistorted_calib");
    std::vector<std::string> img_list = calib.get_img_list();
    for (auto img_path : img_list) {
        cv::Mat img = cv::imread(img_path);
        cv::Mat undistorted_img;
        calib.undistort_image(img, undistorted_img);
        std::string img_name = img_path.substr(img_path.find_last_of("/") + 1);
        std::string debug_img_path =
                "./undistorted_calib/" + img_name + "_undistorted.jpg";
        cv::imwrite(debug_img_path, undistorted_img);
    }

    // Test pipeline
    std::cout << "Test pipeline interface" << std::endl;
    calibration_data* calib_data = CalibCamera::calib(config_file, debug_mode);
    cv::Mat test_img = cv::imread(img_list[0]);
    cv::Mat fixed_img = CalibCamera::undisort(config_file, test_img);
    cv::imwrite("./test.jpg", fixed_img);
    std::cout << "Test pipeline interface" << std::endl;

    return 0;
}
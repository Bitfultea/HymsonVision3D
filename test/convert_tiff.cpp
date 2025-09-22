#include "Converter.h"
#include "FileTool.h"
#include "Logger.h"
#include "fmtfallback.h"

using namespace hymson3d;
int main() {
    std::string tiff_path =
            "/home/charles/Data/Dataset/Collected/密封钉/yang/3D/"
            "CapTime_Dev7_2025_08_04_00_00_02_204_CurTime_00_00_02_461_"
            "0B5CBP2MBL0P8MF811012063.tiff";

    //     core::converter::tiff_to_pointcloud(
    //             "/home/charles/Data/Dataset/Collected/NG/1/NG/"
    //             "35596_00_06_11_876_0KECB70L000009EAB1229832_"
    //             "0KECB70L000009EAB1229844.tiff",
    //             "test.ply");
    //     geometry::PointCloud::Ptr pointcloud =
    //             std::make_shared<geometry::PointCloud>();
    //     core::converter::tiff_to_pointcloud(
    //             "/home/charles/Data/Dataset/Collected/NG/1/NG/"
    //             "35596_00_06_11_876_0KECB70L000009EAB1229832_"
    //             "0KECB70L000009EAB1229844.tiff",
    //             pointcloud);
    //     utility::write_ply("test_hpy.ply", pointcloud,
    //     utility::FileFormat::BINARY); pointcloud->Clear();
    //     utility::read_ply("test_hpy.ply", pointcloud);
    //     for (int i = 0; i < 5; i++) {
    //         std::cout << pointcloud->points_[i] << std::endl;
    //     }

    //     cv::Mat tiff_image;
    //     utility::read_tiff(
    //             "/home/charles/Data/Dataset/Collected/NG/1/NG/"
    //             "35596_00_06_11_876_0KECB70L000009EAB1229832_"
    //             "0KECB70L000009EAB1229844.tiff",
    //             tiff_image);
    //     geometry::PointCloud::Ptr tiff_pointcloud =
    //             std::make_shared<geometry::PointCloud>();
    //     core::converter::mat_to_pointcloud(tiff_image, tiff_pointcloud);
    //     LOG_INFO("Number of Point in the pointcloud: {}",
    //              tiff_pointcloud->points_.size());

    //     cv::Mat tiff_image;
    //     utility::read_tiff(
    //             "/home/charles/Data/Dataset/Collected/密封钉/yang/3D/"
    //             "CapTime_Dev7_2025_08_04_00_00_02_204_CurTime_00_00_02_461_"
    //             "0B5CBP2MBL0P8MF811012063.tiff",
    //             tiff_image);

    //     std::pair<cv::Mat, cv::Mat> result =
    //             utility::separate_float_components(tiff_image);

    //     cv::Mat integer_part = result.first;
    //     cv::Mat decimal_part = result.second;

    //     // 归一化以便可视化
    //     cv::Mat integer_vis, decimal_vis;

    //     // 将整数部分归一化到0-255范围
    //     cv::normalize(integer_part, integer_vis, 0, 255, cv::NORM_MINMAX,
    //     CV_8U);

    //     // 将小数部分归一化到0-255范围
    //     cv::normalize(decimal_part, decimal_vis, 0, 255, cv::NORM_MINMAX,
    //     CV_8U);

    //     cv::imwrite("integer_part.png", integer_vis);
    //     cv::imwrite("decimal_part.png", decimal_vis);

    cv::Mat height_map, intensity_map;
    utility::read_tiff_libtiff(tiff_path, height_map, intensity_map);
    // 将整数部分归一化到0-255范围
    cv::normalize(height_map, height_map, 0, 255, cv::NORM_MINMAX, CV_8U);

    // 将小数部分归一化到0-255范围
    cv::normalize(intensity_map, intensity_map, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::imwrite("height_map.png", height_map);
    cv::imwrite("intensity_map.png", intensity_map);
    return 0;
}
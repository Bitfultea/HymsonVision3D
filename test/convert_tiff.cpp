#include "Converter.h"
#include "FileTool.h"
#include "Logger.h"
#include "fmtfallback.h"

using namespace hymson3d;
int main() {
    core::converter::tiff_to_pointcloud(
            "/home/charles/Data/Dataset/Collected/NG/1/NG/"
            "35596_00_06_11_876_0KECB70L000009EAB1229832_"
            "0KECB70L000009EAB1229844.tiff",
            "test.ply");
    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();
    core::converter::tiff_to_pointcloud(
            "/home/charles/Data/Dataset/Collected/NG/1/NG/"
            "35596_00_06_11_876_0KECB70L000009EAB1229832_"
            "0KECB70L000009EAB1229844.tiff",
            pointcloud);
    utility::write_ply("test_hpy.ply", pointcloud, utility::FileFormat::BINARY);
    pointcloud->Clear();
    utility::read_ply("test_hpy.ply", pointcloud);
    for (int i = 0; i < 5; i++) {
        std::cout << pointcloud->points_[i] << std::endl;
    }

    cv::Mat tiff_image;
    utility::read_tiff(
            "/home/charles/Data/Dataset/Collected/NG/1/NG/"
            "35596_00_06_11_876_0KECB70L000009EAB1229832_"
            "0KECB70L000009EAB1229844.tiff",
            tiff_image);
    geometry::PointCloud::Ptr tiff_pointcloud =
            std::make_shared<geometry::PointCloud>();
    core::converter::mat_to_pointcloud(tiff_image, tiff_pointcloud);
    LOG_INFO("Number of Point in the pointcloud: {}",
             tiff_pointcloud->points_.size());

    return 0;
}
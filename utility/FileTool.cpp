#include "FileTool.h"
namespace hymson3d {
namespace utility {
bool read_tiff(const std::string& filename, cv::Mat& image) {
    image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        LOG_ERROR("无法读取TIFF文件！");
        return false;
    }

    // 显示图像信息
    LOG_DEBUG("图像路径：{}", filename);
    LOG_DEBUG("图像宽度：{}", image.cols);
    LOG_DEBUG("图像高度：{}", image.rows);
    LOG_DEBUG("通道数：{}", image.channels());
    return true;
}

void write_ply(const std::string& filename,
               const std::vector<Eigen::Vector3d>& points) {
    // Create a point cloud
    pcl::PointCloud<pcl::PointXYZ> cloud;

    // Add points to the cloud
    for (auto pt : points) {
        cloud.points.emplace_back(pcl::PointXYZ(pt.x(), pt.y(), pt.z()));
    }

    // Create a PCD writer
    pcl::PLYWriter writer;

    // Write the point cloud to a file
    writer.write<pcl::PointXYZ>(filename, cloud);
}
}  // namespace utility
}  // namespace hymson3d
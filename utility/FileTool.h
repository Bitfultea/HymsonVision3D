#pragma once
#include <3D/PointCloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
// #include "TinyTIFF/tinytiffreader.h
#include "Logger.h"

namespace hymson3d {
namespace utility {
enum FileFormat { BINARY = 0, ASCII = 1 };

bool read_tiff(const std::string& filename, cv::Mat& image);
void read_tiff(const char* filename);
void write_tiff(const std::string& filename, const cv::Mat& image);
void write_ply(const std::string& filename,
               const std::vector<Eigen::Vector3d>& points);
void write_ply(const std::string& filename,
               geometry::PointCloud::Ptr pointcloud,
               FileFormat format = FileFormat::ASCII);
void read_ply(const std::string& filename,
              std::vector<Eigen::Vector3d>& points);
void read_ply(const std::string& filename,
              geometry::PointCloud::Ptr pointcloud);

}  // namespace utility
}  // namespace hymson3d
// convert data between different types
#pragma once

#include <pcl/point_types.h>

#include "2D/Image.h"
#include "3D/PointCloud.h"
#include "FileTool.h"
#include "Logger.h"

namespace hymson3d {
namespace core {
namespace converter {

// convert tiff data to point cloud
void tiff_to_pointcloud(const std::string& tiff_path,
                        const std::string& ply_path,
                        const Eigen::Vector3d& ratio = Eigen::Vector3d(0.01,
                                                                       0.03,
                                                                       0.001));
void tiff_to_pointcloud(const std::string& tiff_path,
                        geometry::PointCloud::Ptr pointcloud,
                        const Eigen::Vector3d& ratio = Eigen::Vector3d(0.01,
                                                                       0.03,
                                                                       0.001));

void mat_to_pointcloud(const cv::Mat& mat,
                       geometry::PointCloud::Ptr pointcloud);

// convert hymson3d point cloud data to pcl point cloud
void to_pcl_pointcloud(geometry::PointCloud::Ptr src,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr dst);
void pcl_to_hymson3d(pcl::PointCloud<pcl::PointXYZ>::Ptr src,
                     geometry::PointCloud::Ptr dst);
void pcl_to_hymson3d(pcl::PointCloud<pcl::PointXYZ>::Ptr src,
                     pcl::PointCloud<pcl::Normal>::Ptr src_normals,
                     geometry::PointCloud::Ptr dst);

void pcl_to_hymson3d_normals(pcl::PointCloud<pcl::Normal>::Ptr src,
                             geometry::PointCloud::Ptr dst);

}  // namespace converter
}  // namespace core
}  // namespace hymson3d

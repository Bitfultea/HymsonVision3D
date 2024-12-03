#include "Converter.h"

namespace hymson3d {
namespace core {
namespace converter {
void tiff_to_pointcloud(const std::string& tiff_path,
                        const std::string& ply_path,
                        const Eigen::Vector3d& ratio) {
    cv::Mat tiff_image;
    if (!utility::read_tiff(tiff_path, tiff_image)) {
        return;
    }
    std::vector<Eigen::Vector3d> pcd;
    // Reserve memory to avoid reallocations
    pcd.resize(tiff_image.rows * tiff_image.cols);
// Use OpenMP
#pragma omp parallel for
    for (int i = 0; i < tiff_image.rows; ++i) {
        const float* row_ptr =
                tiff_image.ptr<float>(i);  // Get pointer to the row

        float y = i * ratio.y();
        for (int j = 0; j < tiff_image.cols; ++j) {
            float x = j * ratio.x();
            float z = row_ptr[j] * ratio.z();
            pcd[i * tiff_image.cols + j] = Eigen::Vector3d(x, y, z);
        }
    }

    utility::write_ply(ply_path, pcd);
}

void tiff_to_pointcloud(const std::string& tiff_path,
                        geometry::PointCloud::Ptr pointcloud,
                        const Eigen::Vector3d& ratio) {
    cv::Mat tiff_image;
    if (!utility::read_tiff(tiff_path, tiff_image)) {
        return;
    }
    std::vector<Eigen::Vector3d> pcd;
    // Reserve memory to avoid reallocations
    pcd.resize(tiff_image.rows * tiff_image.cols);
// Use OpenMP
#pragma omp parallel for
    for (int i = 0; i < tiff_image.rows; ++i) {
        const float* row_ptr =
                tiff_image.ptr<float>(i);  // Get pointer to the row
        float y = i * ratio.y();
        for (int j = 0; j < tiff_image.cols; ++j) {
            float x = j * ratio.x();
            float z = row_ptr[j] * ratio.z();
            pcd[i * tiff_image.cols + j] = Eigen::Vector3d(x, y, z);
        }
    }  // #pragma omp critical

    pointcloud->points_ = pcd;
}

void mat_to_pointcloud(const cv::Mat& mat,
                       geometry::PointCloud::Ptr pointcloud) {
    std::vector<Eigen::Vector3d> pcd;
    pcd.resize(mat.rows * mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        const float* row_ptr = mat.ptr<float>(i);
        float y = i * 0.03;
#pragma omp parallel for
        for (int j = 0; j < mat.cols; ++j) {
            float x = j * 0.01;
            float z = row_ptr[j] * 0.001;
            pcd[i * mat.cols + j] = Eigen::Vector3d(x, y, z);
        }
    }
    pointcloud->points_ = pcd;
}

void to_pcl_pointcloud(geometry::PointCloud& src,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr dst) {
    for (auto pt : src.points_) {
        dst->push_back(pcl::PointXYZ(pt.x(), pt.y(), pt.z()));
    }
}

void pcl_to_hymson3d(pcl::PointCloud<pcl::PointXYZ>& src,
                     geometry::PointCloud::Ptr dst) {
    for (auto pt : src.points) {
        dst->points_.emplace_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
    }
}

void pcl_to_hymson3d(pcl::PointCloud<pcl::PointXYZ>::Ptr src,
                     pcl::PointCloud<pcl::Normal>::Ptr src_normals,
                     geometry::PointCloud& dst) {
    for (int i = 0; i < src->size(); ++i) {
        dst.points_.emplace_back(Eigen::Vector3d(
                src->points[i].x, src->points[i].y, src->points[i].z));
        dst.normals_.emplace_back(
                Eigen::Vector3d(src_normals->points[i].normal_x,
                                src_normals->points[i].normal_y,
                                src_normals->points[i].normal_z));
    }
}

void pcl_to_hymson3d_normals(pcl::PointCloud<pcl::Normal>::Ptr src,
                             geometry::PointCloud& dst) {
    for (auto pt : src->points) {
        dst.normals_.emplace_back(
                Eigen::Vector3d(pt.normal_x, pt.normal_y, pt.normal_z));
    }
}

}  // namespace converter
}  // namespace core
}  // namespace hymson3d
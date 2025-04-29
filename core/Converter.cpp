#include "Converter.h"

namespace hymson3d {
namespace core {
namespace converter {
void tiff_to_pointcloud(const std::string& tiff_path,
                        const std::string& ply_path,
                        const Eigen::Vector3d& ratio,
                        bool remove_bottom) {
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

// void tiff_to_pointcloud(const std::string& tiff_path,
//                         geometry::PointCloud::Ptr pointcloud,
//                         const Eigen::Vector3d& ratio,
//                         bool remove_bottom) {
//     cv::Mat tiff_image;
//     if (!utility::read_tiff(tiff_path, tiff_image)) {
//         return;
//     }
//     // FIXME: Add support for more data type
//     std::vector<Eigen::Vector3d> pcd;
//     // Reserve memory to avoid reallocations
//     pcd.resize(tiff_image.rows * tiff_image.cols);
//     if (tiff_image.type() == CV_32FC1) {
// // Use OpenMP
// #pragma omp parallel for
//         for (int i = 0; i < tiff_image.rows; ++i) {
//             const float* row_ptr =
//                     tiff_image.ptr<float>(i);  // Get pointer to the row
//             double y = i * ratio.y();
//             for (int j = 0; j < tiff_image.cols; ++j) {
//                 double x = j * ratio.x();
//                 double z = row_ptr[j] * ratio.z();
//                 pcd[i * tiff_image.cols + j] = Eigen::Vector3d(x, y, z);
//             }
//         }
//     } else if (tiff_image.type() == CV_16SC1) {
//         // stored data as 16-bit integer
// // Use OpenMP
// #pragma omp parallel for
//         for (int i = 0; i < tiff_image.rows; ++i) {
//             const short* row_ptr = tiff_image.ptr<short>(i);
//             double y = i * ratio.y();
//             for (int j = 0; j < tiff_image.cols; ++j) {
//                 double x = j * ratio.x();
//                 double z = row_ptr[j] * ratio.z();
//                 pcd[i * tiff_image.cols + j] = Eigen::Vector3d(x, y, z);
//             }
//         }
//     }

//     if (remove_bottom) {
//         std::vector<Eigen::Vector3d> new_pcd;
//         for (auto pt : pcd) {
//             if (pt.z() > 0) {
//                 new_pcd.push_back(pt);
//             }
//         }
//         pcd = new_pcd;
//     }
//     pointcloud->points_ = pcd;
//     LOG_DEBUG("Read from tiff file with pointcloud size: {}",
//               pointcloud->points_.size());
// }

void tiff_to_pointcloud(const std::string& tiff_path,
                        geometry::PointCloud::Ptr pointcloud,
                        const Eigen::Vector3d& ratio,
                        bool remove_bottom) {
    cv::Mat tiff_image;
    if (!utility::read_tiff(tiff_path, tiff_image)) {
        return;
    }
    // FIXME: Add support for more data type
    std::vector<Eigen::Vector3d> pcd;
    // Reserve memory to avoid reallocations
    pcd.resize(tiff_image.rows * tiff_image.cols);
    if (tiff_image.type() == CV_32FC1) {
#pragma omp parallel for
        for (int i = 0; i < tiff_image.rows; ++i) {
            const float* row_ptr =
                    tiff_image.ptr<float>(i);  // Get pointer to the row
            double y = i * ratio.y();
            for (int j = 0; j < tiff_image.cols; ++j) {
                double x = j * ratio.x();
                double z = row_ptr[j] * ratio.z();
                pcd[i * tiff_image.cols + j] = Eigen::Vector3d(x, y, z);
            }
        }
    } else if (tiff_image.type() == CV_16SC1) {
        // stored data as 16-bit integer
#pragma omp parallel for
        for (int i = 0; i < tiff_image.rows; ++i) {
            const short* row_ptr = tiff_image.ptr<short>(i);
            double y = i * ratio.y();
            for (int j = 0; j < tiff_image.cols; ++j) {
                double x = j * ratio.x();
                double z = row_ptr[j] * ratio.z();
                pcd[i * tiff_image.cols + j] = Eigen::Vector3d(x, y, z);
            }
        }
    }

    if (remove_bottom) {
        std::vector<Eigen::Vector3d> new_pcd;
        for (auto pt : pcd) {
            if (pt.z() > 0 && pt.x() > 1.4) {
                // if (pt.z() > 0) {
                new_pcd.push_back(pt);
            }
        }
        pcd = new_pcd;
    }

    pointcloud->points_ = pcd;
    LOG_DEBUG("Read from tiff file with pointcloud size: {}",
              pointcloud->points_.size());
}

// fusion with grayscale intensity
void tiff_to_pointcloud(const std::string& tiff_path,
                        const std::string& intensity_map_path,
                        geometry::PointCloud::Ptr pointcloud,
                        const Eigen::Vector3d& ratio,
                        bool remove_bottom) {
    cv::Mat tiff_image;
    if (!utility::read_tiff(tiff_path, tiff_image)) {
        return;
    }
    cv::Mat intensity_map =
            cv::imread(intensity_map_path, cv::IMREAD_UNCHANGED);
    cv::imwrite("inetsnsity_map.png", intensity_map);
    if (intensity_map.empty()) {
        LOG_ERROR("intensity map is empty");
        return;
    }
    if (intensity_map.size() != tiff_image.size()) {
        LOG_ERROR("intensity map size does not match with tiff image size");
        return;
    }
    if (intensity_map.type() != CV_8UC1) {
        LOG_WARN("intensity map is not grayscale, convert to grayscale");
        cv::cvtColor(intensity_map, intensity_map, cv::COLOR_BGR2GRAY);
    }
    // FIXME: Add support for more data type
    std::vector<Eigen::Vector3d> pcd;
    std::vector<float> intensities;
    // Reserve memory to avoid reallocations
    pcd.resize(tiff_image.rows * tiff_image.cols);
    intensities.resize(tiff_image.rows * tiff_image.cols);
    if (tiff_image.type() == CV_32FC1) {
// Use OpenMP
#pragma omp parallel for
        for (int i = 0; i < tiff_image.rows; ++i) {
            const float* row_ptr = tiff_image.ptr<float>(i);
            const uint8_t* int_row_ptr = intensity_map.ptr<uint8_t>(i);
            double y = i * ratio.y();
            for (int j = 0; j < tiff_image.cols; ++j) {
                double x = j * ratio.x();
                double z = row_ptr[j] * ratio.z();
                pcd[i * tiff_image.cols + j] = Eigen::Vector3d(x, y, z);
                intensities[i * tiff_image.cols + j] =
                        static_cast<float>(int_row_ptr[j]);
            }
        }
    } else if (tiff_image.type() == CV_16SC1) {
        // stored data as 16-bit integer
        // Use OpenMP
        // #pragma omp parallel for
        for (int i = 0; i < tiff_image.rows; ++i) {
            const short* row_ptr = tiff_image.ptr<short>(i);
            const uint8_t* int_row_ptr = intensity_map.ptr<uint8_t>(i);
            double y = i * ratio.y();
            for (int j = 0; j < tiff_image.cols; ++j) {
                double x = j * ratio.x();
                double z = row_ptr[j] * ratio.z();
                pcd[i * tiff_image.cols + j] = Eigen::Vector3d(x, y, z);
                intensities[i * tiff_image.cols + j] =
                        static_cast<float>(int_row_ptr[j]);
            }
        }
    }

    if (remove_bottom) {
        std::vector<Eigen::Vector3d> new_pcd;
        std::vector<float> new_intensities;
        for (size_t i = 0; i < pcd.size(); ++i) {
            if (pcd[i].z() > 0) {
                new_pcd.push_back(pcd[i]);
                new_intensities.push_back(intensities[i]);
            }
        }
        pcd = new_pcd;
        intensities = new_intensities;
    }
    pointcloud->points_ = pcd;
    pointcloud->intensities_ = intensities;
    LOG_DEBUG("Read from tiff file with pointcloud(intensity) size: {}",
              pointcloud->points_.size());
}

void mat_to_pointcloud(const cv::Mat& mat,
                       geometry::PointCloud::Ptr pointcloud,
                       const Eigen::Vector3d& ratio,
                       bool remove_bottom) {
    std::vector<Eigen::Vector3d> pcd;
    pcd.resize(mat.rows * mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        const float* row_ptr = mat.ptr<float>(i);
        float y = i * ratio.y();
#pragma omp parallel for
        for (int j = 0; j < mat.cols; ++j) {
            float x = j * ratio.x();
            float z = row_ptr[j] * ratio.z();
            pcd[i * mat.cols + j] = Eigen::Vector3d(x, y, z);
        }
    }
    pointcloud->points_ = pcd;
}

void to_pcl_pointcloud(geometry::PointCloud& src,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr dst) {
    dst->reserve(src.points_.size());
    for (auto pt : src.points_) {
        dst->push_back(pcl::PointXYZ(pt.x(), pt.y(), pt.z()));
    }
}

void to_pcl_pointcloud(geometry::PointCloud& src,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr dst,
                       pcl::PointCloud<pcl::Normal>::Ptr dst_normals) {
    dst->reserve(src.points_.size());
    dst_normals->reserve(src.points_.size());
    for (int i = 0; i < src.points_.size(); ++i) {
        dst->push_back(pcl::PointXYZ(src.points_[i].x(), src.points_[i].y(),
                                     src.points_[i].z()));
        dst_normals->emplace_back(pcl::Normal(
                src.normals_[i].x(), src.normals_[i].y(), src.normals_[i].z()));
    }
}

void pcl_to_hymson3d(pcl::PointCloud<pcl::PointXYZ>& src,
                     geometry::PointCloud::Ptr dst) {
    dst->points_.reserve(src.size());
    for (auto pt : src.points) {
        dst->points_.emplace_back(Eigen::Vector3d(pt.x, pt.y, pt.z));
    }
}

void pcl_to_hymson3d(pcl::PointCloud<pcl::PointXYZ>::Ptr src,
                     pcl::PointCloud<pcl::Normal>::Ptr src_normals,
                     geometry::PointCloud& dst) {
    dst.points_.reserve(src->size());
    dst.normals_.reserve(src_normals->size());
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
    dst.normals_.reserve(src->size());
    for (auto pt : src->points) {
        dst.normals_.emplace_back(
                Eigen::Vector3d(pt.normal_x, pt.normal_y, pt.normal_z));
    }
}

}  // namespace converter
}  // namespace core
}  // namespace hymson3d
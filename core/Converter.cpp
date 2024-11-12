#include "Converter.h"

namespace hymson3d {
namespace core {
namespace converter {
void tiff_to_pointcloud(const std::string& tiff_path) {
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
        float y = i * 0.03;
        for (int j = 0; j < tiff_image.cols; ++j) {
            float x = j * 0.01;
            float z = row_ptr[j] * 0.001;
            pcd[i * tiff_image.cols + j] = Eigen::Vector3d(x, y, z);
            // #pragma omp critical
            //             { pcd.emplace_back(x, y, z); }
        }
    }

    utility::write_ply("test.ply", pcd);
}
}  // namespace converter
}  // namespace core
}  // namespace hymson3d
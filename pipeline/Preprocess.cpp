#include "Preprocess.h"

#include "Filter.h"

namespace hymson3d {
namespace pipeline {

// TODO:Unit test for this part
void Preprocess::RemoveOutliers(geometry::PointCloud::Ptr src_cloud,
                                geometry::PointCloud::Ptr dst_cloud,
                                PreprocessParam& param) {
    core::Filter filter;
    std::tuple<PointCloud::Ptr, std::vector<size_t>> filter_out;
    switch (param.op) {
        case StatisticalOutlierRemoval:
            filter_out = filter.StatisticalOutliers(src_cloud, param.nb_points,
                                                    param.radius);
            break;
        case RadiusOutlierRemoval:
            filter_out = filter.RadiusOutliers(src_cloud, param.nb_points,
                                               param.std_ratio);
            break;
        default:
            break;
    }
    dst_cloud = std::get<0>(filter_out);
}

void Preprocess::DownSample(geometry::PointCloud::Ptr src_cloud,
                            geometry::PointCloud::Ptr dst_cloud,
                            PreprocessParam& param) {
    core::Filter sampler;
    switch (param.op) {
        case VoxelGrid: {
            std::tuple<PointCloud::Ptr, Eigen::MatrixXi,
                       std::vector<std::vector<int>>>
                    voxel_downsample_result;
            voxel_downsample_result =
                    sampler.VoxelDownSample(src_cloud, param.voxel_size);
            dst_cloud = std::get<0>(voxel_downsample_result);
            break;
        }
        case Uniform:
            std::cout << "Not implemented" << std::endl;
            break;
        case Random:
            std::cout << "Not implemented" << std::endl;
            break;
        default:
            std::cout << "Not implemented" << std::endl;
            break;
    }
}

}  // namespace pipeline
}  // namespace hymson3d
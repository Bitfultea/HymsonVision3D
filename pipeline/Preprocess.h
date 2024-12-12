#pragma once
#include "3D/PointCloud.h"

namespace hymson3d {
namespace pipeline {

enum Operator {
    StatisticalOutlierRemoval,
    RadiusOutlierRemoval,
    VoxelGrid,
    Uniform,
    Random,
    PassThrough
};

// TODO::Use union instead. Chech witch CAO
struct PreprocessParam {
    Operator op;
    double radius;
    size_t nb_points;
    float voxel_size;
    double std_ratio;
};

// TODO::More preprcess on
// https://doc.cgal.org/latest/Point_set_processing_3/index.html#Point_set_processing_3Simplification
class Preprocess {
public:
    Preprocess() {}
    ~Preprocess() {}

public:
    static void RemoveOutliers(geometry::PointCloud::Ptr src_cloud,
                               geometry::PointCloud::Ptr dst_cloud,
                               PreprocessParam& param);

    // TODO::Implement MLS, Bilateral and Jet Smoothing
    static void Smooth(geometry::PointCloud::Ptr src_cloud,
                       geometry::PointCloud::Ptr dst_cloud,
                       PreprocessParam& param);

    static void DownSample(geometry::PointCloud::Ptr src_cloud,
                           geometry::PointCloud::Ptr dst_cloud,
                           PreprocessParam& param);
};

}  // namespace pipeline
}  // namespace hymson3d
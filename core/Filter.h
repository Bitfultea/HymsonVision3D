#pragma once

#include <memory>
#include <vector>

#include "3D/PointCloud.h"
#include "Hash.h"

using namespace hymson3d::geometry;
namespace hymson3d {
namespace core {

class AccumulatedPoint {
public:
    void AddPoint(const PointCloud &cloud, int index) {
        point_ += cloud.points_[index];
        if (cloud.HasNormals()) {
            if (!std::isnan(cloud.normals_[index](0)) &&
                !std::isnan(cloud.normals_[index](1)) &&
                !std::isnan(cloud.normals_[index](2))) {
                normal_ += cloud.normals_[index];
            }
        }
        if (cloud.HasColors()) {
            color_ += cloud.colors_[index];
        }
        if (cloud.HasCovariances()) {
            covariance_ += cloud.covariances_[index];
        }
        // FIXME:should take the majority of label instead of average
        // if (cloud.HasLabels()) {
        //     label_ += cloud.labels_[index];
        // }
        if (cloud.HasIntensities()) {
            intensity_ += cloud.intensities_[index];
        }
        if (cloud.HasCurvatures()) {
            curvature_->mean_curvature +=
                    cloud.curvatures_[index]->mean_curvature;
            curvature_->gaussian_curvature +=
                    cloud.curvatures_[index]->gaussian_curvature;
            curvature_->total_curvature +=
                    cloud.curvatures_[index]->total_curvature;
        }
        num_of_points_++;
    }

    Eigen::Vector3d GetAveragePoint() const {
        return point_ / double(num_of_points_);
    }

    Eigen::Vector3d GetAverageNormal() const {
        // Call NormalizeNormals() afterwards if necessary
        return normal_ / double(num_of_points_);
    }

    Eigen::Vector3d GetAverageColor() const {
        return color_ / double(num_of_points_);
    }

    Eigen::Matrix3d GetAverageCovariance() const {
        return covariance_ / double(num_of_points_);
    }

    int GetAverageIntensity() const { return intensity_ / num_of_points_; }

    curvature *GetAverageCurvature() const {
        curvature *voxel_curvature = new curvature();
        voxel_curvature->mean_curvature =
                curvature_->mean_curvature / num_of_points_;
        voxel_curvature->gaussian_curvature =
                curvature_->gaussian_curvature / num_of_points_;
        voxel_curvature->total_curvature =
                curvature_->total_curvature / num_of_points_;
        return voxel_curvature;
    }

public:
    int num_of_points_ = 0;
    Eigen::Vector3d point_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d color_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance_ = Eigen::Matrix3d::Zero();
    int label_ = 0;
    float intensity_ = 0;
    curvature *curvature_ = new curvature{0, 0, 0};
};

class point_cubic_id {
public:
    size_t point_id;
    int cubic_id;
};

class AccumulatedPointForTrace : public AccumulatedPoint {
public:
    void AddPoint(const PointCloud &cloud,
                  size_t index,
                  int cubic_index,
                  bool approximate_class) {
        point_ += cloud.points_[index];
        if (cloud.HasNormals()) {
            if (!std::isnan(cloud.normals_[index](0)) &&
                !std::isnan(cloud.normals_[index](1)) &&
                !std::isnan(cloud.normals_[index](2))) {
                normal_ += cloud.normals_[index];
            }
        }
        if (cloud.HasColors()) {
            if (approximate_class) {
                auto got = classes.find(int(cloud.colors_[index][0]));
                if (got == classes.end())
                    classes[int(cloud.colors_[index][0])] = 1;
                else
                    classes[int(cloud.colors_[index][0])] += 1;
            } else {
                color_ += cloud.colors_[index];
            }
        }
        if (cloud.HasCovariances()) {
            covariance_ += cloud.covariances_[index];
        }
        point_cubic_id new_id;
        new_id.point_id = index;
        new_id.cubic_id = cubic_index;
        original_id.push_back(new_id);
        num_of_points_++;
    }

    Eigen::Vector3d GetMaxClass() {
        int max_class = -1;
        int max_count = -1;
        for (auto it = classes.begin(); it != classes.end(); it++) {
            if (it->second > max_count) {
                max_count = it->second;
                max_class = it->first;
            }
        }
        return Eigen::Vector3d(max_class, max_class, max_class);
    }

    std::vector<point_cubic_id> GetOriginalID() { return original_id; }

private:
    // original point cloud id in higher resolution + its cubic id
    std::vector<point_cubic_id> original_id;
    std::unordered_map<int, int> classes;
};

enum Axis { Axi_X, Axi_Y, Axi_Z };

class Filter {
public:
    Filter() = default;
    ~Filter() = default;

public:
    std::tuple<PointCloud::Ptr, Eigen::MatrixXi, std::vector<std::vector<int>>>
    VoxelDownSample(PointCloud::Ptr point_cloud, float voxel_size);

    PointCloud::Ptr UniformDownSample(PointCloud::Ptr point_cloud,
                                      size_t every_k_points);

    PointCloud::Ptr RandomDownSample(PointCloud::Ptr point_cloud,
                                     double sampling_ratio);

    PointCloud::Ptr IndexDownSample(PointCloud::Ptr point_cloud,
                                    const std::vector<size_t> &indices,
                                    bool invert = false);

    PointCloud::Ptr AxiFilter(PointCloud::Ptr point_cloud,
                              std::pair<double, double> range,
                              Axis axis);

    std::tuple<PointCloud::Ptr, std::vector<size_t>> RadiusOutliers(
            PointCloud::Ptr pointcloud, size_t nb_points, double search_radius);

    std::tuple<PointCloud::Ptr, std::vector<size_t>> StatisticalOutliers(
            PointCloud::Ptr pointcloud, size_t nb_neighbors, double std_ratio);
};

}  // namespace core
}  // namespace hymson3d
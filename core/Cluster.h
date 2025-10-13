#pragma once
#include <random>

#include "3D/KDtree.h"
#include "3D/PointCloud.h"
#include "fmtfallback.h"

namespace hymson3d {
namespace core {

class Cluster {
public:
    static void RegionGrowing_PCL(geometry::PointCloud& cloud,
                                  float normal_degree,
                                  float curvature_threshold,
                                  size_t min_cluster_size,
                                  size_t max_cluster_size,
                                  int knn);

    static void RegionGrowing_PCL(geometry::PointCloud& cloud,
                                  float normal_degree,
                                  float curvature_threshold,
                                  int knn);

    static int DBSCANCluster(geometry::PointCloud& cloud,
                             double eps,
                             size_t min_points);

    static void RegionGrowingCluster(geometry::PointCloud& cloud,
                                     float radius,
                                     float normal_degree,
                                     float curvature_threshold,
                                     int min_cluster_size);

    static int PlanarCluster(geometry::PointCloud& cloud,
                             float radius,
                             float normal_degree,
                             float curvature_threshold = 0.0f,
                             bool use_curvature = false,
                             int min_cluster_size = 100,
                             bool debug_mode = true);

    static Eigen::Vector3d GenerateRandomColor() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        Eigen::Vector3d random_vector;
        random_vector << dis(gen), dis(gen), dis(gen);
        return random_vector;
    }

private:
    static void paint_cluster(geometry::PointCloud& cloud, int num_cluster) {
        std::vector<Eigen::Vector3d> cluster_colors;
        for (int i = 0; i < num_cluster; i++) {
            cluster_colors.emplace_back(GenerateRandomColor());
        }
        cloud.colors_.resize(cloud.points_.size());
#pragma omp parallel for
        for (int i = 0; i < cloud.points_.size(); i++) {
            if (cloud.labels_[i] >= 0)
                cloud.colors_[i] = cluster_colors[cloud.labels_[i]];
        }
    }
};

}  // namespace core
}  // namespace hymson3d
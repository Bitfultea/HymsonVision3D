#pragma once
#include <random>

#include "3D/KDtree.h"
#include "3D/PointCloud.h"

namespace hymson3d {
namespace core {

class Cluster {
public:
    static void RegionGrowing_PCL(geometry::PointCloud& cloud,
                                  size_t min_cluster_size,
                                  size_t max_cluster_size,
                                  int knn);

private:
    static Eigen::Vector3d GenerateRandomColor() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        Eigen::Vector3d random_vector;
        random_vector << dis(gen), dis(gen), dis(gen);
        return random_vector;
    }
};

}  // namespace core
}  // namespace hymson3d
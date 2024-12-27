
#include "Cluster.h"

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/region_growing.h>

#include <iostream>
#include <vector>

#include "3D/KDtree.h"
#include "Converter.h"

namespace hymson3d {
namespace core {
void Cluster::RegionGrowing_PCL(geometry::PointCloud& cloud,
                                float normal_degree,
                                float curvature_threshold,
                                size_t min_cluster_size,
                                size_t max_cluster_size,
                                int knn) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(
            new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::Search<pcl::PointXYZ>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZ>);

    if (!cloud.HasNormals()) {
        converter::to_pcl_pointcloud(cloud, pcl_cloud);
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
        normal_estimator.setSearchMethod(tree);
        normal_estimator.setViewPoint(0, 0, std::numeric_limits<float>::max());
        normal_estimator.setInputCloud(pcl_cloud);
        normal_estimator.setKSearch(50);
        normal_estimator.compute(*normals);
    } else {
        LOG_DEBUG("Already have normals. Use previsous computed normals");
        converter::to_pcl_pointcloud(cloud, pcl_cloud, normals);
    }

    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::removeNaNFromPointCloud(*pcl_cloud, *indices);

    pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
    reg.setMinClusterSize(min_cluster_size);
    reg.setMaxClusterSize(max_cluster_size);
    reg.setSearchMethod(tree);
    reg.setNumberOfNeighbours(knn);
    reg.setInputCloud(pcl_cloud);
    reg.setIndices(indices);
    reg.setInputNormals(normals);
    reg.setSmoothnessThreshold(normal_degree / 180.0 * M_PI);
    reg.setCurvatureThreshold(curvature_threshold);

    // pointindices.indices = vector<std::int32_t>
    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);  // clusters is a vector of pcl::PointIndices

    // cloud.PaintUniformColor(Eigen::Vector3d(1, 1, 1));
    for (size_t i = 0; i < clusters.size(); i++) {
        Eigen::Vector3d color = GenerateRandomColor();
        for (size_t j = 0; j < clusters[i].indices.size(); j++) {
            cloud.colors_[clusters[i].indices[j]] = color;
            cloud.labels_[clusters[i].indices[j]] = 1;
        }
    }
    std::cout << "Cluster size: " << clusters.size() << std::endl;
}

int Cluster::DBSCANCluster(geometry::PointCloud& cloud,
                           double eps,
                           size_t min_points) {
    // geometry::PointCloud cloud;
    // cloud.points_.resize(src.points_.size());
    // for (size_t i = 0; i < src.points_.size(); i++) {
    //     cloud.points_[i] =
    //             Eigen::Vector3d(src.points_[i].x(), src.points_[i].y(), 0);
    // }

    geometry::KDTree kdtree;
    kdtree.SetData(cloud);

    // Precompute all neighbors.
    std::vector<std::vector<int>> nbs(cloud.points_.size());
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < int(cloud.points_.size()); ++idx) {
        std::vector<double> dists2;
        kdtree.SearchRadius(cloud.points_[idx], eps, nbs[idx], dists2);
    }

    // Set all labels to undefined (-2).
    LOG_DEBUG("Compute Clusters");
    std::vector<int> labels(cloud.points_.size(), -2);
    int cluster_label = 0;
    for (size_t idx = 0; idx < cloud.points_.size(); ++idx) {
        // Label is not undefined.
        if (labels[idx] != -2) {
            continue;
        }

        // Check density.
        if (nbs[idx].size() < min_points) {
            labels[idx] = -1;
            continue;
        }

        std::unordered_set<int> nbs_next(nbs[idx].begin(), nbs[idx].end());
        std::unordered_set<int> nbs_visited;
        nbs_visited.insert(int(idx));

        labels[idx] = cluster_label;
        while (!nbs_next.empty()) {
            int nb = *nbs_next.begin();
            nbs_next.erase(nbs_next.begin());
            nbs_visited.insert(nb);

            // Noise label.
            if (labels[nb] == -1) {
                labels[nb] = cluster_label;
            }
            // Not undefined label.
            if (labels[nb] != -2) {
                continue;
            }
            labels[nb] = cluster_label;

            if (nbs[nb].size() >= min_points) {
                for (int qnb : nbs[nb]) {
                    if (nbs_visited.count(qnb) == 0) {
                        nbs_next.insert(qnb);
                    }
                }
            }
        }

        cluster_label++;
    }

    cloud.labels_ = labels;
    paint_cluster(cloud, cluster_label);
    LOG_INFO("Done Compute Clusters: {}", cluster_label);
    return cluster_label;
}

}  // namespace core
}  // namespace hymson3d
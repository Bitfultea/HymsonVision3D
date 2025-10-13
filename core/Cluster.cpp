
#include "Cluster.h"
#define PCL_MULTITHREADING 4
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
#include "Normal.h"

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

void Cluster::RegionGrowingCluster(geometry::PointCloud& cloud,
                                   float radius,
                                   float normal_degree,
                                   float curvature_threshold,
                                   int min_cluster_size) {
    geometry::KDTree kdtree;
    kdtree.SetData(cloud);
    // compute normals
    if (!cloud.HasNormals()) {
        geometry::KDTreeSearchParamRadius param(radius);
        feature::ComputeNormals_PCA(cloud, param);
        feature::orient_normals_towards_positive_z(cloud);
    }
    // compute curvature alternatively
    if (!cloud.HasCurvatures()) {
        double sum_angle = 0.0;
        int neighbor_count = 0;
        cloud.curvatures_.resize(cloud.points_.size());
#pragma omp parallel for
        for (int i = 0; i < cloud.normals_.size(); i++) {
            std::vector<int> neighbors;
            std::vector<double> dists2;
            kdtree.SearchRadius(cloud.points_[i], radius, neighbors, dists2);
            for (size_t j = 0; j < neighbors.size(); j++) {
                if (i != j) {
                    double product = cloud.normals_[i].dot(cloud.normals_[j]);
                    double angle = acos(product);
                    sum_angle += angle;
                    neighbor_count++;
                }
            }
            geometry::curvature* curvature = new geometry::curvature;
            if (neighbor_count > 0) {
                curvature->total_curvature = sum_angle / neighbor_count;
            } else {
                curvature->total_curvature = 0.0;
            }
            cloud.curvatures_[i] = curvature;
        }
    }

    std::vector<int> labels(cloud.points_.size(), -1);
    int cluster_label = 0;
    for (size_t i = 0; i < cloud.points_.size(); i++) {
        if (labels[i] != -1) continue;

        std::vector<int> cluster;
        std::unordered_set<int> visited;
        cluster.push_back(i);
        visited.insert(i);

        while (!cluster.empty()) {
            int current = cluster.back();
            cluster.pop_back();

            std::vector<int> neighbors;
            std::vector<double> dists2;
            kdtree.SearchRadius(cloud.points_[current], radius, neighbors,
                                dists2);
            for (int neighbor : neighbors) {
                if (visited.find(neighbor) != visited.end()) continue;

                // check the normal degree
                double angle = acos(cloud.normals_[current].x() *
                                            cloud.normals_[neighbor].x() +
                                    cloud.normals_[current].y() *
                                            cloud.normals_[neighbor].y() +
                                    cloud.normals_[current].z() *
                                            cloud.normals_[neighbor].z());
                // std::cout << "angle: " << angle << std::endl;
                if (angle > normal_degree / 180.0 * M_PI) continue;

                // check the curvature
                if (std::abs(cloud.curvatures_[neighbor]->total_curvature -
                             cloud.curvatures_[current]->total_curvature) >
                            curvature_threshold ||
                    std::abs(cloud.intensities_[neighbor] -
                             cloud.intensities_[current]) > 100)
                    continue;

                // // check the intensity
                // if (std::abs(cloud.intensities_[neighbor] -
                //              cloud.intensities_[current]) > 50) {
                //     continue;
                // }
                // check the distance(not necessary)
                // if (dists2[neighbor] > 0.1 * 0.1) continue;

                cluster.push_back(neighbor);
                visited.insert(neighbor);
                labels[neighbor] = cluster_label;
            }
        }

        if (visited.size() >= min_cluster_size) {
            cluster_label++;
        } else {
            for (int idx : visited) {
                labels[idx] = -1;
            }
        }
    }
    cloud.labels_ = labels;

    // colorise the different clusters
    paint_cluster(cloud, cluster_label);
    LOG_INFO("Done Compute Clusters: {}", cluster_label);
}

int Cluster::PlanarCluster(geometry::PointCloud& cloud,
                           float radius,
                           float normal_degree,
                           float curvature_threshold,
                           bool use_curvature,
                           int min_cluster_size,
                           bool debug_mode) {
    LOG_INFO(
            "Start Planar Clusters with parameters: radius: {}, normal_degree: "
            "{},  min_cluster_size: {}",
            radius, normal_degree, min_cluster_size);

    geometry::KDTree kdtree;
    kdtree.SetData(cloud);
    // compute normals
    if (!cloud.HasNormals()) {
        geometry::KDTreeSearchParamRadius param(radius);
        feature::ComputeNormals_PCA(cloud, param);
        feature::orient_normals_towards_positive_z(cloud);
    }

    // compute curvature alternatively
    if (!cloud.HasCurvatures() && use_curvature) {
        double sum_angle = 0.0;
        int neighbor_count = 0;
        cloud.curvatures_.resize(cloud.points_.size());
#pragma omp parallel for
        for (int i = 0; i < cloud.normals_.size(); i++) {
            std::vector<int> neighbors;
            std::vector<double> dists2;
            kdtree.SearchRadius(cloud.points_[i], radius, neighbors, dists2);
            for (size_t j = 0; j < neighbors.size(); j++) {
                if (i != j) {
                    double product = cloud.normals_[i].dot(cloud.normals_[j]);
                    double angle = acos(product);
                    sum_angle += angle;
                    neighbor_count++;
                }
            }
            geometry::curvature* curvature = new geometry::curvature;
            if (neighbor_count > 0) {
                curvature->total_curvature = sum_angle / neighbor_count;
            } else {
                curvature->total_curvature = 0.0;
            }
            cloud.curvatures_[i] = curvature;
        }
    }

    // Perform actual planar region growing
    std::vector<int> labels(cloud.points_.size(), -1);
    int cluster_label = 0;
    for (size_t i = 0; i < cloud.points_.size(); i++) {
        if (labels[i] != -1) continue;

        std::vector<int> cluster;
        std::unordered_set<int> visited;
        cluster.push_back(i);
        visited.insert(i);

        while (!cluster.empty()) {
            int current = cluster.back();
            cluster.pop_back();

            std::vector<int> neighbors;
            std::vector<double> dists2;
            kdtree.SearchRadius(cloud.points_[current], radius, neighbors,
                                dists2);
            for (int neighbor : neighbors) {
                if (visited.find(neighbor) != visited.end()) continue;

                // check the normal degree
                double angle = acos(cloud.normals_[current].x() *
                                            cloud.normals_[neighbor].x() +
                                    cloud.normals_[current].y() *
                                            cloud.normals_[neighbor].y() +
                                    cloud.normals_[current].z() *
                                            cloud.normals_[neighbor].z());

                if (angle > normal_degree / 180.0 * M_PI) continue;

                // check the curvature
                if (use_curvature) {
                    if (std::abs(cloud.curvatures_[neighbor]->total_curvature -
                                 cloud.curvatures_[current]->total_curvature) >
                                curvature_threshold ||
                        std::abs(cloud.intensities_[neighbor] -
                                 cloud.intensities_[current]) > 100)
                        continue;
                }
                cluster.push_back(neighbor);
                visited.insert(neighbor);
                labels[neighbor] = cluster_label;
            }
        }

        if (visited.size() >= min_cluster_size) {
            cluster_label++;
        } else {
            for (int idx : visited) {
                labels[idx] = -1;
            }
        }
    }
    cloud.labels_ = labels;

    // colorise the different clusters
    if (debug_mode) paint_cluster(cloud, cluster_label);
    LOG_INFO("Done Compute Clusters: {}", cluster_label);
    return cluster_label;
}

}  // namespace core
}  // namespace hymson3d

#include "Cluster.h"

#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/region_growing.h>

#include <iostream>
#include <vector>

#include "Converter.h"

namespace hymson3d {
namespace core {
void Cluster::RegionGrowing_PCL(geometry::PointCloud& cloud,
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
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        normal_estimator.setSearchMethod(tree);
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
    reg.setSmoothnessThreshold(1.0 / 180.0 * M_PI);
    reg.setCurvatureThreshold(0.05);

    // pointindices.indices = vector<std::int32_t>
    std::vector<pcl::PointIndices> clusters;
    reg.extract(clusters);  // clusters is a vector of pcl::PointIndices

    cloud.PaintUniformColor(Eigen::Vector3d(1, 1, 1));
    for (size_t i = 0; i < clusters.size(); i++) {
        Eigen::Vector3d color = GenerateRandomColor();
        for (size_t j = 0; j < clusters[i].indices.size(); j++) {
            cloud.colors_[clusters[i].indices[j]] = color;
        }
    }
    std::cout << "Cluster size: " << clusters.size() << std::endl;
}

}  // namespace core
}  // namespace hymson3d
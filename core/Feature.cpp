#include "Feature.h"

#include <pcl/features/feature.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d.h>  // 法线
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>

#include "Converter.h"

namespace hymson3d {
namespace core {
namespace feature {

typedef pcl::PointCloud<pcl::FPFHSignature33> fpfhFeature;

Eigen::MatrixXf compute_fpfh(geometry::PointCloud& cloud) {
    LOG_DEBUG("Start Computing FPFH feature");
    Eigen::MatrixXf fpfh_data;
    fpfh_data.resize(cloud.points_.size(), 33);
    fpfh_data.setZero();

    // Convert to PCL PointCloud
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
        normal_estimator.setKSearch(30);
        // normal_estimator.setRadiusSearch(30);
        normal_estimator.compute(*normals);
    } else {
        LOG_DEBUG("Already have normals. Use previsous computed normals");
        converter::to_pcl_pointcloud(cloud, pcl_cloud, normals);
    }

    // Compute FPFH feature
    fpfhFeature::Ptr fpfh(new fpfhFeature);
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33>
            est_fpfh;
    est_fpfh.setNumberOfThreads(8);
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree4 (new
    // pcl::search::KdTree<pcl::PointXYZ> ());
    est_fpfh.setInputCloud(pcl_cloud);
    est_fpfh.setInputNormals(normals);
    est_fpfh.setSearchMethod(tree);
    // est_fpfh.setKSearch(12);
    est_fpfh.setRadiusSearch(0.05);
    est_fpfh.compute(*fpfh);

#pragma omp parallel for
    for (int i = 0; i < cloud.points_.size(); i++) {
        for (int j = 0; j < 33; ++j) {
            fpfh_data(i, j) = fpfh->points[i].histogram[j];
        }
    }

    LOG_DEBUG("Completing Computing FPFH feature");
    return fpfh_data;
}

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
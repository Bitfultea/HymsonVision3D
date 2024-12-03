#include "Normal.h"

#include <pcl/features/normal_3d.h>
#include <pcl/point_types.h>

#include "Converter.h"
#include "MathTool.h"

namespace hymson3d {
namespace core {
namespace feature {

void ComputeNormals_PCL(geometry::PointCloud& cloud,
                        geometry::KDTreeSearchParam& param) {
    if (param.GetSearchType() !=
        geometry::KDTreeSearchParam::SearchType::Radius) {
        LOG_ERROR("Invalid search type for ComputeNormals_PCL");
        return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcl(
            new pcl::PointCloud<pcl::PointXYZ>);
    converter::to_pcl_pointcloud(cloud, cloud_pcl);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> estimator;
    estimator.setInputCloud(cloud_pcl);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZ>());
    estimator.setSearchMethod(tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
            new pcl::PointCloud<pcl::Normal>);
    estimator.setRadiusSearch(
            static_cast<hymson3d::geometry::KDTreeSearchParamRadius&>(param)
                    .radius_);
    estimator.compute(*cloud_normals);

    converter::pcl_to_hymson3d_normals(cloud_normals, cloud);
}

void ComputeNormals_PCA(geometry::PointCloud& cloud,
                        geometry::KDTreeSearchParam& param) {
    if (!cloud.HasNormals()) {
        cloud.normals_.resize(cloud.points_.size());
    }
    std::vector<Eigen::Matrix3d> covariances;
    if (!cloud.HasCovariances()) {
        const auto& points = cloud.points_;
        std::vector<Eigen::Matrix3d> covariances;
        covariances.resize(points.size());

        hymson3d::geometry::KDTree kdtree;
        kdtree.SetData(cloud);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)points.size(); i++) {
            std::vector<int> indices;
            std::vector<double> distance2;
            if (kdtree.Search(points[i], param, indices, distance2) >= 3) {
                auto covariance = utility::ComputeCovariance(points, indices);
                if (cloud.HasCovariances() && covariance.isIdentity(1e-4)) {
                    covariances[i] = cloud.covariances_[i];
                } else {
                    covariances[i] = covariance;
                }
            } else {
                covariances[i] = Eigen::Matrix3d::Identity();
            }
        }

    } else {
        covariances = cloud.covariances_;
    }
#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)covariances.size(); i++) {
        // auto normal = ComputeNormal(covariances[i], fast_normal_computation);
        Eigen::Vector3d normal =
                utility::mathtool::FastEigen3x3(covariances[i]).normalized();
        // TODO:oritent the normal w.r.t
        // viewpoint/tangentplane/givendirection
        cloud.normals_[i] = normal;
    }
}

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
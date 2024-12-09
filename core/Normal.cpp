#include "Normal.h"

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>
#include <pcl/search/organized.h>

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

    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> estimator;

    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
    //         new pcl::search::KdTree<pcl::PointXYZ>());

    // Create a search tree, use KDTreee for non-organized data.
    pcl::search::Search<pcl::PointXYZ>::Ptr tree;
    std::cout << cloud_pcl->isOrganized() << std::endl;
    std::cout << cloud_pcl->height << std::endl;
    if (cloud_pcl->isOrganized()) {
        tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZ>());
    } else {
        tree.reset(new pcl::search::KdTree<pcl::PointXYZ>(false));
    }

    estimator.setInputCloud(cloud_pcl);
    estimator.setViewPoint(std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max());
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
        // viewpoint/tangentplane/givendirection
        cloud.normals_[i] = normal;
    }
}

// TODO:oritent the normal w.r.t
void orient_normals_towards_positive_z(
        geometry::PointCloud& cloud,
        const Eigen::Vector3d& orientation_reference
        /* = Eigen::Vector3d(0.0, 0.0, 1.0)*/) {
    if (!cloud.HasNormals()) {
        LOG_ERROR("No normals in the PointCloud. Can not orient normals");
    }
#pragma omp parallel for
    for (int i = 0; i < (int)cloud.points_.size(); i++) {
        auto& normal = cloud.normals_[i];
        if (normal.norm() == 0.0) {
            normal = orientation_reference;
        } else if (normal.dot(orientation_reference) < 0.0) {
            normal *= -1.0;  // flip the normal
        }
    }
}

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
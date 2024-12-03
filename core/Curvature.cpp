#include "Curvature.h"

#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/point_types.h>

#include <cmath>

#include "3D/PointCloud.h"
#include "Converter.h"
#include "Normal.h"

namespace hymson3d {
namespace core {
namespace feature {

void ComputeCurvature_PCL(geometry::PointCloud& cloud,
                          geometry::KDTreeSearchParam& param) {
    if (param.GetSearchType() !=
        geometry::KDTreeSearchParam::SearchType::Radius) {
        LOG_ERROR("Invalid search type for Compute Curvature");
        return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcl(
            new pcl::PointCloud<pcl::PointXYZ>);
    converter::to_pcl_pointcloud(cloud, cloud_pcl);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
            new pcl::search::KdTree<pcl::PointXYZ>());
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
            new pcl::PointCloud<pcl::Normal>);

    if (!cloud.HasNormals()) {
        LOG_INFO("Estimate normal for curvature computation");
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        normal_estimator.setInputCloud(cloud_pcl);
        normal_estimator.setSearchMethod(tree);
        std::cout << static_cast<hymson3d::geometry::KDTreeSearchParamRadius&>(
                             param)
                             .radius_
                  << std::endl;
        normal_estimator.setRadiusSearch(
                static_cast<hymson3d::geometry::KDTreeSearchParamRadius&>(param)
                        .radius_);
        normal_estimator.compute(*cloud_normals);
        std::cout << (*cloud_normals)[0] << std::endl;
    } else {
        cloud_normals->reserve(cloud.normals_.size());
        for (auto pt : cloud.normals_) {
            cloud_normals->push_back(pcl::Normal(pt.x(), pt.y(), pt.z()));
        }
    }

    pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal,
                                       pcl::PrincipalCurvatures>
            curvature_estimator;
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr cloud_curvatures(
            new pcl::PointCloud<pcl::PrincipalCurvatures>);
    curvature_estimator.setInputCloud(cloud_pcl);
    curvature_estimator.setInputNormals(cloud_normals);
    curvature_estimator.setSearchMethod(tree);
    curvature_estimator.setRadiusSearch(
            static_cast<hymson3d::geometry::KDTreeSearchParamRadius&>(param)
                    .radius_);
    curvature_estimator.compute(*cloud_curvatures);

    // std::cout << cloud_curvatures->size() << " vs: " << cloud.points_.size()
    //           << std::endl;
    cloud.curvatures_.reserve(cloud_curvatures->size());

    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::min();

    for (int i = 0; i < cloud_curvatures->size(); i++) {
        // std::cout << (*cloud_curvatures)[i].pc1 << "  "
        //           << (*cloud_curvatures)[i].pc2 << std::endl;

        geometry::curvature* curvature = new geometry::curvature();
        curvature->mean_curvature =
                ((*cloud_curvatures)[i].pc1 + (*cloud_curvatures)[i].pc2) / 2;
        curvature->gaussian_curvature =
                (*cloud_curvatures)[i].pc1 * (*cloud_curvatures)[i].pc2;
        //  [Wardetzky et al. 2007] total_curva = k1^2 + k2^2
        curvature->total_curvature = pow((*cloud_curvatures)[i].pc1, 2) +
                                     pow((*cloud_curvatures)[i].pc2, 2);
        if (curvature->total_curvature < min_val)
            min_val = curvature->total_curvature;
        if (curvature->total_curvature > max_val)
            max_val = curvature->total_curvature;
        cloud.curvatures_.emplace_back(curvature);
    }

    // color pointcloud according to curvature
    cloud.colors_.reserve(cloud.points_.size());
    for (int i = 0; i < cloud.curvatures_.size(); i++) {
        cloud.colors_.emplace_back(color_with_curvature(
                cloud.curvatures_[i]->total_curvature, min_val, max_val));
    }

    LOG_INFO("Compute curvature done");
}

// TODO::implement https://arxiv.org/pdf/2305.12653

Eigen::Vector3d color_with_curvature(double curvature,
                                     double min_val,
                                     double max_val) {
    // std::cout << curvature << std::endl;
    double value = (curvature - min_val) / (max_val - min_val);
    double r = 1.0, g = 1.0, b = 1.0;

    if (value < 0.5) {
        r = value * 2.0;
        g = value * 2.0;
        b = 1.0;
    } else {
        r = 1.0;
        g = 1.0 - (value - 0.5) * 2.0;
        b = 1.0 - (value - 0.5) * 2.0;
    }

    return Eigen::Vector3d(r, g, b);
}

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
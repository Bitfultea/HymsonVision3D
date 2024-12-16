#include "Curvature.h"

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
// #include <pcl/features/principal_curvatures_omp.h>
#include <pcl/point_types.h>

#include <cmath>

#include "3D/PointCloud.h"
#include "Converter.h"
#include "MathTool.h"
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

    // pcl::search::Search<pcl::PointXYZ>::Ptr tree;
    // // std::cout << cloud_pcl->isOrganized() << std::endl;
    // // std::cout << cloud_pcl->height << std::endl;
    // if (cloud_pcl->isOrganized()) {
    //     tree.reset(new pcl::search::OrganizedNeighbor<pcl::PointXYZ>());
    // } else {
    //     tree.reset(new pcl::search::KdTree<pcl::PointXYZ>(false));
    // }

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
            new pcl::PointCloud<pcl::Normal>);

    if (!cloud.HasNormals()) {
        LOG_INFO("Estimate normal for curvature computation");
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
        normal_estimator.setInputCloud(cloud_pcl);
        normal_estimator.setSearchMethod(tree);
        // normal_estimator.setViewPoint(0.0, 0.0, 100.0);
        normal_estimator.setViewPoint(0, 0, std::numeric_limits<float>::max());
        // std::cout <<
        // static_cast<hymson3d::geometry::KDTreeSearchParamRadius&>(
        //                      param)
        //                      .radius_
        //           << std::endl;
        normal_estimator.setRadiusSearch(
                static_cast<hymson3d::geometry::KDTreeSearchParamRadius&>(param)
                        .radius_);
        normal_estimator.compute(*cloud_normals);
        converter::pcl_to_hymson3d_normals(cloud_normals, cloud);
        LOG_INFO("Complete normal estimation");
        // std::cout << (*cloud_normals)[0] << std::endl;
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

    // std::cout << min_val << " " << max_val << std::endl;
    // color pointcloud according to curvature
    cloud.colors_.reserve(cloud.points_.size());
    for (int i = 0; i < cloud.curvatures_.size(); i++) {
        cloud.colors_.emplace_back(color_with_curvature(
                cloud.curvatures_[i]->total_curvature, min_val, max_val));
    }

    LOG_INFO("Compute curvature done");
}

// TODO::implement https://arxiv.org/pdf/2305.12653

// TODO::test this method and compare with pcl implementation
void ComputeSurfaceVariation(geometry::PointCloud& cloud,
                             geometry::KDTreeSearchParam& param) {
    std::cout << "gg" << std::endl;

    if (!cloud.HasCurvatures()) {
        cloud.curvatures_.resize(cloud.points_.size());
    } else {
        LOG_DEBUG("Curvatures already exist. Overwrite");
    }

    std::vector<Eigen::Matrix3d> covariances;
    if (!cloud.HasCovariances()) {
        const auto& points = cloud.points_;
        std::vector<Eigen::Matrix3d> covariances;
        covariances.resize(points.size());
        hymson3d::geometry::KDTree kdtree;
        kdtree.SetData(cloud);
        std::cout << "not covariance 2" << std::endl;

// use knn for testing
#pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)points.size(); i++) {
            std::vector<int> indices;
            std::vector<double> distance2;
            std::cout << "0" << std::endl;

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
    std::cout << "0" << std::endl;

    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::min();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)covariances.size(); i++) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
        solver.compute(covariances[i]);
        Eigen::VectorXd eigenvalues = solver.eigenvalues();
        std::sort(eigenvalues.begin(), eigenvalues.end());
        // NOTE:
        // https://pointclouds.org/documentation/group__features.html
        // https://graphics.stanford.edu/~mapauly/Pdfs/Simplification.pdf
        // curvature 和 surface variation 不是一个东西
        std::cout << "1" << std::endl;
        cloud.curvatures_[i]->total_curvature =
                eigenvalues[0] /
                (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);
        if (cloud.curvatures_[i]->total_curvature < min_val)
            min_val = cloud.curvatures_[i]->total_curvature;
        if (cloud.curvatures_[i]->total_curvature > max_val)
            max_val = cloud.curvatures_[i]->total_curvature;

        // fake data
        cloud.curvatures_[i]->mean_curvature = eigenvalues[0];
        cloud.curvatures_[i]->gaussian_curvature =
                eigenvalues[0] * eigenvalues[1];
    }

    cloud.colors_.reserve(cloud.points_.size());
    for (int i = 0; i < cloud.curvatures_.size(); i++) {
        cloud.colors_.emplace_back(color_with_curvature(
                cloud.curvatures_[i]->total_curvature, min_val, max_val));
    }
}

Eigen::Vector3d color_with_curvature(double curvature,
                                     double min_val,
                                     double max_val) {
    // std::cout << curvature << std::endl;
    double value = (curvature - min_val) / (max_val - min_val);
    // value = 1 / (-log(value));
    double r = 1.0, g = 1.0, b = 1.0;

    if (value < 0.5) {
        r = value * 2.0;
        g = value * 2.0;
        b = 1.0;
    } else {
        r = 1.0;
        g = 1.0 - (value - 0.5) * 2.0;
        b = 1.0 - (value - 0.5) * 2.0;
        // std::cout << "value: " << value << " r: " << r << " g: " << g
        //           << " b: " << b << std::endl;
    }

    return Eigen::Vector3d(r, g, b);
}

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
#include "Normal.h"

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_types.h>
#include <pcl/search/organized.h>

#include <cmath>

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
    const size_t N = cloud.points_.size();
    if (N == 0) return;

    bool has_cov = cloud.HasCovariances();
    if (!has_cov) {
        cloud.covariances_.resize(N);
    }

    cloud.normals_.resize(N);
    cloud.curvatures_.resize(N);

    std::unique_ptr<geometry::KDTree> kdtree_ptr;
    if (!has_cov) {
        kdtree_ptr = std::make_unique<geometry::KDTree>();
        kdtree_ptr->SetData(cloud);
    }

#pragma omp parallel
    {
        // Thread Local Storage
        // avoid repeated memory-allocation
        std::vector<int> indices;
        std::vector<double> dists;
        indices.reserve(100);  // estimated k value
        dists.reserve(100);

#pragma omp for schedule(guided)
        for (int i = 0; i < (int)N; i++) {
            Eigen::Matrix3d covariance;
            bool valid_covariance = false;

            // --- stage 1: compute covariance ---
            if (has_cov) {
                covariance = cloud.covariances_[i];
                valid_covariance = true;
            } else {
                indices.clear();
                dists.clear();
                if (kdtree_ptr->Search(cloud.points_[i], param, indices,
                                       dists) >= 3) {
                    covariance =
                            utility::ComputeCovariance(cloud.points_, indices);
                    cloud.covariances_[i] = covariance;  // 存回结果
                    valid_covariance = true;
                } else {
                    cloud.covariances_[i] = Eigen::Matrix3d::Identity();
                }
            }

            // --- stage 2. eigen decomposition && compute normal vector ---
            // 直接在寄存器/L1缓存中处理 covariance
            geometry::curvature curv_ptr;  //

            if (valid_covariance) {
                auto result = utility::mathtool::FastEigen3x3(covariance);
                cloud.normals_[i] = std::get<0>(result).normalized();

                const auto& evals = std::get<1>(result);

                // remove pow(2)
                double e1 = evals[1];
                double e2 = evals[2];

                curv_ptr.gaussian_curvature = e1 * e2;
                curv_ptr.mean_curvature = 0.5 * (e1 + e2);
                curv_ptr.total_curvature = e1 * e1 + e2 * e2;
            } else {
                cloud.normals_[i].setZero();
                curv_ptr.gaussian_curvature = 0;
                curv_ptr.mean_curvature = 0;
                curv_ptr.total_curvature = 0;
            }
            cloud.curvatures_[i] = curv_ptr;
        }
    }

    // ----------------------------------------------------------------------

    //         std::vector<Eigen::Matrix3d> covariances;
    //         const bool has_covariance = cloud.HasCovariances();
    //         cloud.covariances_.resize(cloud.points_.size());
    //         if (!has_covariance) {
    //             const auto& points = cloud.points_;
    //             covariances.resize(points.size());
    //             hymson3d::geometry::KDTree kdtree;
    //             kdtree.SetData(cloud);

    // #pragma omp parallel for schedule(static)
    //             for (int i = 0; i < (int)points.size(); i++) {
    //                 std::vector<int> indices;
    //                 std::vector<double> distance2;
    //                 if (kdtree.Search(points[i], param, indices,
    //                 distance2) >= 3) {
    //                     // std::cout << indices.size() << std::endl;
    //                     auto covariance =
    //                             utility::ComputeCovariance(points,
    //                             indices);
    //                     if (has_covariance &&
    //                     covariance.isIdentity(1e-4)) {
    //                         covariances[i] = cloud.covariances_[i];
    //                     } else {
    //                         covariances[i] = covariance;
    //                     }
    //                 } else {
    //                     covariances[i] = Eigen::Matrix3d::Identity();
    //                 }
    //             }
    //         } else {
    //             covariances = cloud.covariances_;
    //         }

    //         bool has_normals = cloud.HasNormals();
    //         if (!has_normals) {
    //             cloud.normals_.resize(cloud.points_.size());
    //         }

    //         cloud.normals_.resize(cloud.points_.size());
    //         cloud.curvatures_.resize(cloud.points_.size());
    //         // #pragma omp parallel for schedule(static)
    //         for (int i = 0; i < (int)covariances.size(); i++) {
    //             auto result =
    //             utility::mathtool::FastEigen3x3(covariances[i]);
    //             Eigen::Vector3d normal =
    //             std::get<0>(result).normalized(); std::vector<double>
    //             evals = std::get<1>(result);

    //             // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>
    //             // solver(covariances[i]); Eigen::Vector3d
    //             eigenvalues =
    //             // solver.eigenvalues(); Eigen::Matrix3d eigenvectors
    //             =
    //             // solver.eigenvectors();

    //             // for (int j = 0; j < 3; j++) {
    //             //     std::cout << evals[j] << ";" << eigenvalues(j)
    //             << std::endl;
    //             // }
    //             // std::cout << std::endl;

    //             geometry::curvature* curvature = new
    //             geometry::curvature(); curvature->gaussian_curvature
    //             = evals[1] * evals[2]; curvature->mean_curvature =
    //             0.5 * (evals[1] + evals[2]);
    //             curvature->total_curvature = pow(evals[1], 2) +
    //             pow(evals[2], 2);
    //             // viewpoint/tangentplane/givendirection
    //             cloud.normals_[i] = normal;
    //             cloud.covariances_[i] = covariances[i];
    //             cloud.curvatures_[i] = curvature;
    //         }
}

void ComputeRotateNormals_PCA_Fast(geometry::PointCloud& cloud,
                                   const geometry::KDTreeSearchParam& param) {
    const size_t N = cloud.points_.size();
    const bool has_cov = cloud.HasCovariances();
    const bool has_norm = cloud.HasNormals();

    // 1) Ԥ���������������
    cloud.covariances_.resize(N);
    cloud.normals_.resize(N);
    cloud.curvatures_.resize(N);

    // 2) ֻ����һ��ȫ�� KDTree
    hymson3d::geometry::KDTree global_kdtree;
    if (!has_cov) {
        global_kdtree.SetData(cloud);
    }

    // 3) ���м���Э�������
    // global_kdtree��
    std::vector<Eigen::Matrix3d> covs(N);
#pragma omp parallel for schedule(dynamic, 128)
    for (int i = 0; i < static_cast<int>(N); i++) {
        if (!has_cov) {
            // ÿ���߳�����ʱ���� indices/d2���Զ�����
            std::vector<int> indices;
            std::vector<double> d2;
            int cnt =
                    global_kdtree.Search(cloud.points_[i], param, indices, d2);
            if (cnt >= 3) {
                covs[i] = utility::ComputeCovariance(cloud.points_, indices);
                if (covs[i].isIdentity(1e-4)) {
                    covs[i] = cloud.covariances_[i];
                }
            } else {
                covs[i].setIdentity();
            }
        } else {
            covs[i] = cloud.covariances_[i];
        }
    }

    // std::vector<geometry::curvature> curvs(N);
    const Eigen::Vector3d orientation_reference =
            Eigen::Vector3d(0.0, 0.0, 1.0);
    // 4) ���м��㷨�ߺ����ʣ�ֱ�� new curvature ����ԭ�ӿ�
#pragma omp parallel for schedule(dynamic, 128)
    for (int i = 0; i < static_cast<int>(N); i++) {
        // FastEigen3x3 ���� (����������, [��0,��1,��2])
        auto [dir, evals] = utility::mathtool::FastEigen3x3(covs[i]);
        Eigen::Vector3d n = dir.normalized();

        geometry::curvature curvature;
        curvature.gaussian_curvature = evals[1] * evals[2];
        curvature.mean_curvature = 0.5 * (evals[1] + evals[2]);
        curvature.total_curvature = pow(evals[1], 2) + pow(evals[2], 2);
        // curvs[i].gaussian_curvature = evals[1] * evals[2];
        // curvs[i].mean_curvature = 0.5 * (evals[1] + evals[2]);
        // curvs[i].total_curvature = evals[1] * evals[1] + evals[2] *
        // evals[2];
        if (n.norm() == 0.0) {
            n = orientation_reference;
        } else if (n.dot(orientation_reference) < 0.0) {
            n *= -1.0;  // flip the normal
        }
        cloud.normals_[i] = n;
        cloud.covariances_[i] = covs[i];
        cloud.curvatures_[i] = curvature;
    }
}

// TODO:oritent the normal w.r.t
void orient_normals_towards_positive_z(geometry::PointCloud& cloud) {
    const Eigen::Vector3d orientation_reference =
            Eigen::Vector3d(0.0, 0.0, 1.0);
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

void normal_aggregation_x(geometry::PointCloud& cloud,
                          geometry::PointCloud::Ptr target_cloud,
                          float ratio) {
    if (!cloud.HasNormals()) {
        LOG_ERROR(
                "No normals in the PointCloud. Can not aggregate "
                "normals");
    }
    target_cloud->points_.resize(cloud.points_.size());
    target_cloud->normals_ = cloud.normals_;
#pragma omp parallel for
    for (int i = 0; i < cloud.points_.size(); i++) {
        Eigen::Vector3d point = {
                cloud.points_[i].x() + ratio * cloud.normals_[i].x(),
                cloud.points_[i].y(), cloud.points_[i].z()};
        target_cloud->points_[i] = point;
    }
}

void normal_aggregation_y(geometry::PointCloud& cloud,
                          geometry::PointCloud::Ptr target_cloud,
                          float ratio) {
    if (!cloud.HasNormals()) {
        LOG_ERROR(
                "No normals in the PointCloud. Can not aggregate "
                "normals");
    }
    target_cloud->points_.resize(cloud.points_.size());
    target_cloud->normals_ = cloud.normals_;
#pragma omp parallel for
    for (int i = 0; i < cloud.points_.size(); i++) {
        Eigen::Vector3d point = {
                cloud.points_[i].x(),
                cloud.points_[i].y() + ratio * cloud.normals_[i].y(),
                cloud.points_[i].z()};
        target_cloud->points_[i] = point;
    }
}

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
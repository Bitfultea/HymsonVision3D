#include "DefectDetection.h"

#include <pcl/console/print.h>

#include "3D/Mesh.h"
#include "Cluster.h"
#include "Converter.h"
#include "Curvature.h"
#include "Filter.h"

namespace hymson3d {
namespace pipeline {

void DefectDetection::detect_defects(
        std::shared_ptr<geometry::PointCloud> cloud,
        geometry::KDTreeSearchParamRadius param,
        float long_normal_degree,
        float long_curvature_threshold,
        float rcorner_normal_degree,
        float rcorner_curvature_threshold,
        float height_threshold,
        float radius,
        size_t min_points,
        bool debug_mode) {
    if (debug_mode) {
        utility::write_ply("test_curvature.ply", cloud,
                           utility::FileFormat::BINARY);
    }

    // 1.0 keep egde points and r-corner points
    std::shared_ptr<geometry::PointCloud> points =
            std::make_shared<geometry::PointCloud>();

    for (size_t i = 0; i < cloud->points_.size(); i++) {
        if (cloud->points_[i].z() > height_threshold) {
            points->points_.emplace_back(cloud->points_[i]);
        }
    }

    // 1.1 denoise
    LOG_INFO("Denoise: {} before denoised.", points->points_.size());
    core::Filter filter;
    auto filter_res = filter.StatisticalOutliers(points, 250, 2.0);
    points = std::get<0>(filter_res);
    LOG_INFO("Denoise: {} After denoised.", points->points_.size());

    if (debug_mode) {
        utility::write_ply("test_curvature_1.ply", points,
                           utility::FileFormat::BINARY);
    }

    // 1.2 normal estimation
    core::feature::ComputeNormals_PCA(*points, param);
    core::feature::orient_normals_towards_positive_z(*points);

    // 1.2 cluster
    int num_clusters =
            core::Cluster::DBSCANCluster(*points, radius, min_points);
    std::vector<geometry::PointCloud::Ptr> clusters;
    for (int i = 0; i < num_clusters; i++) {
        auto pcd = std::make_shared<geometry::PointCloud>();
        clusters.emplace_back(pcd);
    }
    for (size_t i = 0; i < points->points_.size(); i++) {
        if (points->labels_[i] >= 0) {
            clusters[points->labels_[i]]->points_.emplace_back(
                    points->points_[i]);
            clusters[points->labels_[i]]->normals_.emplace_back(
                    points->normals_[i]);
        }
    }
    if (debug_mode) {
        utility::write_ply("test_curvature_2.ply", points,
                           utility::FileFormat::BINARY);
    }

    // 2.0 separate r-corner and the flip-edge
    std::vector<geometry::PointCloud::Ptr> long_clouds;
    std::vector<geometry::PointCloud::Ptr> corners_clouds;
    for (int i = 0; i < num_clusters; i++) {
        std::cout << "cluster " << i << ": " << clusters[i]->points_.size()
                  << std::endl;
        if (clusters[i]->points_.size() > 100000) {
            double y_min = clusters[i]->GetMinBound().y();
            double y_max = clusters[i]->GetMaxBound().y();

            std::shared_ptr<geometry::PointCloud> corner_left_cloud =
                    std::make_shared<geometry::PointCloud>();
            std::shared_ptr<geometry::PointCloud> long_cloud =
                    std::make_shared<geometry::PointCloud>();
            std::shared_ptr<geometry::PointCloud> corner_right_cloud =
                    std::make_shared<geometry::PointCloud>();
            for (size_t j = 0; j < clusters[i]->points_.size(); j++) {
                if (clusters[i]->points_[j].y() < y_min + 3.0) {
                    corner_left_cloud->points_.emplace_back(
                            clusters[i]->points_[j]);
                    corner_left_cloud->normals_.emplace_back(
                            clusters[i]->normals_[j]);
                    corner_left_cloud->labels_.emplace_back(0);
                } else if (clusters[i]->points_[j].y() >= y_min + 3.0 &&
                           clusters[i]->points_[j].y() <= y_max - 3.0) {
                    long_cloud->points_.emplace_back(clusters[i]->points_[j]);
                    long_cloud->normals_.emplace_back(clusters[i]->normals_[j]);
                    long_cloud->labels_.emplace_back(0);
                } else {
                    corner_right_cloud->points_.emplace_back(
                            clusters[i]->points_[j]);
                    corner_right_cloud->normals_.emplace_back(
                            clusters[i]->normals_[j]);
                    corner_right_cloud->labels_.emplace_back(0);
                }
            }

            corners_clouds.emplace_back(corner_left_cloud);
            corners_clouds.emplace_back(corner_right_cloud);
            long_clouds.emplace_back(long_cloud);
        }
    }

    geometry::PointCloud::Ptr total_cloud =
            std::make_shared<geometry::PointCloud>();
    for (auto cloud : corners_clouds) {
        cloud->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));
        (*total_cloud) += *cloud;
    }
    for (auto cloud : long_clouds) {
        cloud->PaintUniformColor(Eigen::Vector3d(0.0, 1.0, 0.0));
        (*total_cloud) += *cloud;
    }
    if (debug_mode) {
        utility::write_ply("test_curvature_3.ply", total_cloud,
                           utility::FileFormat::BINARY);
    }

    // 2.1 region growing on long clouds
    size_t max_cluster_size = 2000000;
    size_t min_cluster_size = 100;
    for (int i = 0; i < long_clouds.size(); i++) {
        long_clouds[i]->PaintUniformColor(Eigen::Vector3d(0.0, 0.0, 0.0));
        core::Cluster::RegionGrowing_PCL(
                *long_clouds[i], long_normal_degree, long_curvature_threshold,
                min_cluster_size, max_cluster_size, 100);
    }

    // 2.2 region growing on corner clouds
    max_cluster_size = 2000000;
    min_cluster_size = 100;
    for (int i = 0; i < corners_clouds.size(); i++) {
        corners_clouds[i]->PaintUniformColor(Eigen::Vector3d(0.0, 0.0, 0.0));
        core::Cluster::RegionGrowing_PCL(
                *corners_clouds[i], rcorner_normal_degree,
                rcorner_curvature_threshold, min_cluster_size, max_cluster_size,
                100);
    }

    // 2.3 merge clouds
    total_cloud->Clear();
    for (auto cloud : corners_clouds) {
        (*total_cloud) += *cloud;
    }
    for (auto cloud : long_clouds) {
        (*total_cloud) += *cloud;
    }
    if (debug_mode) {
        utility::write_ply("test_curvature_4.ply", total_cloud,
                           utility::FileFormat::BINARY);
    }

    // 3.0 extract defects
    geometry::PointCloud::Ptr defects =
            std::make_shared<geometry::PointCloud>();
    for (int i = 0; i < long_clouds.size(); i++) {
        for (size_t j = 0; j < long_clouds[i]->points_.size(); j++) {
            if (long_clouds[i]->labels_[j] == 0) {
                defects->points_.emplace_back(long_clouds[i]->points_[j]);
                defects->normals_.emplace_back(long_clouds[i]->normals_[j]);
            }
        }
    }
    if (debug_mode) {
        utility::write_ply("test_curvature_5.ply", defects,
                           utility::FileFormat::BINARY);
    }

    // 3.1 defects clusters
    int num_defects =
            core::Cluster::DBSCANCluster(*defects, radius, min_points);
    std::vector<geometry::PointCloud::Ptr> defect_clouds(num_defects);
    if (debug_mode) {
        utility::write_ply("test_curvature_6.ply", defects,
                           utility::FileFormat::BINARY);
    }

    for (auto &pcd : defect_clouds) {
        pcd = std::make_shared<geometry::PointCloud>();
    }
    for (size_t i = 0; i < defects->points_.size(); i++) {
        if (defects->labels_[i] >= 0) {
            defect_clouds[defects->labels_[i]]->points_.emplace_back(
                    defects->points_[i]);
            defect_clouds[defects->labels_[i]]->normals_.emplace_back(
                    defects->normals_[i]);
        }
    }
    std::cout << "Defects: " << defect_clouds.size() << std::endl;
    std::vector<geometry::PointCloud::Ptr> res;
    for (auto cloud : defect_clouds) {
        std::cout << "cloud size: " << cloud->points_.size() << std::endl;
        if (cloud->points_.size() >= 500) {
            res.emplace_back(cloud);
        }
    }
    defects->Clear();

    // 3.2 defects report
    LOG_INFO("Detect {} defects.", res.size());
    for (int i = 0; i < res.size(); i++) {
        LOG_INFO("Defect {} has {} points.", i, res[i]->points_.size());
        Eigen::Vector3d min_bound = res[i]->GetMinBound();
        Eigen::Vector3d max_bound = res[i]->GetMaxBound();
        Eigen::Vector3d extent = max_bound - min_bound;
        LOG_INFO("Defect size is {:.3f}x{:.3f}", extent.x(), extent.y());
    }
}

void DefectDetection::detect_pinholes(
        std::shared_ptr<geometry::PointCloud> cloud,
        geometry::KDTreeSearchParamRadius param,
        float height_threshold,
        float radius,
        size_t min_points,
        Eigen::Vector3d transformation_matrix,
        bool denoise,
        bool debug_mode) {
    // 1.0 keep egde points and r-corner points
    std::shared_ptr<geometry::PointCloud> points =
            std::make_shared<geometry::PointCloud>();

    for (size_t i = 0; i < cloud->points_.size(); i++) {
        if (cloud->points_[i].z() > height_threshold) {
            points->points_.emplace_back(cloud->points_[i]);
        }
    }

    // 1.1 denoise
    if (denoise) {
        LOG_INFO("Denoise: {} before denoised.", points->points_.size());
        core::Filter filter;
        auto filter_res = filter.StatisticalOutliers(points, 250, 2.0);
        points = std::get<0>(filter_res);
        LOG_INFO("Denoise: {} After denoised.", points->points_.size());
    }

    // 1.2 normal estimation
    core::feature::ComputeNormals_PCA(*points, param);
    core::feature::orient_normals_towards_positive_z(*points);

    // 1.2 cluster
    int num_clusters =
            core::Cluster::DBSCANCluster(*points, radius, min_points);
    std::vector<geometry::PointCloud::Ptr> clusters;
    for (int i = 0; i < num_clusters; i++) {
        auto pcd = std::make_shared<geometry::PointCloud>();
        clusters.emplace_back(pcd);
    }
    for (size_t i = 0; i < points->points_.size(); i++) {
        if (points->labels_[i] >= 0) {
            clusters[points->labels_[i]]->points_.emplace_back(
                    points->points_[i]);
            clusters[points->labels_[i]]->normals_.emplace_back(
                    points->normals_[i]);
        }
    }
    if (debug_mode) {
        utility::write_ply("pinhole_0.ply", points,
                           utility::FileFormat::BINARY);
        geometry::HymsonMesh mesh;
        mesh.construct_mesh(points);
    }

    // 1.3 separate r-corner and the flip-edge
    std::vector<geometry::PointCloud::Ptr> long_clouds;
    std::vector<geometry::PointCloud::Ptr> corners_clouds;
    for (int i = 0; i < num_clusters; i++) {
        std::cout << "cluster " << i << ": " << clusters[i]->points_.size()
                  << std::endl;
        if (clusters[i]->points_.size() > 100000) {
            double y_min = clusters[i]->GetMinBound().y();
            double y_max = clusters[i]->GetMaxBound().y();

            std::shared_ptr<geometry::PointCloud> corner_left_cloud =
                    std::make_shared<geometry::PointCloud>();
            std::shared_ptr<geometry::PointCloud> long_cloud =
                    std::make_shared<geometry::PointCloud>();
            std::shared_ptr<geometry::PointCloud> corner_right_cloud =
                    std::make_shared<geometry::PointCloud>();
            for (size_t j = 0; j < clusters[i]->points_.size(); j++) {
                if (clusters[i]->points_[j].y() < y_min + 3.0) {
                    corner_left_cloud->points_.emplace_back(
                            clusters[i]->points_[j]);
                    corner_left_cloud->normals_.emplace_back(
                            clusters[i]->normals_[j]);
                    corner_left_cloud->labels_.emplace_back(0);
                } else if (clusters[i]->points_[j].y() >= y_min + 3.0 &&
                           clusters[i]->points_[j].y() <= y_max - 3.0) {
                    long_cloud->points_.emplace_back(clusters[i]->points_[j]);
                    long_cloud->normals_.emplace_back(clusters[i]->normals_[j]);
                    long_cloud->labels_.emplace_back(0);
                } else {
                    corner_right_cloud->points_.emplace_back(
                            clusters[i]->points_[j]);
                    corner_right_cloud->normals_.emplace_back(
                            clusters[i]->normals_[j]);
                    corner_right_cloud->labels_.emplace_back(0);
                }
            }

            corners_clouds.emplace_back(corner_left_cloud);
            corners_clouds.emplace_back(corner_right_cloud);
            long_clouds.emplace_back(long_cloud);
        }
    }

    // 2.0 slicing along y-axi direction
    for (int i = 0; i < long_clouds.size(); i++) {
        Eigen::Vector3d min_bound = long_clouds[i]->GetMinBound();
        Eigen::Vector3d max_bound = long_clouds[i]->GetMaxBound();
        int num_slice = (int)(((max_bound.y() - min_bound.y()) /
                               transformation_matrix.y()) +
                              1);
        long_clouds[i]->y_slices_.resize(num_slice);
        long_clouds[i]->ny_slices_.resize(num_slice);
        long_clouds[i]->y_slice_idxs.resize(num_slice);
        for (size_t j = 0; j < long_clouds[i]->points_.size(); j++) {
            auto pt = long_clouds[i]->points_[j];
            auto n = long_clouds[i]->normals_[j];
            int slice_idx =
                    (int)((pt.y() - min_bound.y()) / transformation_matrix.y());
            if (slice_idx == 0) std::cout << pt.y() << std::endl;
            long_clouds[i]->y_slices_[slice_idx].emplace_back(
                    Eigen::Vector2d(pt.x(), pt.z()));
            long_clouds[i]->ny_slices_[slice_idx].emplace_back(
                    Eigen::Vector3d(n.x(), n.y(), n.z()));
            long_clouds[i]->y_slice_idxs[slice_idx].emplace_back(j);
        }
    }

    // 2.1 derivative along x-axi direction
    core::Filter filter;
    for (int i = 0; i < long_clouds.size(); i++) {
        std::vector<size_t> defect_pt_idxs;
        for (int j = 0; j < long_clouds[i]->y_slices_.size(); j++) {
            std::vector<size_t> local_index = long_clouds[i]->y_slice_idxs[j];
            std::vector<double> derivative;
            std::vector<size_t> pt_idxs;
            process_y_slice(long_clouds[i]->y_slices_[j],
                            long_clouds[i]->ny_slices_[j], derivative, pt_idxs);
            for (auto idx : pt_idxs) {
                defect_pt_idxs.emplace_back(local_index[idx]);
            }
        }
        auto tmp = filter.IndexDownSample(long_clouds[i], defect_pt_idxs);
        utility::write_ply("pinhole_g" + std::to_string(i) + ".ply", tmp,
                           utility::FileFormat::BINARY);
    }
}

void DefectDetection::process_y_slice(std::vector<Eigen::Vector2d> &y_slice,
                                      std::vector<Eigen::Vector3d> &ny_slice,
                                      std::vector<double> &y_derivative,
                                      std::vector<size_t> &local_idxs) {
    for (int i = 0; i < y_slice.size(); i++) {
        Eigen::Vector2d diff;
        if (i == 0) {
            diff = y_slice[1] - y_slice[0];
            y_derivative.emplace_back(diff.y() / diff.x());
        } else if (i == y_slice.size() - 1) {
            diff = y_slice[i] - y_slice[i - 1];
            y_derivative.emplace_back(diff.y() / diff.x());
        } else {
            diff = y_slice[i + 1] - y_slice[i - 1];
            y_derivative.emplace_back(diff.y() / diff.x());
        }
    }
    std::vector<double> second_derivative;
    // skip the border pts
    if (y_derivative.size() < 3) return;
    for (int i = 1; i < y_derivative.size() - 1; i++) {
        double diff = y_derivative[i] - y_derivative[i - 1];
        second_derivative.emplace_back(diff);
        if (diff > 0.0) {
            local_idxs.emplace_back(i);
        }
    }
}
}  // namespace pipeline
}  // namespace hymson3d
#include "DefectDetection.h"

#include <math.h>
#include <pcl/console/print.h>

#include "3D/Mesh.h"
#include "Cluster.h"
#include "Converter.h"
#include "Curvature.h"
#include "Feature.h"
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
        size_t min_defects_size,
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
            points->intensities_.emplace_back(cloud->intensities_[i]);
        }
    }

    // 1.1 denoise
    LOG_INFO("Denoise: {} before denoised.", points->points_.size());
    core::Filter filter;
    auto filter_res = filter.StatisticalOutliers(points, 250, 2.0);
    points = std::get<0>(filter_res);
    LOG_INFO("Denoise: {} After denoised.", points->points_.size());

    // 1.2 normal estimation
    core::feature::ComputeNormals_PCA(*points, param);
    core::feature::orient_normals_towards_positive_z(*points);

    if (debug_mode) {
        utility::write_ply("test_curvature_1.ply", points,
                           utility::FileFormat::BINARY);
    }

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
            clusters[points->labels_[i]]->intensities_.emplace_back(
                    points->intensities_[i]);
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
                    corner_left_cloud->intensities_.emplace_back(
                            clusters[i]->intensities_[j]);
                } else if (clusters[i]->points_[j].y() >= y_min + 3.0 &&
                           clusters[i]->points_[j].y() <= y_max - 3.0) {
                    long_cloud->points_.emplace_back(clusters[i]->points_[j]);
                    long_cloud->normals_.emplace_back(clusters[i]->normals_[j]);
                    long_cloud->labels_.emplace_back(0);
                    long_cloud->intensities_.emplace_back(
                            clusters[i]->intensities_[j]);
                } else {
                    corner_right_cloud->points_.emplace_back(
                            clusters[i]->points_[j]);
                    corner_right_cloud->normals_.emplace_back(
                            clusters[i]->normals_[j]);
                    corner_right_cloud->labels_.emplace_back(0);
                    corner_right_cloud->intensities_.emplace_back(
                            clusters[i]->intensities_[j]);
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
        // core::Cluster::RegionGrowing_PCL(
        //         *long_clouds[i], long_normal_degree,
        //         long_curvature_threshold, min_cluster_size, max_cluster_size,
        //         100);
        core::Cluster::RegionGrowingCluster(
                *long_clouds[i], param.radius_, long_normal_degree,
                long_curvature_threshold, min_cluster_size);
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
    // for (auto cloud : corners_clouds) {
    //     (*total_cloud) += *cloud;
    // }
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
            if (long_clouds[i]->labels_[j] == -1) {
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
    std::vector<geometry::PointCloud::Ptr> res;
    for (auto cloud : defect_clouds) {
        if (cloud->points_.size() >= min_defects_size) {
            res.emplace_back(cloud);
        }
    }
    defects->Clear();

    // 3.2 defects report
    LOG_INFO("Detect {} defects.", res.size());
    geometry::PointCloud::Ptr show_cloud =
            std::make_shared<geometry::PointCloud>();
    for (int i = 0; i < res.size(); i++) {
        LOG_INFO("Defect {} has {} points.", i, res[i]->points_.size());
        Eigen::Vector3d min_bound = res[i]->GetMinBound();
        Eigen::Vector3d max_bound = res[i]->GetMaxBound();
        Eigen::Vector3d extent = max_bound - min_bound;
        *(show_cloud) += *(res[i]);
        LOG_INFO("Defect size is {:.3f}x{:.3f}", extent.x(), extent.y());
    }

    if (debug_mode) {
        show_cloud->PaintUniformColor(Eigen::Vector3d(1.0, 0.0, 0.0));
        utility::write_ply("test_curvature_7.ply", show_cloud,
                           utility::FileFormat::BINARY);
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

void DefectDetection::detect_pinholes_nva(
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
    height_filter(cloud, points, height_threshold);

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
    part_separation(points, clusters, num_clusters);

    // if (debug_mode) {
    //     utility::write_ply("pinhole_0.ply", points,
    //                        utility::FileFormat::BINARY);
    //     geometry::HymsonMesh mesh;
    //     mesh.construct_mesh(points);
    // }

    // 1.3 separate r-corner and the flip-edge
    std::vector<geometry::PointCloud::Ptr> long_clouds;
    std::vector<geometry::PointCloud::Ptr> corners_clouds;
    extract_long_edge(long_clouds, corners_clouds, clusters, num_clusters);

    // 2.0 nva method
    for (int i = 0; i < long_clouds.size(); i++) {
        long_clouds[i]->PaintUniformColor(Eigen::Vector3d(1.0, 1.0, 1.0));
        float ratio_x = 0.25;
        float ratio_y = 0.5;
        double dist_x = 0.0001;
        double dist_y = 0.0001;
        bool use_fpfh = true;
        FPFH_NVA(long_clouds[i], ratio_x, ratio_y, dist_x, dist_y, use_fpfh);
        utility::write_ply("pinhole_nva" + std::to_string(i) + ".ply",
                           long_clouds[i], utility::FileFormat::BINARY);
    }
}

void DefectDetection::FPFH_NVA(std::shared_ptr<geometry::PointCloud> cloud,
                               float ratio_x,
                               float ratio_y,
                               double dist_x,
                               double dist_y,
                               bool use_fpfh) {
    LOG_DEBUG("Start FPFH NVA Method");
    std::vector<int> fpfh_marker(cloud->points_.size(), 0);
    if (use_fpfh) {
        Eigen::MatrixXf fpfh_data = core::feature::compute_fpfh(*cloud);
#pragma omp parallel for
        for (size_t i = 0; i < fpfh_data.rows(); i++) {
            if (std::min(abs(fpfh_data(i, 27) - fpfh_data(i, 26)),
                         abs(fpfh_data(i, 28) - fpfh_data(i, 27))) < 60) {
                fpfh_marker[i] = 1;
            }
        }
    }

    // 2. normal vector aggregation
    geometry::PointCloud::Ptr cloud_y =
            std::make_shared<geometry::PointCloud>();
    core::feature::normal_aggregation_y(*cloud, cloud_y, ratio_y);
    geometry::KDTreeSearchParamKNN param(2);  // search the nearest neighbor
    hymson3d::geometry::KDTree kdtree_y;
    kdtree_y.SetData(*cloud_y);
    std::vector<int> marker_y(cloud_y->points_.size(), 0);
#pragma omp parallel for
    for (size_t i = 0; i < cloud_y->points_.size(); i++) {
        std::vector<int> neightbor_idx;
        std::vector<double> dis_y;
        kdtree_y.Search(cloud_y->points_[i], param, neightbor_idx, dis_y);
        if (dis_y[1] < dist_y) {
            marker_y[i] = 1;
        }
    }

    geometry::PointCloud::Ptr cloud_x =
            std::make_shared<geometry::PointCloud>();
    core::feature::normal_aggregation_x(*cloud, cloud_x, ratio_x);
    hymson3d::geometry::KDTree kdtree_x;
    kdtree_x.SetData(*cloud_x);
    std::vector<int> marker_x(cloud_x->points_.size(), 0);

#pragma omp parallel for
    for (size_t i = 0; i < cloud_x->points_.size(); i++) {
        std::vector<int> neightbor_idx;
        std::vector<double> dis_x;
        kdtree_x.Search(cloud_x->points_[i], param, neightbor_idx, dis_x);
        if (dis_x[1] < dist_x) {
            marker_x[i] = 1;
        }
    }

    if (use_fpfh) {
#pragma omp parallel for
        for (int i = 0; i < cloud->points_.size(); i++) {
            // if (fpfh_marker[i] == 1 && (marker_x[i] == 1 || marker_y[i] ==
            // 1)) {
            if (fpfh_marker[i] == 1) {
                cloud->colors_[i] = Eigen::Vector3d(1.0, 0.0, 0.0);
            }
            if (marker_x[i] == 1 || marker_y[i] == 1) {
                cloud->colors_[i] = Eigen::Vector3d(0.0, 1.0, 0.0);
            }
        }
    } else {
#pragma omp parallel for
        for (int i = 0; i < cloud->points_.size(); i++) {
            if (marker_x[i] == 1 || marker_y[i] == 1) {
                cloud->colors_[i] = Eigen::Vector3d(1.0, 0.0, 0.0);
            }
        }
    }
    LOG_DEBUG("Complete FPFH NVA Method");
}

void DefectDetection::height_filter(
        std::shared_ptr<geometry::PointCloud> cloud,
        std::shared_ptr<geometry::PointCloud> points,
        float height_threshold) {
    if (cloud->HasIntensities()) {
        for (size_t i = 0; i < cloud->points_.size(); i++) {
            if (cloud->points_[i].z() > height_threshold) {
                points->points_.emplace_back(cloud->points_[i]);
                points->intensities_.emplace_back(cloud->intensities_[i]);
            }
        }
    } else {
        for (size_t i = 0; i < cloud->points_.size(); i++) {
            if (cloud->points_[i].z() > height_threshold) {
                points->points_.emplace_back(cloud->points_[i]);
            }
        }
    }
}

void DefectDetection::part_separation(
        std::shared_ptr<geometry::PointCloud> cloud,
        std::vector<geometry::PointCloud::Ptr> &clusters,
        int num_clusters) {
    for (int i = 0; i < num_clusters; i++) {
        auto pcd = std::make_shared<geometry::PointCloud>();
        clusters.emplace_back(pcd);
    }
    for (size_t i = 0; i < cloud->points_.size(); i++) {
        if (cloud->labels_[i] >= 0) {
            clusters[cloud->labels_[i]]->points_.emplace_back(
                    cloud->points_[i]);
            clusters[cloud->labels_[i]]->normals_.emplace_back(
                    cloud->normals_[i]);
        }
    }
}

void DefectDetection::extract_long_edge(
        std::vector<geometry::PointCloud::Ptr> &long_clouds,
        std::vector<geometry::PointCloud::Ptr> &corners_clouds,
        std::vector<geometry::PointCloud::Ptr> &clusters,
        int num_clusters) {
    LOG_DEBUG("Start extracttion long edge");
    for (int i = 0; i < num_clusters; i++) {
        LOG_DEBUG("cluster {}: {} points", i, clusters[i]->points_.size());
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
    LOG_DEBUG("Complete extracttion long edge");
}

void DefectDetection::slice_along_y(
        std::vector<geometry::PointCloud::Ptr> &long_clouds,
        Eigen::Vector3d transformation_matrix) {
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
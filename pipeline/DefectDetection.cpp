#include "DefectDetection.h"

#include <math.h>
#include <pcl/console/print.h>

#include "3D/Mesh.h"
#include "Cluster.h"
#include "Converter.h"
#include "Curvature.h"
#include "Feature.h"
#include "Filter.h"
#include "MathTool.h"
#include "PlaneDetection.h"

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
        float ratio_x,
        float ratio_y,
        double dist_x,
        double dist_y,
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

    // 1.3 separate r-corner and the flip-edge
    std::vector<geometry::PointCloud::Ptr> long_clouds;
    std::vector<geometry::PointCloud::Ptr> corners_clouds;
    extract_long_edge(long_clouds, corners_clouds, clusters, num_clusters);

    // 2.0 nva method
    std::vector<geometry::PointCloud::Ptr> defect_clouds;
    defect_clouds.resize(long_clouds.size());
    for (int i = 0; i < long_clouds.size(); i++) {
        long_clouds[i]->PaintUniformColor(Eigen::Vector3d(1.0, 1.0, 1.0));
        bool use_fpfh = false;
        geometry::PointCloud::Ptr defect_cloud;
        defect_cloud = FPFH_NVA(long_clouds[i], ratio_x, ratio_y, dist_x,
                                dist_y, use_fpfh);
        if (defect_cloud->HasPoints() && defect_cloud->points_.size() > 20) {
            defect_clouds[i] = defect_cloud;
        }
        if (debug_mode)
            utility::write_ply("pinhole_nva" + std::to_string(i) + ".ply",
                               long_clouds[i], utility::FileFormat::BINARY);
    }

    // 3.0 process the initial defect clouds
    slice_along_y(long_clouds, transformation_matrix);
    std::vector<std::vector<geometry::PointCloud::Ptr>> total_defects;
    total_defects.resize(long_clouds.size());
    for (int i = 0; i < defect_clouds.size(); i++) {
        if (defect_clouds[i] == nullptr) continue;
        double r = radius * 2;
        int m = 20;
        int num_defects = core::Cluster::DBSCANCluster(*defect_clouds[i], r, m);
        std::vector<geometry::PointCloud::Ptr> defects;
        part_separation(defect_clouds[i], defects, num_defects);
        total_defects[i] = defects;
    }

    // 3.1 selection of initial defects
    std::vector<geometry::PointCloud::Ptr> filtered_defects;
    for (int i = 0; i < total_defects.size(); i++) {
        if (total_defects[i].size() == 0) continue;
        for (int j = 0; j < total_defects[i].size(); j++) {
            Eigen::Vector3d center = total_defects[i][j]->points_.at(
                    int(total_defects[i][j]->points_.size() / 2));

            // find the peak of slices
            int idx = (int)((center.y() - long_clouds[i]->points_.front().y()) /
                            transformation_matrix.y());
            // std::cout << "idx: " << idx << std::endl;
            // std::cout << "center.y(): " << center.y() << std::endl;
            // std::cout << "bottom: " << long_clouds[i]->points_.front().y()
            //           << std::endl;
            double up_bound = long_clouds[i]->y_slice_peaks[idx];
            // std::cout << "center: " << center << std::endl;
            // std::cout << "up_bound: " << up_bound << std::endl;
            // std::cout << "center.z(): " << center.z() << std::endl;
            if ((up_bound - 0.3) < center.z()) {
                filtered_defects.emplace_back(total_defects[i][j]);
            }
        }
    }
    if (debug_mode) {
        for (int i = 0; i < filtered_defects.size(); i++)
            utility::write_ply("defects_" + std::to_string(i) + ".ply",
                               filtered_defects[i],
                               utility::FileFormat::BINARY);
    }
    LOG_INFO("Detected {} defects.", filtered_defects.size());
}

void DefectDetection::detect_CSAD(std::shared_ptr<geometry::PointCloud> cloud,
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

    // 1.1 cluster
    int num_clusters =
            core::Cluster::DBSCANCluster(*points, radius, min_points);
    std::vector<geometry::PointCloud::Ptr> clusters;
    part_separation(points, clusters, num_clusters);

    // 1.1 separate r-corner and the flip-edge
    std::vector<geometry::PointCloud::Ptr> long_clouds;
    std::vector<geometry::PointCloud::Ptr> corners_clouds;
    extract_long_edge(long_clouds, corners_clouds, clusters, num_clusters);

    // 2.0 nva method
    slice_along_y(long_clouds, transformation_matrix);
    for (int i = 0; i < long_clouds.size(); i++) {
        Eigen::MatrixXd generated_map;
        generated_map = bspline_interpolation(long_clouds[i]);
        if (debug_mode) plot_matrix(generated_map, "matrix.png");
        generate_low_rank_matrix(generated_map);
    }
}

std::shared_ptr<geometry::PointCloud> DefectDetection::FPFH_NVA(
        std::shared_ptr<geometry::PointCloud> cloud,
        float ratio_x,
        float ratio_y,
        double dist_x,
        double dist_y,
        bool use_fpfh) {
    LOG_DEBUG("Start FPFH NVA Method");

    std::shared_ptr<geometry::PointCloud> defect_points =
            std::make_shared<geometry::PointCloud>();
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
        for (int i = 0; i < cloud->points_.size(); i++) {
            if (marker_x[i] == 1 || marker_y[i] == 1) {
                cloud->colors_[i] = Eigen::Vector3d(1.0, 0.0, 0.0);
                defect_points->points_.emplace_back(cloud->points_[i]);
                // defect_points->points_.emplace_back(cloud->normals_[i]);
            }
        }
    }
    LOG_DEBUG("Complete FPFH NVA Method");
    return defect_points;
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
    if (cloud->HasNormals()) {
        for (size_t i = 0; i < cloud->points_.size(); i++) {
            if (cloud->labels_[i] >= 0) {
                clusters[cloud->labels_[i]]->points_.emplace_back(
                        cloud->points_[i]);
                clusters[cloud->labels_[i]]->normals_.emplace_back(
                        cloud->normals_[i]);
            }
        }
    } else {
        for (size_t i = 0; i < cloud->points_.size(); i++) {
            if (cloud->labels_[i] >= 0) {
                clusters[cloud->labels_[i]]->points_.emplace_back(
                        cloud->points_[i]);
            }
        }
    }
}

void DefectDetection::extract_long_edge(
        std::vector<geometry::PointCloud::Ptr> &long_clouds,
        std::vector<geometry::PointCloud::Ptr> &corners_clouds,
        std::vector<geometry::PointCloud::Ptr> &clusters,
        int num_clusters) {
    LOG_DEBUG("Start extracttion long edge");
    bool has_normals = clusters[0]->HasNormals();
    for (int i = 0; i < num_clusters; i++) {
        LOG_DEBUG("cluster {}: {} points", i, clusters[i]->points_.size());
        if (clusters[i]->points_.size() > 100000) {
            double y_min = clusters[i]->GetMinBound().y();
            double y_max = clusters[i]->GetMaxBound().y();
            // double part_y_min, part_y_max;

            std::shared_ptr<geometry::PointCloud> corner_left_cloud =
                    std::make_shared<geometry::PointCloud>();
            std::shared_ptr<geometry::PointCloud> long_cloud =
                    std::make_shared<geometry::PointCloud>();
            std::shared_ptr<geometry::PointCloud> corner_right_cloud =
                    std::make_shared<geometry::PointCloud>();
            if (has_normals) {
                for (size_t j = 0; j < clusters[i]->points_.size(); j++) {
                    if (clusters[i]->points_[j].y() < y_min + 3.0) {
                        corner_left_cloud->points_.emplace_back(
                                clusters[i]->points_[j]);
                        corner_left_cloud->normals_.emplace_back(
                                clusters[i]->normals_[j]);
                        corner_left_cloud->labels_.emplace_back(0);
                    } else if (clusters[i]->points_[j].y() >= y_min + 3.0 &&
                               clusters[i]->points_[j].y() <= y_max - 3.0) {
                        long_cloud->points_.emplace_back(
                                clusters[i]->points_[j]);
                        long_cloud->normals_.emplace_back(
                                clusters[i]->normals_[j]);
                        long_cloud->labels_.emplace_back(0);
                    } else {
                        corner_right_cloud->points_.emplace_back(
                                clusters[i]->points_[j]);
                        corner_right_cloud->normals_.emplace_back(
                                clusters[i]->normals_[j]);
                        corner_right_cloud->labels_.emplace_back(0);
                    }
                }
            } else {
                for (size_t j = 0; j < clusters[i]->points_.size(); j++) {
                    if (clusters[i]->points_[j].y() < y_min + 3.0) {
                        corner_left_cloud->points_.emplace_back(
                                clusters[i]->points_[j]);

                        corner_left_cloud->labels_.emplace_back(0);
                    } else if (clusters[i]->points_[j].y() >= y_min + 3.0 &&
                               clusters[i]->points_[j].y() <= y_max - 3.0) {
                        long_cloud->points_.emplace_back(
                                clusters[i]->points_[j]);

                        long_cloud->labels_.emplace_back(0);
                    } else {
                        corner_right_cloud->points_.emplace_back(
                                clusters[i]->points_[j]);

                        corner_right_cloud->labels_.emplace_back(0);
                    }
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
    bool has_normals = long_clouds[0]->HasNormals();
    if (has_normals) {
        for (int i = 0; i < long_clouds.size(); i++) {
            Eigen::Vector3d min_bound = long_clouds[i]->GetMinBound();
            Eigen::Vector3d max_bound = long_clouds[i]->GetMaxBound();
            int num_slice = (int)(((max_bound.y() - min_bound.y()) /
                                   transformation_matrix.y()) +
                                  1);
            std::vector<double> y_slice_peaks(num_slice, 0);
            long_clouds[i]->y_slice_peaks = y_slice_peaks;
            long_clouds[i]->y_slices_.resize(num_slice);
            long_clouds[i]->ny_slices_.resize(num_slice);
            long_clouds[i]->y_slice_idxs.resize(num_slice);
            for (size_t j = 0; j < long_clouds[i]->points_.size(); j++) {
                auto pt = long_clouds[i]->points_[j];
                auto n = long_clouds[i]->normals_[j];
                int slice_idx = (int)((pt.y() - min_bound.y()) /
                                      transformation_matrix.y());
                if (long_clouds[i]->y_slice_peaks[slice_idx] <= pt.z()) {
                    long_clouds[i]->y_slice_peaks[slice_idx] = pt.z();
                }
                long_clouds[i]->y_slices_[slice_idx].emplace_back(
                        Eigen::Vector2d(pt.x(), pt.z()));
                long_clouds[i]->ny_slices_[slice_idx].emplace_back(
                        Eigen::Vector3d(n.x(), n.y(), n.z()));
                long_clouds[i]->y_slice_idxs[slice_idx].emplace_back(j);
            }
        }
    } else {
        for (int i = 0; i < long_clouds.size(); i++) {
            Eigen::Vector3d min_bound = long_clouds[i]->GetMinBound();
            Eigen::Vector3d max_bound = long_clouds[i]->GetMaxBound();
            int num_slice =
                    (int)((max_bound.y() / transformation_matrix.y()) -
                          (min_bound.y()) / transformation_matrix.y() + 1);
            std::vector<double> y_slice_peaks(num_slice, 0);
            long_clouds[i]->y_slice_peaks = y_slice_peaks;
            long_clouds[i]->y_slices_.reserve(num_slice);
            long_clouds[i]->y_slice_idxs.reserve(num_slice);
            auto pre_y = long_clouds[i]->points_[0].y();
            int slice_idx = 0;
            for (size_t j = 0; j < long_clouds[i]->points_.size(); j++) {
                auto pt = long_clouds[i]->points_[j];
                if (pt.y() != pre_y) {
                    slice_idx += 1;
                    pre_y = pt.y();
                }
                if (long_clouds[i]->y_slice_peaks[slice_idx] <= pt.z()) {
                    long_clouds[i]->y_slice_peaks[slice_idx] = pt.z();
                }
                long_clouds[i]->y_slices_[slice_idx].emplace_back(
                        Eigen::Vector2d(pt.x(), pt.z()));
            }
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

Eigen::MatrixXd DefectDetection::bspline_interpolation(
        geometry::PointCloud::Ptr cloud) {
    // get common area
    std::vector<std::vector<Eigen::Vector2d>> common_parts;
    double min_x = -1 * std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::max();
    for (int i = 0; i < cloud->y_slices_.size(); i++) {
        if (cloud->y_slices_[i].size() == 0) continue;
        if (cloud->y_slices_[i][0].x() > min_x)
            min_x = cloud->y_slices_[i][0].x();
        if (cloud->y_slices_[i].back().x() < max_x)
            max_x = cloud->y_slices_[i].back().x();
    }
    for (int i = 0; i < cloud->y_slices_.size(); i++) {
        std::vector<Eigen::Vector2d> selected_area;
        for (int j = 0; j < cloud->y_slices_[i].size(); j++) {
            if (cloud->y_slices_[i][j].x() >= min_x &&
                cloud->y_slices_[i][j].x() <= max_x) {
                selected_area.emplace_back(cloud->y_slices_[i][j]);
            }
        }
        common_parts.emplace_back(selected_area);
    }

    // use common part to fit a curve
    core::PlaneDetection plane_detector;
    int sampled_pts = 100;
    Eigen::MatrixXd sampled_map(sampled_pts, common_parts.size());
    for (int i = 0; i < common_parts.size(); i++) {
        Eigen::VectorXd gen_pts;
        gen_pts = plane_detector.fit_a_curve(common_parts[i], sampled_pts, i,
                                             false);
        sampled_map.col(i) = gen_pts;
    }

    return sampled_map;
}

void DefectDetection::plot_matrix(Eigen::MatrixXd &mat, std::string name) {
    // Convert the matrix to a OpenCV Mat
    cv::Mat cvMat(mat.rows(), mat.cols(), CV_64F);
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            cvMat.at<double>(i, j) = mat(i, j);
        }
    }

    // Plot the matrix as an image
    cv::Mat image;
    cv::normalize(cvMat, image, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // Apply a rainbow color map to the grayscale image
    cv::Mat rainbowImage;
    cv::applyColorMap(image, rainbowImage, cv::COLORMAP_RAINBOW);

    cv::imwrite(name, rainbowImage);
}

class R_pca {
public:
    Eigen::MatrixXd D;
    Eigen::MatrixXd S;
    Eigen::MatrixXd Y;
    Eigen::MatrixXd L;
    double mu;
    double mu_inv;
    double lmbda;

    R_pca(const Eigen::MatrixXd &D_in,
          double mu_in = 0.0,
          double lmbda_in = 0.0)
        : D(D_in), S(D_in.rows(), D_in.cols()), Y(D_in.rows(), D_in.cols()) {
        if (mu_in != 0.0) {
            mu = mu_in;
        } else {
            mu = D.size() / (4 * norm(D, 1));
        }
        mu_inv = 1.0 / mu;

        if (lmbda_in != 0.0) {
            lmbda = lmbda_in;
        } else {
            lmbda = 1.0 / std::sqrt(std::max(D.rows(), D.cols()));
        }
    }

    // Frobenius norm
    static double frobenius_norm(const Eigen::MatrixXd &M) { return M.norm(); }

    // Shrinkage function
    static Eigen::MatrixXd shrink(const Eigen::MatrixXd &M, double tau) {
        Eigen::MatrixXd result = M.unaryExpr([tau](double val) {
            return std::copysign(std::max(std::abs(val) - tau, 0.0), val);
        });
        return result;
    }

    // SVD thresholding function
    Eigen::MatrixXd svd_threshold(const Eigen::MatrixXd &M, double tau) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::VectorXd S = svd.singularValues();
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();

        // Shrink the singular values
        Eigen::VectorXd S_shrunk = shrink(S, tau);

        // Reconstruct the matrix with the shrunk singular values
        return U * S_shrunk.asDiagonal() * V.transpose();
    }

    // Fit function (Principal Component Pursuit)
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> fit(double tol = -1.0,
                                                    int max_iter = 1000,
                                                    int iter_print = 100) {
        int iter = 0;
        double err = std::numeric_limits<double>::infinity();
        Eigen::MatrixXd Sk = S;
        Eigen::MatrixXd Yk = Y;
        Eigen::MatrixXd Lk = Eigen::MatrixXd::Zero(D.rows(), D.cols());

        // Set tolerance
        double _tol = (tol > 0) ? tol : 1E-7 * frobenius_norm(D);

        // PCP algorithm loop
        while (err > _tol && iter < max_iter) {
            Lk = svd_threshold(D - Sk + mu_inv * Yk, mu_inv);   // Step 3
            Sk = shrink(D - Lk + mu_inv * Yk, mu_inv * lmbda);  // Step 4
            Yk = Yk + mu * (D - Lk - Sk);                       // Step 5
            err = frobenius_norm(D - Lk - Sk);
            iter++;

            if (iter % iter_print == 0 || iter == 1 || iter > max_iter ||
                err <= _tol) {
                std::cout << "Iteration: " << iter << ", Error: " << err
                          << std::endl;
            }
        }

        L = Lk;
        S = Sk;
        return {Lk, Sk};
    }

private:
    static double norm(const Eigen::MatrixXd &M, int order) {
        if (order == 1) {
            return M.cwiseAbs().sum();
        }
        return 0.0;
    }
};

void DefectDetection::generate_low_rank_matrix(Eigen::MatrixXd &mat) {
    Eigen::MatrixXd L, S;
    // utility::mathtool::RPCA(mat, L, S);
    R_pca r_pca(mat);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> L_S = r_pca.fit();
    L = L_S.first;
    S = L_S.second;
    plot_matrix(L, "L.png");
    plot_matrix(S, "S.png");
}

}  // namespace pipeline
}  // namespace hymson3d
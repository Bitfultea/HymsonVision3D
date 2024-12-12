#include "Filter.h"

#include <Eigen/Core>

#include "3D/KDtree.h"

namespace hymson3d {
namespace core {

std::tuple<PointCloud::Ptr, Eigen::MatrixXi, std::vector<std::vector<int>>>
Filter::VoxelDownSample(PointCloud::Ptr point_cloud, float voxel_size) {
    auto output = std::make_shared<PointCloud>();
    bool approximate_class = false;
    Eigen::MatrixXi cubic_id;
    if (voxel_size <= 0.0) {
        LOG_ERROR("voxel_size <= 0.");
    }

    auto voxel_min_bound = point_cloud->GetMinBound();
    auto voxel_max_bound = point_cloud->GetMaxBound();

    std::unordered_map<Eigen::Vector3i, AccumulatedPointForTrace,
                       utility::hash_eigen<Eigen::Vector3i>>
            voxelindex_to_accpoint;
    int cid_temp[3] = {1, 2, 4};

    for (size_t i = 0; i < point_cloud->points_.size(); i++) {
        auto ref_coord =
                (point_cloud->points_[i] - voxel_min_bound) / voxel_size;
        auto voxel_index = Eigen::Vector3i(int(floor(ref_coord(0))),
                                           int(floor(ref_coord(1))),
                                           int(floor(ref_coord(2))));
        int cid = 0;
        for (int c = 0; c < 3; c++) {
            if ((ref_coord(c) - voxel_index(c)) >= 0.5) {
                cid += cid_temp[c];
            }
        }
        voxelindex_to_accpoint[voxel_index].AddPoint(*point_cloud, i, cid,
                                                     approximate_class);
    }
    bool has_normals = point_cloud->HasNormals();
    bool has_colors = point_cloud->HasColors();
    bool has_covariances = point_cloud->HasCovariances();
    bool has_intensities = point_cloud->HasIntensities();
    bool has_labels = point_cloud->HasLabels();
    bool has_curves = point_cloud->HasCurvatures();
    int cnt = 0;
    cubic_id.resize(voxelindex_to_accpoint.size(), 8);
    cubic_id.setConstant(-1);
    std::vector<std::vector<int>> original_indices(
            voxelindex_to_accpoint.size());
    for (auto accpoint : voxelindex_to_accpoint) {
        output->points_.push_back(accpoint.second.GetAveragePoint());
        if (has_normals) {
            output->normals_.push_back(accpoint.second.GetAverageNormal());
        }
        if (has_colors) {
            if (approximate_class) {
                output->colors_.push_back(accpoint.second.GetMaxClass());
            } else {
                output->colors_.push_back(accpoint.second.GetAverageColor());
            }
        }
        if (has_covariances) {
            output->covariances_.emplace_back(
                    accpoint.second.GetAverageCovariance());
        }
        // if (has_labels) {
        //     output->labels_.push_back(accpoint.second.GetAverageLabel());
        // }
        if (has_curves) {
            output->curvatures_.push_back(
                    accpoint.second.GetAverageCurvature());
        }
        if (has_intensities) {
            output->intensities_.push_back(
                    accpoint.second.GetAverageIntensity());
        }
        auto original_id = accpoint.second.GetOriginalID();
        for (int i = 0; i < (int)original_id.size(); i++) {
            size_t pid = original_id[i].point_id;
            int cid = original_id[i].cubic_id;
            cubic_id(cnt, cid) = int(pid);
            original_indices[cnt].push_back(int(pid));
        }
        cnt++;
    }
    LOG_DEBUG("Pointcloud down sampled from {} points to {} points.",
              (int)point_cloud->points_.size(), (int)output->points_.size());
    return std::make_tuple(output, cubic_id, original_indices);
}

PointCloud::Ptr Filter::IndexDownSample(PointCloud::Ptr point_cloud,
                                        const std::vector<size_t> &indices,
                                        bool invert) {
    auto output = std::make_shared<PointCloud>();
    bool has_normals = point_cloud->HasNormals();
    bool has_colors = point_cloud->HasColors();
    bool has_covariance = point_cloud->HasCovariances();
    bool has_intensities = point_cloud->HasIntensities();
    bool has_labels = point_cloud->HasLabels();
    bool has_curvature = point_cloud->HasCurvatures();

    std::vector<bool> mask =
            std::vector<bool>(point_cloud->points_.size(), invert);
    for (size_t i : indices) {
        mask[i] = !invert;
    }

    for (size_t i = 0; i < point_cloud->points_.size(); i++) {
        if (mask[i]) {
            output->points_.push_back(point_cloud->points_[i]);
            if (has_normals)
                output->normals_.push_back(point_cloud->normals_[i]);
            if (has_colors) output->colors_.push_back(point_cloud->colors_[i]);
            if (has_covariance)
                output->covariances_.push_back(point_cloud->covariances_[i]);
            if (has_intensities)
                output->intensities_.push_back(point_cloud->intensities_[i]);
            if (has_labels) output->labels_.push_back(point_cloud->labels_[i]);
            if (has_curvature)
                output->curvatures_.push_back(point_cloud->curvatures_[i]);
        }
    }

    LOG_DEBUG("Pointcloud down sampled from {} points to {} points.",
              (int)point_cloud->points_.size(), (int)output->points_.size());

    return output;
}

std::tuple<PointCloud::Ptr, std::vector<size_t>> Filter::StatisticalOutliers(
        PointCloud::Ptr pointcloud, size_t nb_neighbors, double std_ratio) {
    if (nb_neighbors < 1 || std_ratio <= 0) {
        LOG_ERROR(
                "Illegal input parameters, the number of neighbors and "
                "standard deviation ratio must be positive.");
    }
    if (pointcloud->points_.size() == 0) {
        return std::make_tuple(std::make_shared<PointCloud>(),
                               std::vector<size_t>());
    }
    KDTree kdtree;
    kdtree.SetData(*pointcloud);
    std::vector<double> avg_distances =
            std::vector<double>(pointcloud->points_.size());
    std::vector<size_t> indices;
    size_t valid_distances = 0;

#pragma omp parallel for reduction(+ : valid_distances) schedule(static)
    for (int i = 0; i < int(pointcloud->points_.size()); i++) {
        std::vector<int> tmp_indices;
        std::vector<double> dist;
        kdtree.SearchKNN(pointcloud->points_[i], int(nb_neighbors), tmp_indices,
                         dist);
        double mean = -1.0;
        if (dist.size() > 0u) {
            valid_distances++;
            std::for_each(dist.begin(), dist.end(),
                          [](double &d) { d = std::sqrt(d); });
            mean = std::accumulate(dist.begin(), dist.end(), 0.0) / dist.size();
        }
        avg_distances[i] = mean;
    }

    if (valid_distances == 0) {
        return std::make_tuple(std::make_shared<PointCloud>(),
                               std::vector<size_t>());
    }
    double cloud_mean = std::accumulate(
            avg_distances.begin(), avg_distances.end(), 0.0,
            [](double const &x, double const &y) { return y > 0 ? x + y : x; });
    cloud_mean /= valid_distances;
    double sq_sum = std::inner_product(
            avg_distances.begin(), avg_distances.end(), avg_distances.begin(),
            0.0, [](double const &x, double const &y) { return x + y; },
            [cloud_mean](double const &x, double const &y) {
                return x > 0 ? (x - cloud_mean) * (y - cloud_mean) : 0;
            });
    // Bessel's correction
    double std_dev = std::sqrt(sq_sum / (valid_distances - 1));
    double distance_threshold = cloud_mean + std_ratio * std_dev;
    for (size_t i = 0; i < avg_distances.size(); i++) {
        if (avg_distances[i] > 0 && avg_distances[i] < distance_threshold) {
            indices.push_back(i);
        }
    }
    return std::make_tuple(IndexDownSample(pointcloud, indices), indices);
}

std::tuple<PointCloud::Ptr, std::vector<size_t>> Filter::RadiusOutliers(
        PointCloud::Ptr pointcloud, size_t nb_points, double search_radius) {
    if (nb_points < 1 || search_radius <= 0) {
        LOG_ERROR(
                "Illegal input parameters, the number of points and radius "
                "must be positive.");
    }
    KDTree kdtree;
    kdtree.SetData(*pointcloud);
    std::vector<bool> mask = std::vector<bool>(pointcloud->points_.size());

#pragma omp parallel for schedule(static)
    for (int i = 0; i < int(pointcloud->points_.size()); i++) {
        std::vector<int> tmp_indices;
        std::vector<double> dist;
        size_t nb_neighbors = kdtree.SearchRadius(
                pointcloud->points_[i], search_radius, tmp_indices, dist);
        mask[i] = (nb_neighbors > nb_points);
    }
    std::vector<size_t> indices;
    for (size_t i = 0; i < mask.size(); i++) {
        if (mask[i]) {
            indices.push_back(i);
        }
    }
    return std::make_tuple(IndexDownSample(pointcloud, indices), indices);
}

}  // namespace core
}  // namespace hymson3d
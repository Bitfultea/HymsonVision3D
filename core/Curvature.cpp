#include "Curvature.h"

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Gmpq.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <omp.h>
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#include "3D/PointCloud.h"
#include "Converter.h"
#include "MathTool.h"
#include "Normal.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Projection_traits_xy_3<K> Gt;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, Gt> Vb;
typedef CGAL::Triangulation_data_structure_2<Vb> Tds;
typedef CGAL::Delaunay_triangulation_2<Gt, Tds> Delaunay;

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
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimator;
        normal_estimator.setInputCloud(cloud_pcl);
        normal_estimator.setSearchMethod(tree);
        normal_estimator.setRadiusSearch(
                static_cast<hymson3d::geometry::KDTreeSearchParamRadius&>(param)
                        .radius_);
        normal_estimator.compute(*cloud_normals);
        converter::pcl_to_hymson3d_normals(cloud_normals, cloud);
        LOG_INFO("Complete normal estimation");
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
    double rad =
            static_cast<hymson3d::geometry::KDTreeSearchParamRadius&>(param)
                    .radius_;
    curvature_estimator.setRadiusSearch(rad);
    curvature_estimator.compute(*cloud_curvatures);

    cloud.curvatures_.reserve(cloud_curvatures->size());
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::min();

    for (int i = 0; i < cloud_curvatures->size(); i++) {
        geometry::curvature curvature;
        curvature.mean_curvature =
                ((*cloud_curvatures)[i].pc1 + (*cloud_curvatures)[i].pc2) / 2;
        curvature.gaussian_curvature =
                (*cloud_curvatures)[i].pc1 * (*cloud_curvatures)[i].pc2;
        curvature.total_curvature = pow((*cloud_curvatures)[i].pc1, 2) +
                                    pow((*cloud_curvatures)[i].pc2, 2);
        if (curvature.total_curvature < min_val)
            min_val = curvature.total_curvature;
        if (curvature.total_curvature > max_val)
            max_val = curvature.total_curvature;
        cloud.curvatures_.emplace_back(curvature);
    }

    cloud.colors_.reserve(cloud.points_.size());
    for (int i = 0; i < cloud.curvatures_.size(); i++) {
        cloud.colors_.emplace_back(color_with_curvature(
                cloud.curvatures_[i].total_curvature, min_val, max_val));
    }

    LOG_INFO("Compute curvature done");
}

void ComputeCurvature_TNV(geometry::PointCloud& cloud,
                          geometry::KDTreeSearchParam& param) {
    if (param.GetSearchType() != geometry::KDTreeSearchParam::SearchType::Knn) {
        LOG_ERROR(
                "Invalid search type for Compute Curvature By Triangle Normal "
                "Variation");
        return;
    }
    int knn =
            static_cast<hymson3d::geometry::KDTreeSearchParamKNN&>(param).knn_;
    std::vector<double> total_curvatures;
    total_curvatures.resize(cloud.points_.size());
    std::cout << "knn: " << knn << std::endl;
    TotalCurvaturePointCloud::TotalCurvaturePCD(cloud, total_curvatures, knn);
    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::min();
    for (int i = 0; i < cloud.points_.size(); i++) {
        geometry::curvature curvature;
        curvature.total_curvature = total_curvatures[i];
        cloud.curvatures_.emplace_back(curvature);
        if (total_curvatures[i] < min_val) min_val = total_curvatures[i];
        if (total_curvatures[i] > max_val) max_val = total_curvatures[i];
    }
    std::cout << "bbb" << min_val << " max" << max_val << std::endl;

    cloud.colors_.resize(cloud.points_.size());
    for (int i = 0; i < cloud.curvatures_.size(); i++) {
        cloud.colors_[i] = color_with_curvature(
                cloud.curvatures_[i].total_curvature, min_val, max_val);
    }
}

void ComputeSurfaceVariation(geometry::PointCloud& cloud,
                             geometry::KDTreeSearchParam& param) {
    if (!cloud.HasCurvatures()) {
        cloud.curvatures_.resize(cloud.points_.size());
    } else {
        LOG_DEBUG("Curvatures already exist. Overwrite");
    }

    std::vector<Eigen::Matrix3d> covariances;
    if (!cloud.HasCovariances()) {
        const auto& points = cloud.points_;
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

    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::min();
    double surface_variation;
    std::vector<geometry::curvature> surface_variations;
    surface_variations.resize(covariances.size());

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)covariances.size(); i++) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
        solver.compute(covariances[i]);
        Eigen::VectorXd eigenvalues = solver.eigenvalues();
        std::sort(eigenvalues.begin(), eigenvalues.end());
        surface_variation = eigenvalues[0] /
                            (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]);
        geometry::curvature writen_data;
        writen_data.total_curvature = surface_variation;
        writen_data.mean_curvature = eigenvalues[0];
        writen_data.gaussian_curvature = eigenvalues[0] * eigenvalues[1];
        surface_variations[i] = writen_data;

        if (surface_variation < min_val) min_val = surface_variation;
        if (surface_variation > max_val) max_val = surface_variation;
    }

    cloud.colors_.resize(cloud.points_.size());
#pragma omp parallel for
    for (int i = 0; i < surface_variations.size(); i++) {
        cloud.colors_[i] = color_with_curvature(
                surface_variations[i].total_curvature, min_val, max_val);
    }
}

void ComputeCurvature_PCA(geometry::PointCloud& cloud,
                          geometry::KDTreeSearchParam& param) {
    if (!cloud.HasCurvatures()) {
        cloud.curvatures_.resize(cloud.points_.size());
    } else {
        LOG_DEBUG("Curvatures already exist. Overwrite");
    }

    if (!cloud.HasNormals()) {
        LOG_DEBUG("PointCloud Has not normals");
    } else {
        ComputeNormals_PCA(cloud, param);
    }

    double min_val = std::numeric_limits<double>::max();
    double max_val = std::numeric_limits<double>::min();
    double k1, k2;

    hymson3d::geometry::KDTree kdtree;
    kdtree.SetData(cloud);
    cloud.curvatures_.resize(cloud.points_.size());

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)cloud.points_.size(); i++) {
        std::vector<int> indices;
        std::vector<double> distance2;

        if (kdtree.Search(cloud.points_[i], param, indices, distance2) >= 3) {
            auto res = calculate_point_curvature(cloud, cloud.normals_[i],
                                                 cloud.points_[i], indices);
            k1 = res.first;
            k2 = res.second;
            geometry::curvature writen_data;
            writen_data.total_curvature = pow(k1, 2) + pow(k2, 2);
            writen_data.mean_curvature = (k1 + k2) / 2.0;
            writen_data.gaussian_curvature = k1 * k2;
            cloud.curvatures_[i] = writen_data;

            if (writen_data.total_curvature < min_val)
                min_val = writen_data.total_curvature;
            if (writen_data.total_curvature > max_val)
                max_val = writen_data.total_curvature;
        }
    }

    cloud.colors_.resize(cloud.points_.size());
#pragma omp parallel for
    for (int i = 0; i < cloud.curvatures_.size(); i++) {
        cloud.colors_[i] = color_with_curvature(
                cloud.curvatures_[i].total_curvature, min_val, max_val);
    }
}

std::pair<double, double> calculate_point_curvature(geometry::PointCloud& cloud,
                                                    Eigen::Vector3d normal,
                                                    Eigen::Vector3d pt,
                                                    std::vector<int> indices) {
    Eigen::Vector3d mean;
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 9, 1> cumulants;
    cumulants.setZero();
    for (const auto& idx : indices) {
        Eigen::Vector3d n = normal.normalized();
        double d = (cloud.points_[idx] - pt).dot(n);
        Eigen::Vector3d tangent_point = pt - d * n;

        cumulants(0) += tangent_point(0);
        cumulants(1) += tangent_point(1);
        cumulants(2) += tangent_point(2);
        cumulants(3) += tangent_point(0) * tangent_point(0);
        cumulants(4) += tangent_point(0) * tangent_point(1);
        cumulants(5) += tangent_point(0) * tangent_point(2);
        cumulants(6) += tangent_point(1) * tangent_point(1);
        cumulants(7) += tangent_point(1) * tangent_point(2);
        cumulants(8) += tangent_point(2) * tangent_point(2);
    }
    cumulants /= (double)indices.size();
    mean(0) = cumulants(0);
    mean(1) = cumulants(1);
    mean(2) = cumulants(2);
    covariance(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
    covariance(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
    covariance(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
    covariance(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
    covariance(1, 0) = covariance(0, 1);
    covariance(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
    covariance(2, 0) = covariance(0, 2);
    covariance(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
    covariance(2, 1) = covariance(1, 2);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
    solver.compute(covariance);
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    std::sort(eigenvalues.begin(), eigenvalues.end());

    double k1 = eigenvalues[0];
    double k2 = eigenvalues[2];

    return std::make_pair(k1, k2);
}

Eigen::Vector3d color_with_curvature(double curvature,
                                     double min_val,
                                     double max_val) {
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

Eigen::SparseMatrix<double> CotangentLaplacian(const Eigen::MatrixXd& V,
                                               const Eigen::MatrixXi& F) {
    int n = V.rows();
    Eigen::SparseMatrix<double> L(n, n);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(12 * F.rows());

    Eigen::VectorXd diagonalEntries = Eigen::VectorXd::Zero(n);

    for (int i = 0; i < F.rows(); ++i) {
        for (int j = 0; j < 3; ++j) {
            int v1 = F(i, j);
            int v2 = F(i, (j + 1) % 3);
            int v3 = F(i, (j + 2) % 3);

            Eigen::RowVector3d p1 = V.row(v1);
            Eigen::RowVector3d p2 = V.row(v2);
            Eigen::RowVector3d p3 = V.row(v3);

            Eigen::RowVector3d u = p1 - p3;
            Eigen::RowVector3d v = p2 - p3;

            double cotangent = u.dot(v) / u.cross(v).norm();

            triplets.push_back(Eigen::Triplet<double>(v1, v2, 0.5 * cotangent));
            triplets.push_back(Eigen::Triplet<double>(v2, v1, 0.5 * cotangent));
            diagonalEntries(v1) -= 0.5 * cotangent;
            diagonalEntries(v2) -= 0.5 * cotangent;
        }
    }

    for (int i = 0; i < n; ++i) {
        triplets.push_back(Eigen::Triplet<double>(i, i, diagonalEntries(i)));
    }

    L.setFromTriplets(triplets.begin(), triplets.end());

    return L;
}

std::vector<int> TotalCurvaturePointCloud::Where(
        int i, const Eigen::MatrixXi& inArray) {
    std::vector<int> res;
    for (int r = 0; r < inArray.rows(); r++) {
        for (int c = 0; c < inArray.cols(); c++) {
            if (inArray(r, c) == i) {
                res.push_back(r);
            }
        }
    }
    return res;
}

bool TotalCurvaturePointCloud::compare_triangles(const Eigen::Vector3i& t1,
                                                 const Eigen::Vector3i& t2) {
    int sum_t1 = t1(0) + t1(1) + t1(2);
    int sum_t2 = t2(0) + t2(1) + t2(2);
    return sum_t1 < sum_t2;
}

Eigen::MatrixXi TotalCurvaturePointCloud::DelaunayKNN(
        const Eigen::MatrixXd& knn_locations_including_self,
        const Eigen::MatrixXi& idx) {
    Eigen::MatrixXi adjacent_triangles_idx_local, adjacent_triangles_idx, faces;
    Eigen::MatrixXd n, A;
    Eigen::Vector3d mean_A_vec, r0, x_axis, r1, n_plane;

    mean_A_vec = knn_locations_including_self.transpose().rowwise().mean();
    A = knn_locations_including_self.transpose().colwise() -
        mean_A_vec;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    n = svd.matrixU().col(svd.matrixU().cols() - 1);
    n_plane = {n(0), n(1), n(2)};
    r0 = {knn_locations_including_self(0, 0),
          knn_locations_including_self(0, 1),
          knn_locations_including_self(0, 2)};
    x_axis = {1, 0, 0};
    auto e1 = n_plane.cross(x_axis);
    auto e2 = n_plane.cross(e1);
    auto subtracted_r0 =
            knn_locations_including_self.rowwise() - r0.transpose();
    auto projected_e1 = subtracted_r0 * e1;
    auto projected_e2 = subtracted_r0 * e2;

    Eigen::MatrixXd knn_locations_2d(projected_e1.rows(), 2);
    knn_locations_2d << projected_e1, projected_e2;

    std::vector<std::pair<K::Point_3, int>> points_with_indices;
    for (size_t i = 0; i < knn_locations_2d.rows(); ++i) {
        points_with_indices.emplace_back(
                K::Point_3(knn_locations_2d(i, 0), knn_locations_2d(i, 1), 0.0),
                i);
    }

    Delaunay dt(points_with_indices.begin(), points_with_indices.end());

    Eigen::MatrixXi all_triangles(dt.number_of_faces(), 3);

    size_t triangle_index = 0;
    for (Delaunay::Finite_faces_iterator face = dt.finite_faces_begin();
         face != dt.finite_faces_end(); ++face) {
        Eigen::Vector3i triangle(face->vertex(0)->info(),
                                 face->vertex(1)->info(),
                                 face->vertex(2)->info());
        all_triangles.row(triangle_index) = triangle;
        triangle_index++;
    }

    std::vector<int> adjacent_triangles_mask =
            TotalCurvaturePointCloud::Where(0, all_triangles);
    
    // 替换 Eigen::all - 提取指定行的所有列
    adjacent_triangles_idx_local.resize(adjacent_triangles_mask.size(), all_triangles.cols());
    for (int i = 0; i < adjacent_triangles_mask.size(); ++i) {
        adjacent_triangles_idx_local.row(i) = all_triangles.row(adjacent_triangles_mask[i]);
    }
    
    // 替换 idx(Eigen::all, ...) - idx 是 1xN 矩阵，直接用列索引
    Eigen::MatrixXi adjacent_triangles_idx_x = idx.col(adjacent_triangles_idx_local(0, 0));
    Eigen::MatrixXi adjacent_triangles_idx_y = idx.col(adjacent_triangles_idx_local(0, 1));
    Eigen::MatrixXi adjacent_triangles_idx_z = idx.col(adjacent_triangles_idx_local(0, 2));
    
    Eigen::MatrixXi adjacent_triangles_idx_tmp(3,
                                               adjacent_triangles_idx_x.cols());
    adjacent_triangles_idx_tmp << adjacent_triangles_idx_x.transpose(),
            adjacent_triangles_idx_y.transpose(), adjacent_triangles_idx_z.transpose();
    adjacent_triangles_idx = adjacent_triangles_idx_tmp.transpose();
    return adjacent_triangles_idx;
}

double TotalCurvaturePointCloud::PerTriangleLaplacianTriangleFanCurvature(
        const Eigen::MatrixXi& adjacent_triangles_idx,
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXd& N) {
    Eigen::MatrixXd triangle, n_triangle_face, v_triangle_fan, n_triangle_fan,
            x, y, z;
    Eigen::MatrixXi f(1, 3), adjacent_triangle_idx;
    Eigen::SparseMatrix<double> l;
    double cx, cy, cz;
    double total_curvature = 0.0, total_area = 0.0;
    int N_triangles = adjacent_triangles_idx.rows();
    for (int i = 0; i < N_triangles; i++) {
        // 替换 adjacent_triangles_idx(i, Eigen::all) - 取整行
        Eigen::Vector3i triangle_idx_row = adjacent_triangles_idx.row(i);
        std::vector<int> triangle_verts = {triangle_idx_row(0),
                                           triangle_idx_row(1),
                                           triangle_idx_row(2)};
        
        // 替换 V(triangle_verts, Eigen::all) - 提取指定行的所有列
        triangle.resize(triangle_verts.size(), V.cols());
        n_triangle_face.resize(triangle_verts.size(), N.cols());
        for (int j = 0; j < triangle_verts.size(); ++j) {
            triangle.row(j) = V.row(triangle_verts[j]);
            n_triangle_face.row(j) = N.row(triangle_verts[j]);
        }
        
        // 替换 triangle(0, Eigen::all) - 取整行
        Eigen::Vector3d p0 = triangle.row(0);
        Eigen::Vector3d p1 = triangle.row(1);
        Eigen::Vector3d p2 = triangle.row(2);
        Eigen::Vector3d AB = p0 - p1;
        Eigen::Vector3d AC = p0 - p2;
        
        double triangle_area = 0.5 * sqrt((AB.cross(AC)).squaredNorm());
        f << 0, 1, 2;
        l = CotangentLaplacian(triangle, f);
        
        // 替换 n_triangle_face(Eigen::all, col) - 取整列
        x = n_triangle_face.col(0);
        y = n_triangle_face.col(1);
        z = n_triangle_face.col(2);
        
        cx = (x.transpose() * l * x)(0);
        cy = (y.transpose() * l * y)(0);
        cz = (z.transpose() * l * z)(0);
        total_curvature += (-cx - cy - cz);
        total_area += triangle_area;
    }
    total_curvature = total_curvature / total_area;
    return total_curvature;
}

void TotalCurvaturePointCloud::TotalCurvaturePCD(
        const geometry::PointCloud& cloud, std::vector<double>& k_S, int knn) {
    if (!cloud.HasNormals()) {
        LOG_ERROR("Cloud does not have normals. Exit");
        return;
    }

    const auto& points = cloud.points_;
    hymson3d::geometry::KDTree kdtree;
    kdtree.SetData(cloud);

    Eigen::MatrixXd V(cloud.points_.size(), 3);
    Eigen::MatrixXd N(cloud.normals_.size(), 3);

    for (int i = 0; i < V.rows(); i++) {
        V.row(i) = cloud.points_[i];
        N.row(i) = cloud.normals_[i];
    }

#pragma omp parallel for
    for (int i = 0; i < points.size(); i++) {
        std::vector<int> idx_vec;
        std::vector<double> distances;
        kdtree.SearchKNN(points[i], knn, idx_vec, distances);
        
        // 替换 V(idx_vec, Eigen::all) - 提取指定行的所有列
        Eigen::MatrixXd knn_locations_including_self(idx_vec.size(), V.cols());
        for (int j = 0; j < idx_vec.size(); ++j) {
            knn_locations_including_self.row(j) = V.row(idx_vec[j]);
        }
        
        int n_rows = idx_vec.size();
        Eigen::MatrixXi idx = Eigen::MatrixXi::Zero(1, n_rows);

        for (int j = 0; j < n_rows; j++) {
            idx(0, j) = idx_vec[j];
        }
        Eigen::MatrixXi adjacent_triangles_idx =
                TotalCurvaturePointCloud::DelaunayKNN(
                        knn_locations_including_self, idx);
        k_S[i] = TotalCurvaturePointCloud::
                PerTriangleLaplacianTriangleFanCurvature(adjacent_triangles_idx,
                                                         V, N);
        k_S[i] = std::pow(std::abs(k_S[i]), 0.0425);
    }
}

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
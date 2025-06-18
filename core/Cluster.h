#pragma once
#include <random>

#include "3D/KDtree.h"
#include "3D/PointCloud.h"
#include "fmtfallback.h"

namespace hymson3d {
namespace core {

class Cluster {
public:
    static void RegionGrowing_PCL(geometry::PointCloud& cloud,
                                  float normal_degree,
                                  float curvature_threshold,
                                  size_t min_cluster_size,
                                  size_t max_cluster_size,
                                  int knn);

    static void RegionGrowing_PCL(geometry::PointCloud& cloud,
                                  float normal_degree,
                                  float curvature_threshold,
                                  int knn);

    static int DBSCANCluster(geometry::PointCloud& cloud,
                             double eps,
                             size_t min_points);

    static void RegionGrowingCluster(geometry::PointCloud& cloud,
                                     float radius,
                                     float normal_degree,
                                     float curvature_threshold,
                                     int min_cluster_size);

    static int PlanarCluster(geometry::PointCloud& cloud,
                             float radius,
                             float normal_degree,
                             float curvature_threshold = 0.0f,
                             bool use_curvature = false,
                             int min_cluster_size = 100,
                             bool debug_mode = true);

    //static void RegionGrowingClusterRoi(geometry::PointCloud& cloud,
    //                                 float radius,
    //                                 float normal_degree,
    //                                 float curvature_threshold,
    //                                 int min_cluster_size,
    //                                 bool debug_mode);
    static void RegionGrowingClusterRoiFromSeeds(
            geometry::PointCloud& cloud,
            const std::vector<int>& defect_seed_indices,  // ȱ����������
            float radius,
            float normal_degree,
            float curvature_threshold,
            int min_cluster_size,
            bool debug_mode);

private:
    static Eigen::Vector3d GenerateRandomColor() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        Eigen::Vector3d random_vector;
        random_vector << dis(gen), dis(gen), dis(gen);
        return random_vector;
    }

private:
    static void paint_cluster(geometry::PointCloud& cloud, int num_cluster) {
        std::vector<Eigen::Vector3d> cluster_colors;
        for (int i = 0; i < num_cluster; i++) {
            cluster_colors.emplace_back(GenerateRandomColor());
        }
        cloud.colors_.resize(cloud.points_.size());
#pragma omp parallel for
        for (int i = 0; i < cloud.points_.size(); i++) {
            if (cloud.labels_[i] >= 0)
                cloud.colors_[i] = cluster_colors[cloud.labels_[i]];
        }
    }

    static Eigen::Vector3d HSV2RGB(double h, double s, double v) {
        double c = v * s;
        double x = c * (1 - std::fabs(std::fmod(h / 60.0, 2.0) - 1));
        double m = v - c;
        double rp, gp, bp;
        if (h < 60) {
            rp = c;
            gp = x;
            bp = 0;
        } else if (h < 120) {
            rp = x;
            gp = c;
            bp = 0;
        } else if (h < 180) {
            rp = 0;
            gp = c;
            bp = x;
        } else if (h < 240) {
            rp = 0;
            gp = x;
            bp = c;
        } else if (h < 300) {
            rp = x;
            gp = 0;
            bp = c;
        } else {
            rp = c;
            gp = 0;
            bp = x;
        }
        return Eigen::Vector3d(rp + m, gp + m, bp + m);
    }
    static void paint_cluster_dll(geometry::PointCloud& cloud, int num_cluster) {
        if (num_cluster <= 0) return;  // ��ֹ���� 0

        // Ԥ���䲢���ɵȾ�ɫ��
        std::vector<Eigen::Vector3d> cluster_colors;
        cluster_colors.reserve(num_cluster);
        for (int i = 0; i < num_cluster; i++) {
            double hue = 360.0 * i / num_cluster;
            cluster_colors.emplace_back(HSV2RGB(hue, 1.0, 1.0));
        }

        cloud.colors_.resize(cloud.points_.size());
#pragma omp parallel for
        for (int i = 0; i < cloud.points_.size(); i++) {
            int lbl = cloud.labels_[i];
            if (lbl >= 0 && lbl < num_cluster) {
                cloud.colors_[i] = cluster_colors[lbl];
            } else {
                // δ�����ǩ�ĵ�ͳһ��Ϊ��ɫ
                cloud.colors_[i] = Eigen::Vector3d(0, 0, 0);
            }
        }
    }
};

}  // namespace core
}  // namespace hymson3d
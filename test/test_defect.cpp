#include <chrono>

#include "3D/Mesh.h"
#include "Cluster.h"
#include "Converter.h"
#include "Curvature.h"
#include "DefectDetection.h"
#include "FileTool.h"
#include "Logger.h"
#include "Normal.h"
#include "fmtfallback.h"

using namespace hymson3d;

int main(int argc, char **argv) {
    /*
    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();
    //     core::converter::tiff_to_pointcloud(argv[1], argv[2], pointcloud,
    //                                         Eigen::Vector3d(0.01, 0.03,
    //                                         0.001), true);
    //     core::converter::tiff_to_pointcloud(
    //             argv[1], pointcloud, Eigen::Vector3d(0.01, 0.03, 0.001),
    //             true);
    core::converter::tiff_to_pointcloud(argv[1], pointcloud,
                                        Eigen::Vector3d(0.01, 0.03, 0.1), true);
    //     core::converter::tiff_to_pointcloud(argv[1], pointcloud,
    //                                         Eigen::Vector3d(1, 1, 1), true);

    //     geometry::HymsonMesh mesh;
    //     mesh.construct_mesh(pointcloud);

    //     float height_threshold = atof(argv[3]);
    //     float radius = atof(argv[4]);
    //     size_t min_points = (size_t)atoi(argv[5]);
    float height_threshold = atof(argv[2]);
    float radius = atof(argv[3]);
    size_t min_points = (size_t)atoi(argv[4]);
    float long_normal_degree = 2;
    float long_curvature_threshold = 0.3;
    float rcorner_normal_degree = 1.0;
    float rcorner_curvature_threshold = 0.05;
    geometry::KDTreeSearchParamRadius param(0.08);
    float ratio_x = 0.5;
    float ratio_y = 0.4;
    double dist_x = 1e-5;
    double dist_y = 1e-6;
    //     pipeline::DefectDetection::detect_defects(
    //             pointcloud, param, long_normal_degree,
    //             long_curvature_threshold, rcorner_normal_degree,
    //             rcorner_curvature_threshold, height_threshold, radius,
    //             min_points);

    //     pipeline::DefectDetection::detect_pinholes(
    //             pointcloud, param, height_threshold, radius, min_points);

    bool denoise = false;
    bool debug_mode = false;
    //     Eigen::Vector3d transformation_matrix = Eigen::Vector3d(0.01, 0.03,
    //     0.001);
    Eigen::Vector3d transformation_matrix = Eigen::Vector3d(0.01, 0.03, 0.5);
    //     Eigen::Vector3d transformation_matrix = Eigen::Vector3d(1, 1, 1);
    auto start = std::chrono::high_resolution_clock::now();
    pipeline::DefectDetection::detect_pinholes_nva(
            pointcloud, param, height_threshold, radius, min_points,
            transformation_matrix, ratio_x, ratio_y, dist_x, dist_y, denoise,
            debug_mode);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "times:" << elapsed.count() << "ms" << std::endl;
    //     core::Cluster::RegionGrowingCluster(*pointcloud, radius,
    //     long_normal_degree,
    //                                         long_curvature_threshold,
    //                                         min_points);
    //     utility::write_ply("gg.ply", pointcloud,
    //     utility::FileFormat::BINARY);

    */

    //for windows test
    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();
    utility::read_ply("F:/qhchen/EncapsulationCplus/EncapsulationCplus/output_pointcloud.ply", pointcloud);
    //core::converter::tiff_to_pointcloud(argv[1], pointcloud,
    //                                    Eigen::Vector3d(0.01, 0.03, 0.1), true);
    //     core::converter::tiff_to_pointcloud(argv[1], pointcloud,
    //                                         Eigen::Vector3d(1, 1, 1), true);

    float height_threshold = -0.5;
    float radius = 0.08;
    size_t min_points = 5;
    geometry::KDTreeSearchParamRadius param(0.08);
    float ratio_x = 0.5;
    float ratio_y = 0.4;
    double dist_x = 1e-5;
    double dist_y = 1e-6;
    bool denoise = false;
    bool debug_mode = true;
    std::string debug_path = "C:\\Users\\Administrator\\Desktop\\defect_detection\\";
    Eigen::Vector3d transformation_matrix = Eigen::Vector3d(0.01, 0.03, 0.1);
    std::vector<geometry::PointCloud::Ptr> filtered_defects;
    std::vector<int> filtered_defects_ids;
    //std::vector<Eigen::Vector3d> filtered_defects_bounds;
    std::vector<std::vector<double>> defect_bounds;
    auto start = std::chrono::high_resolution_clock::now();
    pipeline::DefectDetection::detect_pinholes_nva_roi_dll_fast(
            pointcloud, param, filtered_defects, filtered_defects_ids, 
            defect_bounds, debug_path, height_threshold, radius, 
            transformation_matrix, ratio_x, ratio_y, dist_x, dist_y, denoise, debug_mode);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "times:" << elapsed.count() << "ms" << std::endl;
    for (int i : filtered_defects_ids) {
        std::cout << "filter_defects_ids:" << i << std::endl;
    }
    int idx = 0;
    for (const auto& row : defect_bounds) {
        std::cout << "Fragment " << idx++ << ": ";
        for (double v : row) {
            // 例如固定两位小数：
            std::cout << std::fixed << std::setprecision(3) << v << ' ';
        }
        std::cout << std::endl;
    }
    return 0;
}
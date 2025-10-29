#include <chrono>

#include "Cluster.h"
#include "Converter.h"
#include "Curvature.h"
#include "DiskLevelMeasurement.h"
#include "FileTool.h"
#include "Logger.h"
#include "Normal.h"
// #include "PlanarDegree.h"
#include "fmtfallback.h"

using namespace hymson3d;

int main(int argc, char **argv) {
    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();
    std::string tiff_path = "F:\\TestData\\21_04_29_099_.tiff";
    //core::converter::tiff_to_pointcloud(argv[1], pointcloud,
    //                                    Eigen::Vector3d(1, 1, 400), false);
    core::converter::tiff_to_pointcloud(tiff_path, pointcloud,
                                        Eigen::Vector3d(1, 1, 400), false);

    float radius = 5.0f;
    float normal_degree = 0.5;
    float curvature_threshold = 0.0;
    bool use_curvature = false;
    float central_plane_size = 200.0;
    float distance_threshold = 0.0;
    int min_planar_points = 2500;
    bool debug_mode = true;
    geometry::KDTreeSearchParamRadius param(radius);

    // core::Cluster::PlanarCluster(*pointcloud, radius, nromal_degree,
    //                              curvature_threshold, false,
    //                              min_planar_points, true);

    auto start = std::chrono::high_resolution_clock::now();
    pipeline::DiskLevelMeasurementResult result;
    // pipeline::DiskLevelMeasurement::measure_pindisk_heightlevel(
    //         pointcloud, param, &result, central_plane_size, normal_degree,
    //         distance_threshold, min_planar_points, debug_mode);
    pipeline::DiskLevelMeasurement::measure_pindisk_heightlevel(
            pointcloud, &result, central_plane_size, true);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Planar segmentation times:" << elapsed.count() << "ms"
              << std::endl;
    std::cout << "Results: Angle->" << result.plane_angle << ", Gap->"
              << result.plane_height_gap << std::endl;

    // utility::write_ply("planar_cluster.ply", pointcloud,
    //                    utility::FileFormat::BINARY);
    return 0;
}

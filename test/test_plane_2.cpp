#include <chrono>

#include "Cluster.h"
#include "Converter.h"
#include "Curvature.h"
#include "DiskLevelMeasurement.h"
#include "Feature.h"
#include "FileTool.h"
#include "Logger.h"
#include "Normal.h"
#include "Raster.h"
// #include "PlanarDegree.h"
#include "fmtfallback.h"

using namespace hymson3d;

int main(int argc, char** argv) {
    int detect_mode = 2;
    bool debug_mode = true;

    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();
    //core::converter::tiff_to_pointcloud(argv[1], pointcloud,
    //                                    Eigen::Vector3d(1, 1, 200), false);

    cv::Mat tiff_image;
    utility::read_tiff(argv[1], tiff_image);
    int kernal_size = 101;
    float delta = 1.5;
    pipeline::DiskLevelMeasurement::preprocess_img(tiff_image, kernal_size, delta);
    core::converter::mat_to_pointcloud(tiff_image, pointcloud,
                                       Eigen::Vector3d(1, 1, 200), false);
    core::PointCloudRaster raster;
    std::cout << "type of tiff_image:" << tiff_image.type() << std::endl;
    cv::Mat pre_processed = raster.project_to_feature_frame(tiff_image);
    std::pair<bool, cv::Point2f> disk_centre =
            core::feature::detect_green_ring(pre_processed, debug_mode);

    if (detect_mode == 2) {
        float radius = 5.0f;
        float normal_degree = 1;
        float curvature_threshold = 0.0;
        bool use_curvature = false;
        float central_plane_size = 75.0;
        float distance_threshold = 0.0;
        int min_planar_points = 100;
        int method = 1;
        geometry::KDTreeSearchParamRadius param(radius);

        // core::Cluster::PlanarCluster(*pointcloud, radius, nromal_degree,
        //                              curvature_threshold, false,
        //                              min_planar_points, true);

        auto start = std::chrono::high_resolution_clock::now();
        pipeline::DiskLevelMeasurementResult result;
        pipeline::DiskLevelMeasurement::perform_measurement(
                pointcloud, param, &result, disk_centre, central_plane_size,
                normal_degree, distance_threshold, min_planar_points, method,
                debug_mode);
        //     pipeline::DiskLevelMeasurement::measure_pindisk_heightlevel(
        //             pointcloud, param, &result, central_plane_size,
        //             normal_degree, distance_threshold, min_planar_points,
        //             debug_mode);
        //     pipeline::DiskLevelMeasurement::measure_pindisk_heigh tlevel(
        //             pointcloud, &result, central_plane_size, debug_mode);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Planar segmentation times:" << elapsed.count() << "ms"
                  << std::endl;
        std::cout << "Results: Angle->" << result.plane_angle << ", Gap->"
                  << result.plane_height_gap << std::endl;

        // utility::write_ply("planar_cluster.ply", pointcloud,
        //                    utility::FileFormat::BINARY);
    }

    return 0;
}

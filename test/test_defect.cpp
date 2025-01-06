#include <chrono>

#include "3D/Mesh.h"
#include "Cluster.h"
#include "Converter.h"
#include "Curvature.h"
#include "DefectDetection.h"
#include "FileTool.h"
#include "Logger.h"
#include "Normal.h"

using namespace hymson3d;

int main(int argc, char **argv) {
    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();
    core::converter::tiff_to_pointcloud(
            argv[1], pointcloud, Eigen::Vector3d(0.01, 0.03, 0.001), true);

    //     geometry::HymsonMesh mesh;
    //     mesh.construct_mesh(pointcloud);

    float height_threshold = atof(argv[2]);
    float radius = atof(argv[3]);
    size_t min_points = (size_t)atoi(argv[4]);
    float long_normal_degree = 1.3;
    float long_curvature_threshold = 0.5;
    float rcorner_normal_degree = 1.0;
    float rcorner_curvature_threshold = 0.05;
    geometry::KDTreeSearchParamRadius param(0.2);

    //     pipeline::DefectDetection::detect_defects(
    //             pointcloud, param, long_normal_degree,
    //             long_curvature_threshold, rcorner_normal_degree,
    //             rcorner_curvature_threshold, height_threshold, radius,
    //             min_points);

    pipeline::DefectDetection::detect_pinholes(
            pointcloud, param, height_threshold, radius, min_points);
}
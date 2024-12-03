#include "Converter.h"
#include "Curvature.h"
#include "FileTool.h"
#include "Logger.h"
#include "PlanarDegree.h"

using namespace hymson3d;

int main(int argc, char **argv) {
    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();
    core::converter::tiff_to_pointcloud(argv[1], pointcloud,
                                        Eigen::Vector3d(1, 1, 1));
    utility::write_ply("test_plane.ply", pointcloud,
                       utility::FileFormat::BINARY);
    pipeline::PlanarDegree planar_degree;
    double degree = planar_degree.compute_planar_degree(*pointcloud);
    LOG_INFO("Degree: {}", degree);

    // test curvature
    geometry::KDTreeSearchParamRadius param(10.0);
    core::feature::ComputeCurvature_PCL(*pointcloud, param);
    utility::write_ply("test_curvature.ply", pointcloud,
                       utility::FileFormat::BINARY);
    return 0;
}

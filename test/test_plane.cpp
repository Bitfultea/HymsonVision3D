#include "Converter.h"
#include "Curvature.h"
#include "FileTool.h"
#include "Logger.h"
#include "Normal.h"
#include "PlanarDegree.h"

using namespace hymson3d;

int main(int argc, char **argv) {
    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();
    core::converter::tiff_to_pointcloud(
            argv[1], pointcloud, Eigen::Vector3d(0.01, 0.03, 0.001), true);
    utility::write_ply("test_plane.ply", pointcloud,
                       utility::FileFormat::BINARY);
    pipeline::PlanarDegree planar_degree;
    double degree = planar_degree.compute_planar_degree(*pointcloud);
    LOG_INFO("Degree: {}", degree);

    // test curvature
    std::cout << "Has normals: " << pointcloud->HasNormals() << std::endl;
    std::cout << "Has covariances: " << pointcloud->HasCovariances()
              << std::endl;
    std::cout << "Has points: " << pointcloud->points_.size() << std::endl;
    geometry::KDTreeSearchParamRadius param(0.2);
    core::feature::ComputeNormals_PCA(*pointcloud, param);
    std::cout << "Normals computed" << std::endl;

    std::cout << "Has normals: " << pointcloud->HasNormals() << std::endl;
    std::cout << "Has covariances: " << pointcloud->HasCovariances()
              << std::endl;
    std::cout << pointcloud->covariances_.size() << std::endl;
    //     core::feature::ComputeCurvature_PCL(*pointcloud, param);
    core::feature::ComputeSurfaceVariation(*pointcloud, param);
    utility::write_ply("test_curvature_2.ply", pointcloud,
                       utility::FileFormat::BINARY);

    // test histogram

    return 0;
}

#include <chrono>

#include "Cluster.h"
#include "Converter.h"
#include "Curvature.h"
#include "FileTool.h"
#include "Logger.h"
#include "Normal.h"
// #include "PlanarDegree.h"
#include "fmtfallback.h"

using namespace hymson3d;

int main(int argc, char **argv) {
    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();
    core::converter::tiff_to_pointcloud(argv[1], pointcloud,
                                        Eigen::Vector3d(1, 1, 400), false);
    core::Cluster::PlanarCluster(*pointcloud, 5, 0.5, 0.0, false, 100, true);

    utility::write_ply("planar_cluster.ply", pointcloud,
                       utility::FileFormat::BINARY);
    return 0;
}

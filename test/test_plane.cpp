#include "Converter.h"
#include "FileTool.h"
#include "Logger.h"
#include "PlanarDegree.h"

using namespace hymson3d;

int main(int argc, char **argv) {
    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();
    core::converter::tiff_to_pointcloud(argv[1], pointcloud);
    pipeline::PlanarDegree planar_degree;
    double degree = planar_degree.compute_planar_degree(*pointcloud);
    LOG_INFO("Degree: {}", degree);
}

// convert data between different types
#pragma once

#include "2D/Image.h"
#include "3D/PointCloud.h"
#include "FileTool.h"
#include "Logger.h"

namespace hymson3d {
namespace core {
namespace converter {

// convert tiff data to point cloud
void tiff_to_pointcloud(const std::string& tiff_path);

}  // namespace converter
}  // namespace core
}  // namespace hymson3d

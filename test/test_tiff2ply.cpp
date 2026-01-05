#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "3D/Mesh.h"
#include "Converter.h"
#include "FileTool.h"
#include "Logger.h"
#include "Normal.h"
#include "fmtfallback.h"

using namespace hymson3d;
namespace fs = std::filesystem;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <tiff_folder>" << std::endl;
        return -1;
    }

    std::string folder_path = argv[1];

    for (const auto &entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::string extension = entry.path().extension().string();
            // 支持.tiff和.tif
            if (extension == ".tiff" || extension == ".tif" ||
                extension == ".TIFF" || extension == ".TIF") {
                std::cout << "Processing: " << file_path << std::endl;
                geometry::PointCloud::Ptr pointcloud =
                        std::make_shared<geometry::PointCloud>();
                core::converter::tiff_to_pointcloud(
                        file_path, pointcloud, Eigen::Vector3d(1, 1, 200),
                        false);
                // 构造ply输出文件名
                auto path = entry.path();
                path.replace_extension(".ply");
                std::string ply_file = path.string();
                //std::string ply_file =
                //        entry.path().replace_extension(".ply").string();
                utility::write_ply(ply_file, pointcloud,
                                   utility::FileFormat::BINARY);
            }
        }
    }

    return 0;
}

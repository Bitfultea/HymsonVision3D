#include "FileTool.h"

#include "happly.h"
using namespace happly;
namespace hymson3d {
namespace utility {
bool read_tiff(const std::string& filename, cv::Mat& image) {
    image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        LOG_ERROR("无法读取TIFF文件！");
        return false;
    }

    // 显示图像信息
    LOG_DEBUG("图像路径：{}", filename);
    LOG_DEBUG("图像宽度：{}", image.cols);
    LOG_DEBUG("图像高度：{}", image.rows);
    LOG_DEBUG("通道数：{}", image.channels());
    return true;
}

// simple function to write a PLY with xyz datas
void write_ply(const std::string& filename,
               const std::vector<Eigen::Vector3d>& points) {
    // Create a point cloud
    pcl::PointCloud<pcl::PointXYZ> cloud;

    // Add points to the cloud
    for (auto pt : points) {
        cloud.points.emplace_back(pcl::PointXYZ(pt.x(), pt.y(), pt.z()));
    }

    // Create a PCD writer
    pcl::PLYWriter writer;

    // Write the point cloud to a file
    writer.write<pcl::PointXYZ>(filename, cloud);
    LOG_INFO("Write {} points to {}", cloud.size(), filename);
}

void write_ply(const std::string& filename,
               geometry::PointCloud::Ptr pointcloud,
               FileFormat format) {
    // Create an empty object
    LOG_INFO("Start Writing PointCloud to PLY");
    PLYData plyOut;

    std::vector<std::array<double, 3>> vertices;
    vertices.reserve(pointcloud->points_.size());
    for (const auto& point : pointcloud->points_) {
        vertices.push_back({point.x(), point.y(), point.z()});
    }
    plyOut.addVertexPositions(vertices);

    if (pointcloud->HasColors()) {
        LOG_INFO("Write Color to PLY");
        std::vector<unsigned char> red, green, blue;
        red.reserve(pointcloud->points_.size());
        green.reserve(pointcloud->points_.size());
        blue.reserve(pointcloud->points_.size());
        for (const auto& color : pointcloud->colors_) {
            red.push_back(int8_t(color.x() * 255));
            green.push_back(int8_t(color.y() * 255));
            blue.push_back(int8_t(color.z() * 255));
        }
        plyOut.getElement("vertex").addProperty<unsigned char>("red", red);
        plyOut.getElement("vertex").addProperty<unsigned char>("green", green);
        plyOut.getElement("vertex").addProperty<unsigned char>("blue", blue);
    }
    if (pointcloud->HasNormals()) {
        LOG_INFO("Write Normal to PLY");
        std::vector<double> nx, ny, nz;
        nx.reserve(pointcloud->points_.size());
        ny.reserve(pointcloud->points_.size());
        nz.reserve(pointcloud->points_.size());
        for (const auto& normal : pointcloud->normals_) {
            nx.push_back(normal.x());
            ny.push_back(normal.y());
            nz.push_back(normal.z());
        }
        plyOut.getElement("vertex").addProperty<double>("nx", nx);
        plyOut.getElement("vertex").addProperty<double>("ny", ny);
        plyOut.getElement("vertex").addProperty<double>("nz", nz);
    }
    if (pointcloud->HasIntensities()) {
        LOG_INFO("Write Intensity to PLY");
        std::vector<float> intensity;
        intensity.reserve(pointcloud->points_.size());
        for (const auto& i : pointcloud->intensities_) {
            intensity.push_back(i);
        }
        plyOut.getElement("vertex").addProperty<float>("intensity", intensity);
    }
    if (pointcloud->HasCurvatures()) {
        LOG_INFO("Write Curvature to PLY");
        std::vector<double> curvature;
        curvature.reserve(pointcloud->points_.size());
        for (const auto& c : pointcloud->curvatures_) {
            // use total curvature for visualization
            curvature.push_back(c->total_curvature);
        }
        plyOut.getElement("vertex").addProperty<double>("curvature", curvature);
    }

    if (format == FileFormat::ASCII) {
        plyOut.write(filename, happly::DataFormat::ASCII);
    } else {
        plyOut.write(filename, happly::DataFormat::Binary);
    }

    LOG_INFO("Finish Writing PointCloud to PLY");
}

void read_ply(const std::string& filename,
              std::vector<Eigen::Vector3d>& points) {
    LOG_INFO("Start Reading PointCloud from PLY");
    PLYData plyIn(filename);
    std::vector<std::array<double, 3>> vertices = plyIn.getVertexPositions();
    points.reserve(vertices.size());
    for (const auto& vertex : vertices) {
        points.emplace_back(vertex[0], vertex[1], vertex[2]);
    }
    LOG_INFO("Finish Reading PointCloud With {} points from PLY",
             points.size());
}

void read_ply(const std::string& filename,
              geometry::PointCloud::Ptr pointcloud) {
    LOG_INFO("Start Reading PointCloud from PLY");

    PLYData plyIn(filename);
    std::vector<std::array<double, 3>> vertices = plyIn.getVertexPositions();
    pointcloud->points_.reserve(vertices.size());
    for (const auto& vertex : vertices) {
        pointcloud->points_.emplace_back(vertex[0], vertex[1], vertex[2]);
    }

    // check for normals (nx,ny,nz)
    if (plyIn.getElement("vertex").hasProperty("nx") &&
        plyIn.getElement("vertex").hasProperty("ny") &&
        plyIn.getElement("vertex").hasProperty("nz")) {
        LOG_INFO("Read Pointcloud Normals from PLY");

        std::vector<double> nx =
                plyIn.getElement("vertex").getProperty<double>("nx");
        std::vector<double> ny =
                plyIn.getElement("vertex").getProperty<double>("ny");
        std::vector<double> nz =
                plyIn.getElement("vertex").getProperty<double>("nz");

        pointcloud->normals_.reserve(vertices.size());
        for (size_t i = 0; i < nx.size(); ++i) {
            pointcloud->normals_.emplace_back(nx[i], ny[i], nz[i]);
        }
    }

    // check for RGB (red, green, blue)
    if (plyIn.getElement("vertex").hasProperty("red") &&
        plyIn.getElement("vertex").hasProperty("green") &&
        plyIn.getElement("vertex").hasProperty("blue")) {
        std::vector<unsigned char> red =
                plyIn.getElement("vertex").getProperty<unsigned char>("red");
        std::vector<unsigned char> green =
                plyIn.getElement("vertex").getProperty<unsigned char>("green");
        std::vector<unsigned char> blue =
                plyIn.getElement("vertex").getProperty<unsigned char>("blue");

        pointcloud->colors_.reserve(vertices.size());
        for (size_t i = 0; i < red.size(); ++i) {
            pointcloud->colors_.emplace_back(red[i], green[i], blue[i]);
        }
    }
    for (const auto& property : plyIn.getElement("vertex").getPropertyNames()) {
        std::cout << "Property found: " << property << std::endl;
    }
    LOG_INFO("Finish Reading PointCloud from PLY");
}

}  // namespace utility
}  // namespace hymson3d
#include "FileTool.h"

#include <tiffio.h>

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

bool read_tiff_libtiff(const std::string& filename,
                       cv::Mat& height_map,
                       cv::Mat& intensity_map) {
    TIFF* tiff = TIFFOpen(filename.c_str(), "r");
    if (!tiff) {
        LOG_ERROR("无法打开TIFF文件: {}", filename);
        return false;
    }

    uint32_t width, height;
    uint16_t channels, bits_per_sample;
    uint16_t info;

    TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height);
    TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &channels);
    TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
    TIFFGetField(tiff, TIFFTAG_PAGENUMBER, &info);

    LOG_DEBUG("TIFF信息 - 宽度: {}, 高度: {}, 通道数: {}, 位深度: {}", width,
              height, channels, bits_per_sample);
    std::cout << "图像信息：" << info << std::endl;
    if (bits_per_sample != 32) {
        LOG_ERROR("不支持的位深度: {}", bits_per_sample);
        TIFFClose(tiff);
        return false;
    }
    height_map = cv::Mat(height, width, CV_32FC1);
    intensity_map = cv::Mat(height, width, CV_32FC1);

    for (uint32_t row = 0; row < height; row++) {
        float* buf = new float[width];
        if (TIFFReadScanline(tiff, buf, row) == -1) {
            LOG_ERROR("读取扫描行失败: {}", row);
            delete[] buf;
            TIFFClose(tiff);
            return false;
        }

        float* height_ptr = height_map.ptr<float>(row);
        float* intensity_ptr = intensity_map.ptr<float>(row);

        // // 解析每个像素的32位浮点数
        // for (uint32_t col = 0; col < width; col++) {
        //     union {
        //         float f;
        //         uint32_t i;
        //     } converter;
        //     converter.f = buf[col];

        //     // 方法1: 假设高16位是高度，低16位是灰度值
        //     // 这是一种常见的编码方式
        //     uint32_t bits = converter.i;
        //     uint16_t height_bits = (bits >> 16) & 0xFFFF;
        //     uint16_t intensity_bits = bits & 0xFFFF;

        //     // 将位模式转换回浮点数
        //     union {
        //         uint32_t i;
        //         float f;
        //     } height_converter, intensity_converter;

        //     // 构造新的浮点数表示
        //     height_converter.i = (0x40000000 | (height_bits << 8));  //
        //     简单示例 intensity_converter.i = (0x40000000 | (intensity_bits <<
        //     8));

        //     height_ptr[col] = height_converter.f - 2.0f;  // 调整到合理范围
        //     intensity_ptr[col] = intensity_converter.f - 2.0f;
        // }
        for (uint32_t col = 0; col < width; col++) {
            float value = buf[col];
            float integer_part = std::floor(std::abs(value));
            float decimal_part = std::abs(value) - integer_part;

            height_ptr[col] = integer_part;
            intensity_ptr[col] = decimal_part * 10000;  // 放大以便观察
        }
        delete[] buf;
    }

    TIFFClose(tiff);
    LOG_INFO("成功读取TIFF文件，尺寸: {}x{}", width, height);
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

std::pair<cv::Mat, cv::Mat> separate_float_components(
        const cv::Mat& float_image) {
    if (float_image.type() != CV_32FC1) {
        LOG_ERROR("输入必须是单通道32位浮点图像");
        return std::make_pair(cv::Mat(), cv::Mat());
    }

    cv::Mat integer_part(float_image.size(), CV_32FC1);
    cv::Mat decimal_part(float_image.size(), CV_32FC1);

    for (int i = 0; i < float_image.rows; ++i) {
        for (int j = 0; j < float_image.cols; ++j) {
            float value = float_image.at<float>(i, j);
            float integer = std::floor(std::abs(value));  // 获取整数部分
            float decimal = std::abs(value) - integer;    // 获取小数部分

            integer_part.at<float>(i, j) = integer;
            decimal_part.at<float>(i, j) = decimal;

            // integer_part.at<uint8_t>(i, j) = integer_val;
            // decimal_part.at<float>(i, j) = decimal_val;
        }
    }

    return std::make_pair(integer_part, decimal_part);
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

void write_plane_mesh_ply(const std::string& filename,
                          geometry::Plane& plane,
                          double x_min,
                          double x_max,
                          double y_min,
                          double y_max,
                          int resolution) {
    if (plane.coeff_ == Eigen::Vector4d::Zero()) {
        LOG_ERROR("Plane coeff is zero");
        return;
    }

    double a = plane.coeff_(0);
    double b = plane.coeff_(1);
    double c = plane.coeff_(2);
    double d = plane.coeff_(3);
    // std::cout << "Plane coeff: " << a << " " << b << " " << c << " " << d
    //           << std::endl;
    // std::cout << plane.coeff_ << std::endl;

    if (std::abs(c) < 1e-10) {
        LOG_ERROR("平面法向量z分量接近0，无法生成网格");
        return;
    }

    std::vector<std::array<double, 3>> vertices;
    std::vector<std::vector<int>> faces;

    double x_step = (x_max - x_min) / (resolution - 1);
    double y_step = (y_max - y_min) / (resolution - 1);

    for (int i = 0; i < resolution; ++i) {
        for (int j = 0; j < resolution; ++j) {
            double x = x_min + i * x_step;
            double y = y_min + j * y_step;
            // z: z = -(ax + by + d) / c
            double z = -(a * x + b * y + d) / c;
            vertices.push_back({x, y, z});
        }
    }

    // 生成面（三角形）
    for (int i = 0; i < resolution - 1; ++i) {
        for (int j = 0; j < resolution - 1; ++j) {
            int idx0 = i * resolution + j;
            int idx1 = idx0 + 1;
            int idx2 = (i + 1) * resolution + j;
            int idx3 = idx2 + 1;

            // 每个格子生成两个三角形
            faces.push_back({idx0, idx1, idx2});
            faces.push_back({idx1, idx3, idx2});
        }
    }

    // 创建PLY数据对象
    happly::PLYData plyOut;

    // 添加顶点
    plyOut.addVertexPositions(vertices);

    // 添加面
    plyOut.addFaceIndices(faces);

    // 写入文件
    plyOut.write(filename, happly::DataFormat::Binary);

    LOG_INFO("成功将平面网格写入PLY文件: {}", filename);
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
        LOG_DEBUG("Property found: {}", property);
    }
    LOG_INFO("Finish Reading PointCloud with {} points from PLY.",
             pointcloud->points_.size());
}

}  // namespace utility
}  // namespace hymson3d
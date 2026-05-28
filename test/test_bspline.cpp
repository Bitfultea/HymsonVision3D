// B-spline gap/step detection test
//
// Usage:
//   bspline_test [ply_file] [debug_output_dir]
//   bspline_test <tiff_file> [ratio_x,ratio_y,ratio_z] [debug_output_dir]
//
//   ply_file / tiff_file — path to input .ply or .tiff file
//   ratio_x,y,z          — TIFF→点云缩放比例 (default: 0.01,0.03,0.001)
//   debug_output_dir     — path to debug output directory (default:
//   ./bspline_debug/)
//
// Examples:
//   ./bspline_test /data/scan.ply
//   ./bspline_test /data/scan.tiff
//   ./bspline_test /data/scan.tiff 0.005,0.1,0.001 /tmp/bspline_out/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "3D/Mesh.h"
#include "Cluster.h"
#include "Converter.h"
#include "Curvature.h"
#include "DefectDetection.h"
#include "FileSystem.h"
#include "FileTool.h"
#include "GapStepDetection.h"
#include "Logger.h"
#include "MathTool.h"
#include "Normal.h"
#include "PlaneDetection.h"
#include "fmtfallback.h"

using namespace hymson3d;

#ifdef _WIN32
const char* DEFAULT_PLY =
        "F:/qhchen/EncapsulationCplus/EncapsulationCplus/"
        "output_pointcloud.ply";
const char* DEFAULT_DEBUG = "C:/Users/Administrator/Desktop/res/bspline/";
#else
const char* DEFAULT_PLY = "./output_pointcloud.ply";
const char* DEFAULT_DEBUG = "./bspline_debug/";
#endif

static void wait_for_key() {
#ifdef _WIN32
    system("pause");
#else
    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.get();
#endif
}

static bool has_tiff_ext(const char* path) {
    const char* ext = strrchr(path, '.');
    if (!ext) return false;
    return strcasecmp(ext, ".tiff") == 0 || strcasecmp(ext, ".tif") == 0;
}

static bool has_ply_ext(const char* path) {
    const char* ext = strrchr(path, '.');
    if (!ext) return false;
    return strcasecmp(ext, ".ply") == 0;
}

static bool parse_ratio(const char* str, Eigen::Vector3d& ratio) {
    return sscanf(str, "%lf,%lf,%lf", &ratio.x(), &ratio.y(), &ratio.z()) == 3;
}

static void write_self_test_debug(
        const std::string& path,
        const std::vector<Eigen::Vector2d>& pts,
        const std::vector<std::vector<Eigen::Vector2d>>& groups) {
    if (pts.empty()) return;

    std::vector<double> x_vec;
    std::vector<double> y_vec;
    x_vec.reserve(pts.size());
    y_vec.reserve(pts.size());
    for (const auto& pt : pts) {
        x_vec.push_back(pt.x());
        y_vec.push_back(pt.y());
    }
    const double x_min = *std::min_element(x_vec.begin(), x_vec.end());
    const double x_max = *std::max_element(x_vec.begin(), x_vec.end());
    const double y_min = *std::min_element(y_vec.begin(), y_vec.end());
    const double y_max = *std::max_element(y_vec.begin(), y_vec.end());
    const double x_span = std::max(x_max - x_min, 1e-9);
    const double y_span = std::max(y_max - y_min, 1e-9);

    cv::Mat image(520, 840, CV_8UC3, cv::Scalar(255, 255, 255));
    auto to_pixel = [&](const Eigen::Vector2d& pt) {
        int x = static_cast<int>(20 + (pt.x() - x_min) / x_span * 800);
        int y = static_cast<int>(500 - (pt.y() - y_min) / y_span * 480);
        return cv::Point(x, y);
    };

    for (const auto& pt : pts) {
        cv::circle(image, to_pixel(pt), 2, cv::Scalar(180, 180, 180), -1);
    }
    const cv::Scalar colors[] = {cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0),
                                 cv::Scalar(0, 165, 255)};
    for (int i = 0; i < groups.size(); ++i) {
        const cv::Scalar color = colors[i % 3];
        for (const auto& pt : groups[i]) {
            cv::circle(image, to_pixel(pt), 3, color, -1);
        }
        if (groups[i].size() >= 2) {
            auto minmax_x = std::minmax_element(
                    groups[i].begin(), groups[i].end(),
                    [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                        return a.x() < b.x();
                    });
            cv::line(image, to_pixel(*minmax_x.first),
                     to_pixel(*minmax_x.second), color, 2);
        }
    }
    cv::imwrite(path, image);
}

static int run_self_test(const char* debug_dir) {
    std::vector<Eigen::Vector2d> sampled_pts;
    sampled_pts.reserve(92);
    for (int x = 0; x <= 45; ++x) {
        sampled_pts.emplace_back(static_cast<double>(x), 0.0);
    }
    for (int x = 55; x <= 100; ++x) {
        sampled_pts.emplace_back(static_cast<double>(x), 0.0);
    }
    std::sort(sampled_pts.begin(), sampled_pts.end(),
              [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                  if (a.x() == b.x()) return a.y() < b.y();
                  return a.x() < b.x();
              });

    auto groups = pipeline::GapStepDetection::test_group_by_derivative_dll(
            sampled_pts);
    if (debug_dir) {
        utility::filesystem::MakeDirectoryHierarchy(debug_dir);
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_overlap.png", sampled_pts,
                              groups);
    }
    if (groups.size() != 2 || groups[0].empty() || groups[1].empty()) {
        std::cerr << "group_by_derivative_dll should return two non-empty "
                     "lines"
                  << std::endl;
        return 1;
    }

    auto mean_y = [](const std::vector<Eigen::Vector2d>& pts) {
        double sum = 0.0;
        for (const auto& pt : pts) sum += pt.y();
        return sum / static_cast<double>(pts.size());
    };
    const double dy = std::abs(mean_y(groups[0]) - mean_y(groups[1]));
    if (dy > 1e-6) {
        std::cerr << "expected equal-height left/right surfaces, got dy=" << dy
                  << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> sloped_pts;
    sloped_pts.reserve(131);
    for (int x = 0; x <= 100; ++x) {
        sloped_pts.emplace_back(static_cast<double>(x), 20.0 - 0.08 * x);
    }
    for (int x = 112; x <= 141; ++x) {
        sloped_pts.emplace_back(static_cast<double>(x), -25.0 + 0.02 * x);
    }
    std::sort(sloped_pts.begin(), sloped_pts.end(),
              [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                  if (a.x() == b.x()) return a.y() < b.y();
                  return a.x() < b.x();
              });
    auto slopes =
            pipeline::GapStepDetection::test_group_line_slopes(sloped_pts);
    auto sloped_groups =
            pipeline::GapStepDetection::test_group_by_derivative_dll(
                    sloped_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_sloped.png", sloped_pts,
                              sloped_groups);
    }
    if (slopes.size() != 2) {
        std::cerr << "expected two fitted surface lines, got " << slopes.size()
                  << std::endl;
        return 1;
    }
    std::sort(slopes.begin(), slopes.end());
    if (std::abs(slopes[0] - (-0.08)) > 0.02 ||
        std::abs(slopes[1] - 0.02) > 0.02) {
        std::cerr << "expected fitted slopes near -0.08 and 0.02, got "
                  << slopes[0] << ", " << slopes[1] << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> short_low_pts;
    short_low_pts.reserve(111);
    for (int x = 0; x <= 100; ++x) {
        short_low_pts.emplace_back(static_cast<double>(x), 30.0 - 0.06 * x);
    }
    for (int x = 112; x <= 121; ++x) {
        short_low_pts.emplace_back(static_cast<double>(x), -30.0 + 0.03 * x);
    }
    auto short_groups =
            pipeline::GapStepDetection::test_group_by_derivative_dll(
                    short_low_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_short_low.png", short_low_pts,
                              short_groups);
    }
    if (short_groups.size() != 2 || short_groups[0].empty() ||
        short_groups[1].size() < 4) {
        std::cerr << "expected a short right low surface to be preserved, "
                     "got "
                  << short_groups.size() << " groups";
        if (short_groups.size() > 1) {
            std::cerr << " with right group size " << short_groups[1].size();
        }
        std::cerr << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> vertical_edge_pts;
    vertical_edge_pts.reserve(120);
    for (int x = 0; x <= 100; ++x) {
        vertical_edge_pts.emplace_back(static_cast<double>(x), 35.0 - 0.05 * x);
    }
    for (int k = 0; k < 10; ++k) {
        vertical_edge_pts.emplace_back(103.0 + 0.08 * k, 28.0 - 6.0 * k);
    }
    for (int x = 112; x <= 119; ++x) {
        vertical_edge_pts.emplace_back(static_cast<double>(x),
                                       -28.0 + 0.04 * x);
    }
    auto vertical_groups =
            pipeline::GapStepDetection::test_group_by_derivative_dll(
                    vertical_edge_pts);
    auto vertical_slopes = pipeline::GapStepDetection::test_group_line_slopes(
            vertical_edge_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_vertical_edge.png",
                              vertical_edge_pts, vertical_groups);
    }
    if (vertical_groups.size() != 2 || vertical_slopes.size() != 2) {
        std::cerr << "expected two surfaces around the vertical edge, got "
                  << vertical_groups.size() << " groups and "
                  << vertical_slopes.size() << " slopes" << std::endl;
        return 1;
    }
    for (double slope : vertical_slopes) {
        if (std::abs(slope) > 0.2) {
            std::cerr << "vertical edge polluted surface fitting, slope="
                      << slope << std::endl;
            return 1;
        }
    }

    std::vector<Eigen::Vector2d> valley_pts;
    valley_pts.reserve(150);
    for (int x = 0; x <= 100; ++x) {
        valley_pts.emplace_back(static_cast<double>(x), 5.0 - 0.02 * x);
    }
    for (int k = 0; k < 12; ++k) {
        valley_pts.emplace_back(104.0 + 0.05 * k, 3.0 - 5.0 * k);
    }
    for (int x = 112; x <= 118; ++x) {
        valley_pts.emplace_back(static_cast<double>(x), -58.0);
    }
    for (int x = 124; x <= 150; ++x) {
        valley_pts.emplace_back(static_cast<double>(x),
                                -42.0 + 0.04 * (x - 124));
    }
    auto valley_groups =
            pipeline::GapStepDetection::test_group_by_derivative_dll(
                    valley_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_valley_reject.png", valley_pts,
                              valley_groups);
    }
    if (valley_groups.size() != 2 || valley_groups[1].empty()) {
        std::cerr << "expected right support surface with valley present"
                  << std::endl;
        return 1;
    }
    auto [right_min_it, right_max_it] = std::minmax_element(
            valley_groups[1].begin(), valley_groups[1].end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.x() < b.x();
            });
    if (right_min_it->x() < 120.0 || right_max_it->x() < 145.0) {
        std::cerr << "right support should reject valley short plane, got x=["
                  << right_min_it->x() << ", " << right_max_it->x() << "]"
                  << std::endl;
        return 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--self-test") {
        return run_self_test(argc > 2 ? argv[2] : nullptr);
    }

    const char* input_path = (argc > 1) ? argv[1] : DEFAULT_PLY;
    const char* debug_dir = nullptr;
    Eigen::Vector3d tiff_ratio(1, 1, 100);

    if (has_tiff_ext(input_path)) {
        // TIFF: args = <tiff> [ratio] [debug_dir]
        if (argc > 2 && parse_ratio(argv[2], tiff_ratio)) {
            debug_dir = (argc > 3) ? argv[3] : DEFAULT_DEBUG;
        } else {
            debug_dir = (argc > 2) ? argv[2] : DEFAULT_DEBUG;
        }
    } else {
        // PLY: args = <ply> [debug_dir]
        debug_dir = (argc > 2) ? argv[2] : DEFAULT_DEBUG;
    }

    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();

    if (has_tiff_ext(input_path)) {
        std::cout << "Reading TIFF: " << input_path << " ratio=("
                  << tiff_ratio.x() << "," << tiff_ratio.y() << ","
                  << tiff_ratio.z() << ")" << std::endl;
        core::converter::tiff_to_pointcloud(input_path, pointcloud, tiff_ratio,
                                            false);
    } else if (has_ply_ext(input_path)) {
        utility::read_ply(input_path, pointcloud);
    } else {
        std::cerr << "Unsupported file format. Expected .ply or .tiff/.tif"
                  << std::endl;
        return 1;
    }

    // --- 算法参数 ---
    double step_height = 0;  // [输出] 检测到的台阶高度
    double step_width = 0;   // [输出] 检测到的台阶宽度
    double height_threshold =
            1;        // Deprecated: kept for legacy API compatibility.
    bool LHT = true;  // true: 左高右低, false: 左低右高
    Eigen::Vector3d transformation_matrix = Eigen::Vector3d(1, 1, 1);
    std::vector<std::vector<double>> temp_res;
    temp_res.resize(2);
    std::string debug_path(debug_dir);  // debug 输出目录
    bool debug_mode = true;             // 开启则输出 debug 图像

    pipeline::GapStepDetection::detect_gap_step_dll_plot2(
            pointcloud, transformation_matrix, step_height, step_width,
            height_threshold, temp_res, debug_path, LHT, debug_mode);

    wait_for_key();
    return 0;
}

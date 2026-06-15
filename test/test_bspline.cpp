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
//   ./bspline_test /data/scan.tiff 1,1,100 /tmp/bspline_out/ --no-debug

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <string.h>  // _stricmp
#else
#include <strings.h>  // strcasecmp
#endif

#include <opencv2/opencv.hpp>

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
#ifdef _WIN32
    return _stricmp(ext, ".tiff") == 0 || _stricmp(ext, ".tif") == 0;
#else
    return strcasecmp(ext, ".tiff") == 0 || strcasecmp(ext, ".tif") == 0;
#endif
}

static bool has_ply_ext(const char* path) {
    const char* ext = strrchr(path, '.');
    if (!ext) return false;
#ifdef _WIN32
    return _stricmp(ext, ".ply") == 0;
#else
    return strcasecmp(ext, ".ply") == 0;
#endif
}

static bool parse_ratio(const char* str, Eigen::Vector3d& ratio) {
    return sscanf(str, "%lf,%lf,%lf", &ratio.x(), &ratio.y(), &ratio.z()) == 3;
}

static bool has_arg(int argc, char** argv, const char* needle) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == needle) return true;
    }
    return false;
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
    auto endpoint_slope = [](const std::vector<Eigen::Vector2d>& pts) {
        auto [min_it, max_it] = std::minmax_element(
                pts.begin(), pts.end(),
                [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                    return a.x() < b.x();
                });
        const double dx = max_it->x() - min_it->x();
        return std::abs(dx) < 1e-12 ? 0.0 : (max_it->y() - min_it->y()) / dx;
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

    std::vector<Eigen::Vector2d> u_shape_pts;
    u_shape_pts.reserve(160);
    for (int x = 0; x <= 100; ++x) {
        u_shape_pts.emplace_back(static_cast<double>(x), 10.0 - 0.03 * x);
    }
    for (int k = 0; k < 12; ++k) {
        u_shape_pts.emplace_back(104.0 + 0.05 * k, 7.0 - 5.0 * k);
    }
    for (int x = 122; x <= 126; ++x) {
        const double d = static_cast<double>(x - 130);
        u_shape_pts.emplace_back(static_cast<double>(x), -62.0 + 0.22 * d * d);
    }
    for (int x = 127; x <= 133; ++x) {
        const double d = static_cast<double>(x - 130);
        u_shape_pts.emplace_back(static_cast<double>(x), -62.0 + 0.22 * d * d);
    }
    for (int x = 134; x <= 160; ++x) {
        u_shape_pts.emplace_back(static_cast<double>(x),
                                 -48.0 + 0.03 * (x - 134));
    }
    auto u_shape_groups =
            pipeline::GapStepDetection::test_filtered_groups_dll(u_shape_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_u_shape_bridge.png",
                              u_shape_pts, u_shape_groups);
    }
    if (u_shape_groups.size() != 2 || u_shape_groups[1].empty()) {
        std::cerr << "expected right support after U-shaped valley"
                  << std::endl;
        return 1;
    }
    auto [u_right_min_it, u_right_max_it] = std::minmax_element(
            u_shape_groups[1].begin(), u_shape_groups[1].end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.x() < b.x();
            });
    if (u_right_min_it->x() < 134.0 || u_right_max_it->x() < 155.0) {
        std::cerr << "right support should not bridge across U-shaped valley, "
                     "got x=["
                  << u_right_min_it->x() << ", " << u_right_max_it->x() << "]"
                  << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> real_like_u_pts;
    real_like_u_pts.reserve(500);
    for (int x = 0; x <= 420; ++x) {
        real_like_u_pts.emplace_back(static_cast<double>(x),
                                     -3344.0 - 0.11 * x);
    }
    for (int x = 421; x <= 430; ++x) {
        const double t = static_cast<double>(x - 421) / 9.0;
        real_like_u_pts.emplace_back(static_cast<double>(x),
                                     -3390.0 * (1.0 - t) + -3584.0 * t);
    }
    for (int x = 431; x <= 439; ++x) {
        const double d = static_cast<double>(x - 439);
        real_like_u_pts.emplace_back(static_cast<double>(x),
                                     -3590.0 + 0.08 * d * d);
    }
    for (int x = 440; x <= 460; ++x) {
        real_like_u_pts.emplace_back(static_cast<double>(x),
                                     -3590.0 + 1.35 * (x - 440));
    }
    for (int x = 461; x <= 475; ++x) {
        real_like_u_pts.emplace_back(static_cast<double>(x),
                                     -3562.0 - 0.18 * (x - 461));
    }
    auto real_like_groups =
            pipeline::GapStepDetection::test_filtered_groups_dll(
                    real_like_u_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_real_like_u.png",
                              real_like_u_pts, real_like_groups);
    }
    if (real_like_groups.size() != 2 || real_like_groups[1].empty()) {
        std::cerr << "expected right support on real-like U-shaped slice"
                  << std::endl;
        return 1;
    }
    auto [real_right_min_it, real_right_max_it] = std::minmax_element(
            real_like_groups[1].begin(), real_like_groups[1].end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.x() < b.x();
            });
    if (real_right_min_it->x() < 455.0 || real_right_max_it->x() < 468.0) {
        std::cerr << "real-like right support should use the top platform, "
                     "got x=["
                  << real_right_min_it->x() << ", " << real_right_max_it->x()
                  << "]" << std::endl;
        return 1;
    }
    const double real_right_slope = endpoint_slope(real_like_groups[1]);
    if (std::abs(real_right_slope) > 1.0) {
        std::cerr << "real-like right support should be a flat platform, "
                     "slope="
                  << real_right_slope << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> z_scaled_transition_pts;
    z_scaled_transition_pts.reserve(real_like_u_pts.size());
    constexpr double kZScaleStress = 40.0;
    for (const auto& pt : real_like_u_pts) {
        z_scaled_transition_pts.emplace_back(pt.x(), pt.y() * kZScaleStress);
    }
    auto z_scaled_transition_groups =
            pipeline::GapStepDetection::test_fast_path_detect_platforms(
                    z_scaled_transition_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_z_scaled_transition.png",
                              z_scaled_transition_pts,
                              z_scaled_transition_groups);
    }
    if (z_scaled_transition_groups.size() != 2 ||
        z_scaled_transition_groups[0].empty() ||
        z_scaled_transition_groups[1].empty()) {
        std::cerr << "fast path should keep credible references when z scale "
                     "is high"
                  << std::endl;
        return 1;
    }
    auto [z_scaled_right_min_it, z_scaled_right_max_it] =
            std::minmax_element(
                    z_scaled_transition_groups[1].begin(),
                    z_scaled_transition_groups[1].end(),
                    [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                        return a.x() < b.x();
                    });
    const double z_scaled_right_slope =
            endpoint_slope(z_scaled_transition_groups[1]);
    if (z_scaled_right_min_it->x() < 455.0 ||
        z_scaled_right_max_it->x() < 468.0 ||
        std::abs(z_scaled_right_slope) > 12.0) {
        std::cerr << "z-scaled fast path should still use the top right "
                     "support, got x=["
                  << z_scaled_right_min_it->x() << ", "
                  << z_scaled_right_max_it->x() << "] slope="
                  << z_scaled_right_slope << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> low_z_sloped_gap_pts;
    low_z_sloped_gap_pts.reserve(210);
    for (int x = 0; x <= 110; ++x) {
        low_z_sloped_gap_pts.emplace_back(static_cast<double>(x),
                                          20.0 - 0.004 * x);
    }
    for (int x = 130; x <= 165; ++x) {
        low_z_sloped_gap_pts.emplace_back(static_cast<double>(x),
                                          19.0 - 0.18 * (x - 130));
    }
    for (int x = 166; x <= 235; ++x) {
        low_z_sloped_gap_pts.emplace_back(static_cast<double>(x),
                                          12.7 + 0.004 * (x - 166));
    }
    auto low_z_sloped_gap_groups =
            pipeline::GapStepDetection::test_fast_path_detect_platforms(
                    low_z_sloped_gap_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_low_z_sloped_gap.png",
                              low_z_sloped_gap_pts,
                              low_z_sloped_gap_groups);
    }
    if (low_z_sloped_gap_groups.size() != 2 ||
        low_z_sloped_gap_groups[1].empty()) {
        std::cerr << "low-z sloped-gap case should still find the real right "
                     "support"
                  << std::endl;
        return 1;
    }
    auto [low_z_right_min_it, low_z_right_max_it] =
            std::minmax_element(
                    low_z_sloped_gap_groups[1].begin(),
                    low_z_sloped_gap_groups[1].end(),
                    [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                        return a.x() < b.x();
                    });
    const double low_z_right_slope =
            endpoint_slope(low_z_sloped_gap_groups[1]);
    if (low_z_right_min_it->x() < 164.0 ||
        std::abs(low_z_right_slope) > 0.05) {
        std::cerr << "low-z sloped gap should reject the diagonal transition "
                     "as a reference, got right x=["
                  << low_z_right_min_it->x() << ", "
                  << low_z_right_max_it->x() << "] slope="
                  << low_z_right_slope << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> scaled_sloped_gap_pts;
    scaled_sloped_gap_pts.reserve(low_z_sloped_gap_pts.size());
    constexpr double kSlopedGapZScale = 40.0;
    for (const auto& pt : low_z_sloped_gap_pts) {
        scaled_sloped_gap_pts.emplace_back(pt.x(), pt.y() * kSlopedGapZScale);
    }
    auto scaled_sloped_gap_groups =
            pipeline::GapStepDetection::test_fast_path_detect_platforms(
                    scaled_sloped_gap_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_scaled_sloped_gap.png",
                              scaled_sloped_gap_pts,
                              scaled_sloped_gap_groups);
    }
    if (scaled_sloped_gap_groups.size() != 2 ||
        scaled_sloped_gap_groups[1].empty()) {
        std::cerr << "scaled sloped-gap case should still find the real right "
                     "support"
                  << std::endl;
        return 1;
    }
    auto [scaled_gap_right_min_it, scaled_gap_right_max_it] =
            std::minmax_element(
                    scaled_sloped_gap_groups[1].begin(),
                    scaled_sloped_gap_groups[1].end(),
                    [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                        return a.x() < b.x();
                    });
    const double scaled_gap_right_slope =
            endpoint_slope(scaled_sloped_gap_groups[1]);
    if (scaled_gap_right_min_it->x() < 164.0 ||
        std::abs(scaled_gap_right_slope) > 0.05 * kSlopedGapZScale) {
        std::cerr << "scaled sloped gap should reject the diagonal transition "
                     "as a reference, got right x=["
                  << scaled_gap_right_min_it->x() << ", "
                  << scaled_gap_right_max_it->x() << "] slope="
                  << scaled_gap_right_slope << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> fast_path_short_tail_pts;
    fast_path_short_tail_pts.reserve(480);
    for (int x = 0; x <= 420; ++x) {
        fast_path_short_tail_pts.emplace_back(static_cast<double>(x),
                                              100.0 - 0.04 * x);
    }
    for (int x = 421; x <= 439; ++x) {
        const double t = static_cast<double>(x - 421) / 18.0;
        fast_path_short_tail_pts.emplace_back(static_cast<double>(x),
                                              83.0 * (1.0 - t) + -45.0 * t);
    }
    for (int x = 440; x <= 460; ++x) {
        fast_path_short_tail_pts.emplace_back(static_cast<double>(x),
                                              -45.0 + 0.04 * (x - 440));
    }
    for (int x = 461; x <= 469; ++x) {
        fast_path_short_tail_pts.emplace_back(static_cast<double>(x),
                                              -42.0 + 3.0 * (x - 461));
    }
    for (int x = 470; x <= 475; ++x) {
        fast_path_short_tail_pts.emplace_back(static_cast<double>(x),
                                              -16.0 + 0.01 * (x - 470));
    }
    auto fast_path_groups =
            pipeline::GapStepDetection::test_fast_path_detect_platforms(
                    fast_path_short_tail_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_fast_path_short_tail.png",
                              fast_path_short_tail_pts, fast_path_groups);
    }
    if (!fast_path_groups.empty()) {
        auto [fast_right_min_it, fast_right_max_it] = std::minmax_element(
                fast_path_groups[1].begin(), fast_path_groups[1].end(),
                [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                    return a.x() < b.x();
                });
        const double right_span =
                fast_right_max_it->x() - fast_right_min_it->x();
        if (fast_right_min_it->x() > 462.0 || right_span < 12.0) {
            std::cerr << "fast path should not accept a short terminal "
                         "artifact as the right platform, got x=["
                      << fast_right_min_it->x() << ", "
                      << fast_right_max_it->x() << "]" << std::endl;
            return 1;
        }
    }

    std::vector<Eigen::Vector2d> fast_path_left_near_edge_pts;
    fast_path_left_near_edge_pts.reserve(500);
    for (int x = 0; x <= 60; ++x) {
        fast_path_left_near_edge_pts.emplace_back(static_cast<double>(x),
                                                  100.0 - 0.02 * x);
    }
    for (int x = 61; x <= 80; ++x) {
        fast_path_left_near_edge_pts.emplace_back(static_cast<double>(x),
                                                  98.8 - 0.9 * (x - 60));
    }
    for (int x = 81; x <= 260; ++x) {
        fast_path_left_near_edge_pts.emplace_back(static_cast<double>(x),
                                                  81.0 - 0.03 * (x - 81));
    }
    for (int x = 261; x <= 300; ++x) {
        fast_path_left_near_edge_pts.emplace_back(static_cast<double>(x),
                                                  75.6 - 0.55 * (x - 260));
    }
    for (int x = 301; x <= 420; ++x) {
        fast_path_left_near_edge_pts.emplace_back(static_cast<double>(x),
                                                  53.6 - 0.04 * (x - 301));
    }
    for (int x = 421; x <= 439; ++x) {
        const double t = static_cast<double>(x - 421) / 18.0;
        fast_path_left_near_edge_pts.emplace_back(static_cast<double>(x),
                                                  48.8 * (1.0 - t) + -45.0 * t);
    }
    for (int x = 440; x <= 475; ++x) {
        fast_path_left_near_edge_pts.emplace_back(static_cast<double>(x),
                                                  -45.0 + 0.03 * (x - 440));
    }
    auto fast_left_edge_groups =
            pipeline::GapStepDetection::test_fast_path_detect_platforms(
                    fast_path_left_near_edge_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_fast_path_left_near_edge.png",
                              fast_path_left_near_edge_pts,
                              fast_left_edge_groups);
    }
    if (fast_left_edge_groups.size() != 2 || fast_left_edge_groups[0].empty()) {
        std::cerr << "expected fast path to find left and right supports for "
                     "edge-adjacent selection"
                  << std::endl;
        return 1;
    }
    auto [fast_left_min_it, fast_left_max_it] = std::minmax_element(
            fast_left_edge_groups[0].begin(), fast_left_edge_groups[0].end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.x() < b.x();
            });
    if (fast_left_max_it->x() < 390.0 || fast_left_min_it->x() < 280.0) {
        std::cerr << "fast path left support should be the local surface near "
                     "the step edge, got x=["
                  << fast_left_min_it->x() << ", " << fast_left_max_it->x()
                  << "]" << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> short_valley_pts;
    short_valley_pts.reserve(150);
    for (int x = 0; x <= 53; ++x) {
        short_valley_pts.emplace_back(static_cast<double>(x), 48.0 - 0.018 * x);
    }
    for (int x = 54; x <= 63; ++x) {
        short_valley_pts.emplace_back(static_cast<double>(x),
                                      47.0 - 1.15 * (x - 53));
    }
    for (int x = 64; x <= 72; ++x) {
        short_valley_pts.emplace_back(static_cast<double>(x),
                                      35.5 - 0.025 * (x - 64));
    }
    for (int x = 73; x <= 81; ++x) {
        short_valley_pts.emplace_back(static_cast<double>(x),
                                      35.0 + 2.0 * (x - 73));
    }
    for (int x = 82; x <= 145; ++x) {
        short_valley_pts.emplace_back(static_cast<double>(x),
                                      53.5 + 0.011 * (x - 82));
    }
    auto short_valley_groups =
            pipeline::GapStepDetection::test_fast_path_detect_platforms(
                    short_valley_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_short_valley_transition.png",
                              short_valley_pts, short_valley_groups);
    }
    if (short_valley_groups.size() != 2 || short_valley_groups[0].empty() ||
        short_valley_groups[1].empty()) {
        std::cerr << "short-valley transition should still find two reference "
                     "surfaces"
                  << std::endl;
        return 1;
    }
    auto [short_valley_left_min_it, short_valley_left_max_it] =
            std::minmax_element(
                    short_valley_groups[0].begin(),
                    short_valley_groups[0].end(),
                    [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                        return a.x() < b.x();
                    });
    auto [short_valley_right_min_it, short_valley_right_max_it] =
            std::minmax_element(
                    short_valley_groups[1].begin(),
                    short_valley_groups[1].end(),
                    [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                        return a.x() < b.x();
                    });
    if (short_valley_left_max_it->x() > 58.0 ||
        short_valley_right_min_it->x() < 78.0) {
        std::cerr << "short valley floor must not be selected as a reference "
                     "surface, got left x=["
                  << short_valley_left_min_it->x() << ", "
                  << short_valley_left_max_it->x() << "] right x=["
                  << short_valley_right_min_it->x() << ", "
                  << short_valley_right_max_it->x() << "]" << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> long_valley_floor_pts;
    long_valley_floor_pts.reserve(220);
    for (int x = 0; x <= 70; ++x) {
        long_valley_floor_pts.emplace_back(static_cast<double>(x),
                                           50.0 - 0.01 * x);
    }
    for (int x = 71; x <= 90; ++x) {
        const double t = static_cast<double>(x - 71) / 19.0;
        long_valley_floor_pts.emplace_back(
                static_cast<double>(x), 49.0 * (1.0 - t) + 12.0 * t);
    }
    for (int x = 91; x <= 125; ++x) {
        long_valley_floor_pts.emplace_back(static_cast<double>(x),
                                           12.0 + 0.005 * (x - 91));
    }
    for (int x = 126; x <= 132; ++x) {
        const double t = static_cast<double>(x - 126) / 6.0;
        long_valley_floor_pts.emplace_back(
                static_cast<double>(x), 12.2 * (1.0 - t) + 53.0 * t);
    }
    for (int x = 133; x <= 205; ++x) {
        long_valley_floor_pts.emplace_back(static_cast<double>(x),
                                           53.0 - 0.006 * (x - 133));
    }
    auto long_valley_floor_groups =
            pipeline::GapStepDetection::test_fast_path_detect_platforms(
                    long_valley_floor_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_long_valley_floor.png",
                              long_valley_floor_pts,
                              long_valley_floor_groups);
    }
    if (long_valley_floor_groups.size() != 2 ||
        long_valley_floor_groups[0].empty() ||
        long_valley_floor_groups[1].empty()) {
        std::cerr << "long valley floor should still find two outer support "
                     "surfaces"
                  << std::endl;
        return 1;
    }
    auto [long_valley_left_min_it, long_valley_left_max_it] =
            std::minmax_element(
                    long_valley_floor_groups[0].begin(),
                    long_valley_floor_groups[0].end(),
                    [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                        return a.x() < b.x();
                    });
    auto [long_valley_right_min_it, long_valley_right_max_it] =
            std::minmax_element(
                    long_valley_floor_groups[1].begin(),
                    long_valley_floor_groups[1].end(),
                    [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                        return a.x() < b.x();
                    });
    if (long_valley_left_max_it->x() > 76.0 ||
        long_valley_right_min_it->x() < 132.0) {
        std::cerr << "long valley floor must not block the outer platform "
                     "pair, got left x=["
                  << long_valley_left_min_it->x() << ", "
                  << long_valley_left_max_it->x() << "] right x=["
                  << long_valley_right_min_it->x() << ", "
                  << long_valley_right_max_it->x() << "]" << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> missing_gap_pts;
    missing_gap_pts.reserve(460);
    for (int x = 0; x <= 420; ++x) {
        missing_gap_pts.emplace_back(static_cast<double>(x), 80.0 - 0.04 * x);
    }
    for (int x = 440; x <= 475; ++x) {
        missing_gap_pts.emplace_back(static_cast<double>(x),
                                     -45.0 + 0.03 * (x - 440));
    }
    auto missing_gap_groups =
            pipeline::GapStepDetection::test_fast_path_detect_platforms(
                    missing_gap_pts);
    if (debug_dir) {
        std::string base(debug_dir);
        if (!base.empty() && base.back() != '/') base += "/";
        write_self_test_debug(base + "self_test_missing_gap.png",
                              missing_gap_pts, missing_gap_groups);
    }
    if (missing_gap_groups.size() != 2 || missing_gap_groups[0].empty() ||
        missing_gap_groups[1].empty()) {
        std::cerr << "missing-gap fast path should find both visible support "
                     "surfaces"
                  << std::endl;
        return 1;
    }
    auto [missing_left_min_it, missing_left_max_it] = std::minmax_element(
            missing_gap_groups[0].begin(), missing_gap_groups[0].end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.x() < b.x();
            });
    auto [missing_right_min_it, missing_right_max_it] = std::minmax_element(
            missing_gap_groups[1].begin(), missing_gap_groups[1].end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.x() < b.x();
            });
    const double missing_right_span =
            missing_right_max_it->x() - missing_right_min_it->x();
    if (missing_left_max_it->x() < 415.0 || missing_right_min_it->x() > 445.0 ||
        missing_right_span < 20.0) {
        std::cerr << "missing-gap supports should be adjacent to the no-data "
                     "gap, got left x=["
                  << missing_left_min_it->x() << ", "
                  << missing_left_max_it->x() << "] right x=["
                  << missing_right_min_it->x() << ", "
                  << missing_right_max_it->x() << "]" << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> missing_gap_left_low_pts;
    missing_gap_left_low_pts.reserve(460);
    for (int x = 0; x <= 420; ++x) {
        missing_gap_left_low_pts.emplace_back(static_cast<double>(x),
                                              5.0 + 0.01 * x);
    }
    for (int x = 440; x <= 475; ++x) {
        missing_gap_left_low_pts.emplace_back(static_cast<double>(x),
                                              20.0 + 0.02 * (x - 440));
    }
    auto missing_left_low_groups =
            pipeline::GapStepDetection::test_fast_path_detect_platforms(
                    missing_gap_left_low_pts);
    if (missing_left_low_groups.size() != 2 ||
        missing_left_low_groups[0].empty() ||
        missing_left_low_groups[1].empty()) {
        std::cerr << "missing-gap fast path should handle left-low/right-high "
                     "surfaces"
                  << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> missing_gap_equal_height_pts;
    missing_gap_equal_height_pts.reserve(460);
    for (int x = 0; x <= 420; ++x) {
        missing_gap_equal_height_pts.emplace_back(static_cast<double>(x), 12.0);
    }
    for (int x = 440; x <= 475; ++x) {
        missing_gap_equal_height_pts.emplace_back(static_cast<double>(x), 12.0);
    }
    auto missing_equal_groups =
            pipeline::GapStepDetection::test_fast_path_detect_platforms(
                    missing_gap_equal_height_pts);
    if (missing_equal_groups.size() != 2 || missing_equal_groups[0].empty() ||
        missing_equal_groups[1].empty()) {
        std::cerr << "missing-gap fast path should handle equal-height surfaces"
                  << std::endl;
        return 1;
    }
    auto [equal_left_min_it, equal_left_max_it] = std::minmax_element(
            missing_equal_groups[0].begin(), missing_equal_groups[0].end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.x() < b.x();
            });
    auto [equal_right_min_it, equal_right_max_it] = std::minmax_element(
            missing_equal_groups[1].begin(), missing_equal_groups[1].end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.x() < b.x();
            });
    std::vector<Eigen::Vector2d> equal_limits{
            *equal_left_min_it, *equal_left_max_it, *equal_right_min_it,
            *equal_right_max_it};
    auto equal_boundaries =
            pipeline::GapStepDetection::test_compute_step_boundaries(
                    missing_equal_groups[0], missing_equal_groups[1],
                    equal_limits);
    if (std::abs(equal_boundaries.second.y() - equal_boundaries.first.y()) >
        1e-6) {
        std::cerr
                << "equal-height missing-gap surfaces should measure near zero "
                   "signed height, got "
                << equal_boundaries.second.y() - equal_boundaries.first.y()
                << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> isolated_missing_pts;
    isolated_missing_pts.reserve(100);
    for (int x = 0; x <= 100; ++x) {
        if (x == 50 || x == 51) continue;
        isolated_missing_pts.emplace_back(static_cast<double>(x),
                                          30.0 - 0.02 * x);
    }
    auto isolated_missing_groups =
            pipeline::GapStepDetection::test_fast_path_detect_platforms(
                    isolated_missing_pts);
    if (!isolated_missing_groups.empty()) {
        std::cerr << "one or two isolated missing pixels should not be treated "
                     "as a measurable missing gap"
                  << std::endl;
        return 1;
    }

    std::vector<Eigen::Vector2d> endpoint_left_pts;
    std::vector<Eigen::Vector2d> endpoint_right_pts;
    for (int x = 300; x <= 400; ++x) {
        endpoint_left_pts.emplace_back(static_cast<double>(x),
                                       60.0 - 0.04 * (x - 300));
    }
    for (int x = 445; x <= 470; ++x) {
        endpoint_right_pts.emplace_back(static_cast<double>(x),
                                        -45.0 + 0.03 * (x - 445));
    }
    std::vector<Eigen::Vector2d> endpoint_limits{
            endpoint_left_pts.front(), Eigen::Vector2d(420.0, 55.2),
            Eigen::Vector2d(440.0, -45.0), endpoint_right_pts.back()};
    auto projected_boundaries =
            pipeline::GapStepDetection::test_compute_step_boundaries(
                    endpoint_left_pts, endpoint_right_pts, endpoint_limits);
    if (std::abs(projected_boundaries.first.x() - 420.0) > 1e-6 ||
        std::abs(projected_boundaries.second.x() - 440.0) > 1e-6) {
        std::cerr << "measurement boundaries should use limit points projected "
                     "onto fitted support lines, got x=["
                  << projected_boundaries.first.x() << ", "
                  << projected_boundaries.second.x() << "]" << std::endl;
        return 1;
    }

    {
        std::string invalid_tiff_dir =
                debug_dir ? std::string(debug_dir)
                          : utility::filesystem::GetTempDirectoryPath();
        if (!invalid_tiff_dir.empty() && invalid_tiff_dir.back() != '/')
            invalid_tiff_dir += "/";
        invalid_tiff_dir += "self_test_invalid_tiff/";
        utility::filesystem::MakeDirectoryHierarchy(invalid_tiff_dir);

        cv::Mat invalid_tiff(2, 4, CV_32FC1);
        invalid_tiff.at<float>(0, 0) = -3300.0f;
        invalid_tiff.at<float>(0, 1) = -3310.0f;
        invalid_tiff.at<float>(0, 2) = -21474836.0f;
        invalid_tiff.at<float>(0, 3) = -3320.0f;
        invalid_tiff.at<float>(1, 0) = -3330.0f;
        invalid_tiff.at<float>(1, 1) = -21474836.0f;
        invalid_tiff.at<float>(1, 2) = -3340.0f;
        invalid_tiff.at<float>(1, 3) = -3350.0f;
        const std::string invalid_tiff_path =
                invalid_tiff_dir + "invalid_height.tiff";
        if (!cv::imwrite(invalid_tiff_path, invalid_tiff)) {
            std::cerr << "failed to write invalid TIFF self-test input"
                      << std::endl;
            return 1;
        }

        auto invalid_cloud = std::make_shared<geometry::PointCloud>();
        core::converter::tiff_to_pointcloud(invalid_tiff_path, invalid_cloud,
                                            Eigen::Vector3d(1, 1, 100), false);
        for (const auto& pt : invalid_cloud->points_) {
            if (pt.z() < -1e8) {
                std::cerr << "TIFF invalid sentinel height leaked into "
                             "point cloud: z="
                          << pt.z() << std::endl;
                return 1;
            }
        }
        if (invalid_cloud->points_.size() != 6) {
            std::cerr << "expected 6 valid points after filtering invalid "
                         "TIFF pixels, got "
                      << invalid_cloud->points_.size() << std::endl;
            return 1;
        }
        if (invalid_cloud->source_point_count_ != 8 ||
            invalid_cloud->invalid_point_count_ != 2) {
            std::cerr << "expected TIFF metadata source=8 invalid=2, got "
                      << "source=" << invalid_cloud->source_point_count_
                      << " invalid=" << invalid_cloud->invalid_point_count_
                      << std::endl;
            return 1;
        }
    }

    {
        std::string ratio_tiff_dir =
                debug_dir ? std::string(debug_dir)
                          : utility::filesystem::GetTempDirectoryPath();
        if (!ratio_tiff_dir.empty() && ratio_tiff_dir.back() != '/')
            ratio_tiff_dir += "/";
        ratio_tiff_dir += "self_test_z_ratio_invariance/";
        utility::filesystem::MakeDirectoryHierarchy(ratio_tiff_dir);

        cv::Mat valley_tiff(24, 180, CV_32FC1);
        for (int y = 0; y < valley_tiff.rows; ++y) {
            float* row = valley_tiff.ptr<float>(y);
            for (int x = 0; x < valley_tiff.cols; ++x) {
                double z = 10.0;
                if (x >= 70 && x <= 78) {
                    const double t = static_cast<double>(x - 70) / 8.0;
                    z = 10.0 * (1.0 - t) + 7.0 * t;
                } else if (x >= 79 && x <= 87) {
                    const double t = static_cast<double>(x - 79) / 8.0;
                    z = 7.0 * (1.0 - t) + 10.0 * t;
                }
                row[x] = static_cast<float>(z);
            }
        }
        const std::string valley_tiff_path =
                ratio_tiff_dir + "equal_surface_valley.tiff";
        if (!cv::imwrite(valley_tiff_path, valley_tiff)) {
            std::cerr << "failed to write z-ratio invariance TIFF self-test"
                      << std::endl;
            return 1;
        }

        auto run_ratio_case = [&](double z_ratio, double& height,
                                  double& width) {
            auto cloud = std::make_shared<geometry::PointCloud>();
            core::converter::tiff_to_pointcloud(
                    valley_tiff_path, cloud, Eigen::Vector3d(1, 1, z_ratio),
                    false);
            Eigen::Vector3d transformation_matrix(1, 1, 1);
            double height_threshold = 1.0;
            std::vector<std::vector<double>> temp_res(2);
            std::string case_debug_path = ratio_tiff_dir;
            const bool ok = pipeline::GapStepDetection::detect_gap_step_dll_plot2(
                    cloud, transformation_matrix, height, width,
                    height_threshold, temp_res, case_debug_path, true, false);
            if (!ok) {
                std::cerr << "z-ratio invariance detect failed for z_ratio="
                          << z_ratio << std::endl;
                return false;
            }
            height /= z_ratio;
            return true;
        };

        double h1 = 0.0;
        double w1 = 0.0;
        double h40 = 0.0;
        double w40 = 0.0;
        if (!run_ratio_case(1.0, h1, w1) ||
            !run_ratio_case(40.0, h40, w40)) {
            return 1;
        }
        if (w1 <= 0.0 || w40 <= 0.0 || std::abs(w1 - w40) > 0.75 ||
            std::abs(h1 - h40) > 0.01) {
            std::cerr << "z-ratio invariance failed: z=1 width=" << w1
                      << " height=" << h1 << ", z=40 width=" << w40
                      << " height=" << h40 << std::endl;
            return 1;
        }
    }

    std::string mark_debug_dir =
            debug_dir ? std::string(debug_dir)
                      : utility::filesystem::GetTempDirectoryPath();
    if (!mark_debug_dir.empty() && mark_debug_dir.back() != '/')
        mark_debug_dir += "/";
    mark_debug_dir += "self_test_mark_rejected/";
    if (pipeline::GapStepDetection::test_mark_rejected_debug_images(
                mark_debug_dir) != 0) {
        return 1;
    }

    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--self-test") {
        return run_self_test(argc > 2 ? argv[2] : nullptr);
    }
    if (argc > 1 && std::string(argv[1]) == "--self-test-3d") {
        return pipeline::GapStepDetection::test_3d_consistency_filter(
                argc > 2 ? argv[2] : "");
    }

    const char* input_path = (argc > 1) ? argv[1] : DEFAULT_PLY;
    const char* debug_dir = nullptr;
    Eigen::Vector3d tiff_ratio(1, 1, 100);
    bool debug_mode = !has_arg(argc, argv, "--no-debug");

    if (has_tiff_ext(input_path)) {
        // TIFF: args = <tiff> [ratio] [debug_dir]
        if (argc > 2 && parse_ratio(argv[2], tiff_ratio)) {
            debug_dir = (argc > 3 && std::string(argv[3]) != "--no-debug")
                                ? argv[3]
                                : DEFAULT_DEBUG;
        } else {
            debug_dir = (argc > 2 && std::string(argv[2]) != "--no-debug")
                                ? argv[2]
                                : DEFAULT_DEBUG;
        }
    } else {
        // PLY: args = <ply> [debug_dir]
        debug_dir = (argc > 2 && std::string(argv[2]) != "--no-debug")
                            ? argv[2]
                            : DEFAULT_DEBUG;
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
    double height_threshold = 1;  // Z offset used for threshold-width measure.
    bool LHT = true;  // true: 左高右低, false: 左低右高
    Eigen::Vector3d transformation_matrix = Eigen::Vector3d(1, 1, 1);
    std::vector<std::vector<double>> temp_res;
    temp_res.resize(2);
    std::string debug_path(debug_dir);  // debug 输出目录

    pipeline::GapStepDetection::detect_gap_step_dll_plot2(
            pointcloud, transformation_matrix, step_height, step_width,
            height_threshold, temp_res, debug_path, LHT, debug_mode);

    double step_height_ =
            (step_height / transformation_matrix.z()) / tiff_ratio.z();
    double step_width_ =
            (step_width / transformation_matrix.x()) / tiff_ratio.x();
    std::cout << "step_height: " << step_height_ << std::endl;
    std::cout << "step_width: " << step_width_ << std::endl;
    wait_for_key();
    return 0;
}

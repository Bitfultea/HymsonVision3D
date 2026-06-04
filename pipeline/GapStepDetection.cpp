#include "GapStepDetection.h"

#include <math.h>

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "Converter.h"
#include "Curvature.h"
#include "Feature.h"
#include "FileSystem.h"
#include "MathTool.h"
#include "PlaneDetection.h"

namespace hymson3d {
namespace pipeline {

namespace {
constexpr double kMinSurfaceSlopeLimit = 0.35;
constexpr double kMaxSurfaceSlopeLimit = 20.0;
constexpr double kMaxPlatformSlopeLimit = 35.0;
constexpr double kMinStableSurfaceSpan = 6.0;
constexpr int kMaxSurfaceGap = 2;
constexpr size_t kMinSurfacePoints = 4;

using ProfileClock = std::chrono::steady_clock;

bool profile_enabled() {
    static const bool enabled =
            std::getenv("HYMSON3D_PROFILE_BSPLINE") != nullptr;
    return enabled;
}

long long elapsed_us(ProfileClock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
                   ProfileClock::now() - start)
            .count();
}

void ensure_trailing_path_separator(std::string& path) {
    if (path.empty()) return;
    const char last = path.back();
    if (last != '/' && last != '\\') path += "/";
}

class ProfileScope {
public:
    explicit ProfileScope(const char* name) : name_(name) {
        if (profile_enabled()) start_ = ProfileClock::now();
    }

    ~ProfileScope() {
        if (!profile_enabled()) return;
        std::cerr << "[profile] " << name_ << ": "
                  << elapsed_us(start_) / 1000.0 << " ms" << std::endl;
    }

private:
    const char* name_;
    ProfileClock::time_point start_;
};

struct SurfaceCandidate {
    std::vector<Eigen::Vector2d> points;
    std::pair<Eigen::Vector2d, Eigen::Vector2d> line;
    double x_min = 0.0;
    double x_max = 0.0;
    double x_center = 0.0;
    double y_center = 0.0;
    double span = 0.0;
    double rms = 0.0;
};

struct RightSurfaceSearch {
    std::vector<Eigen::Vector2d> search_region;
    std::vector<Eigen::Vector2d> vertical_points;
    std::vector<Eigen::Vector2d> valley_points;
    double edge_x = 0.0;
    double valley_x = 0.0;
};

struct SliceMeasurement {
    int index = -1;
    double width = 0.0;
    // Legacy/statistical height: always non-negative.
    double height = 0.0;
    double signed_height = 0.0;
    double height_abs = 0.0;
    size_t valid_points = 0;
    size_t expected_points = 0;
    double valid_ratio = 1.0;
    Eigen::Vector2d left_boundary =
            Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector2d right_boundary =
            Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::quiet_NaN());
    // 2D (x, z) surface points for left/right platform
    std::vector<Eigen::Vector2d> left_surface_pts;
    std::vector<Eigen::Vector2d> right_surface_pts;
    // 3D surface residuals after plane fitting (for filtering)
    double left_residual = 0.0;
    double right_residual = 0.0;
    bool accepted = false;
    std::string reject_reason;
};

using SliceLineSegments =
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>;

void set_measurement_height(SliceMeasurement& measurement,
                            double signed_height) {
    measurement.signed_height = signed_height;
    measurement.height_abs = std::abs(signed_height);
    measurement.height = measurement.height_abs;
}

std::vector<Eigen::Vector2d> filter_surface_group(
        const std::vector<Eigen::Vector2d>& points,
        bool prefer_stable_subsegment);
std::vector<SurfaceCandidate> collect_surface_candidates(
        const std::vector<Eigen::Vector2d>& sampled_pts);
std::vector<Eigen::Vector2d> select_right_platform_surface(
        const std::vector<Eigen::Vector2d>& points);
std::vector<Eigen::Vector2d> robust_line_fit_inliers(
        const std::vector<Eigen::Vector2d>& pts);
std::vector<std::vector<Eigen::Vector2d>> fast_path_detect_platforms(
        const std::vector<Eigen::Vector2d>& raw_pts,
        std::string* fallback_reason);

std::pair<Eigen::Vector2d, Eigen::Vector2d> invalid_corner() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return std::make_pair(Eigen::Vector2d(nan, nan), Eigen::Vector2d(nan, nan));
}

int adaptive_sample_count(size_t slice_size) {
    return std::clamp(static_cast<int>(slice_size * 3), 500, 1200);
}

double line_y_at_x(const std::pair<Eigen::Vector2d, Eigen::Vector2d>& line,
                   double x) {
    const double dx = line.second.x() - line.first.x();
    if (std::abs(dx) < 1e-12) return line.first.y();
    const double t = (x - line.first.x()) / dx;
    return line.first.y() + t * (line.second.y() - line.first.y());
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> fit_line_segment(
        const std::vector<Eigen::Vector2d>& pts) {
    double left_x = std::numeric_limits<double>::max();
    double right_x = -std::numeric_limits<double>::max();
    double sum_x = 0.0;
    double sum_y = 0.0;
    for (const auto& pt : pts) {
        left_x = std::min(left_x, pt.x());
        right_x = std::max(right_x, pt.x());
        sum_x += pt.x();
        sum_y += pt.y();
    }

    const double n = static_cast<double>(pts.size());
    const double mean_x = sum_x / n;
    const double mean_y = sum_y / n;
    double var_x = 0.0;
    double cov_xy = 0.0;
    for (const auto& pt : pts) {
        const double dx = pt.x() - mean_x;
        var_x += dx * dx;
        cov_xy += dx * (pt.y() - mean_y);
    }

    const double slope = (var_x > 1e-12) ? cov_xy / var_x : 0.0;
    const double intercept = mean_y - slope * mean_x;
    return std::make_pair(
            Eigen::Vector2d(left_x, slope * left_x + intercept),
            Eigen::Vector2d(right_x, slope * right_x + intercept));
}

std::pair<Eigen::Vector2d, Eigen::Vector2d> fit_line_segment_robust(
        const std::vector<Eigen::Vector2d>& pts) {
    return fit_line_segment(robust_line_fit_inliers(pts));
}

std::vector<Eigen::Vector2d> robust_line_fit_inliers(
        const std::vector<Eigen::Vector2d>& pts) {
    if (pts.size() < 8) return pts;

    std::vector<Eigen::Vector2d> inliers = pts;
    for (int iter = 0; iter < 3; ++iter) {
        auto line = fit_line_segment(inliers);
        std::vector<double> residuals;
        residuals.reserve(inliers.size());
        for (const auto& pt : inliers) {
            residuals.push_back(std::abs(pt.y() - line_y_at_x(line, pt.x())));
        }
        std::vector<double> sorted_residuals = residuals;
        std::sort(sorted_residuals.begin(), sorted_residuals.end());
        const size_t keep_count = std::max<size_t>(
                kMinSurfacePoints, sorted_residuals.size() * 7 / 10);
        const double threshold = sorted_residuals[keep_count - 1];

        std::vector<Eigen::Vector2d> trimmed;
        trimmed.reserve(keep_count);
        for (int i = 0; i < inliers.size(); ++i) {
            if (residuals[i] <= threshold) trimmed.push_back(inliers[i]);
        }
        if (trimmed.size() == inliers.size() ||
            trimmed.size() < kMinSurfacePoints) {
            break;
        }
        inliers.swap(trimmed);
    }
    return inliers;
}

double line_slope(const std::pair<Eigen::Vector2d, Eigen::Vector2d>& line) {
    const double dx = line.second.x() - line.first.x();
    if (std::abs(dx) < 1e-12) return 0.0;
    return (line.second.y() - line.first.y()) / dx;
}

double median_positive_x_step(const std::vector<Eigen::Vector2d>& pts) {
    std::vector<double> x_steps;
    x_steps.reserve(pts.size());
    for (int i = 0; i < pts.size() - 1; ++i) {
        const double dx = pts[i + 1].x() - pts[i].x();
        if (dx > 1e-12) x_steps.push_back(dx);
    }
    if (x_steps.empty()) return 0.0;
    std::nth_element(x_steps.begin(), x_steps.begin() + x_steps.size() / 2,
                     x_steps.end());
    return x_steps[x_steps.size() / 2];
}

double estimate_surface_slope_limit(const std::vector<Eigen::Vector2d>& pts) {
    std::vector<double> slopes;
    slopes.reserve(pts.size());
    for (int i = 0; i < pts.size() - 1; ++i) {
        const double dx = pts[i + 1].x() - pts[i].x();
        if (std::abs(dx) < 1e-12) continue;
        slopes.push_back(std::abs((pts[i + 1].y() - pts[i].y()) / dx));
    }
    if (slopes.empty()) return kMinSurfaceSlopeLimit;
    std::nth_element(slopes.begin(), slopes.begin() + slopes.size() / 2,
                     slopes.end());
    const double median_slope = slopes[slopes.size() / 2];
    return std::clamp(median_slope * 5.0, kMinSurfaceSlopeLimit,
                      kMaxSurfaceSlopeLimit);
}

SurfaceCandidate make_surface_candidate(
        const std::vector<Eigen::Vector2d>& points) {
    SurfaceCandidate candidate;
    candidate.points = points;
    candidate.line = fit_line_segment_robust(candidate.points);
    candidate.x_min = candidate.line.first.x();
    candidate.x_max = candidate.line.second.x();
    candidate.x_center = 0.5 * (candidate.x_min + candidate.x_max);
    candidate.span = candidate.x_max - candidate.x_min;

    double sum_y = 0.0;
    double sum_sq_residual = 0.0;
    for (const auto& pt : candidate.points) {
        sum_y += pt.y();
        const double residual = pt.y() - line_y_at_x(candidate.line, pt.x());
        sum_sq_residual += residual * residual;
    }
    candidate.y_center = sum_y / static_cast<double>(candidate.points.size());
    candidate.rms = std::sqrt(sum_sq_residual /
                              static_cast<double>(candidate.points.size()));
    return candidate;
}

std::vector<Eigen::Vector2d> expand_surface_around_candidate(
        const std::vector<Eigen::Vector2d>& region,
        const SurfaceCandidate& seed) {
    if (region.size() <= seed.points.size()) return seed.points;

    const double seed_slope = std::abs(line_slope(seed.line));
    const double residual_limit = std::max(0.5, seed.rms * 3.0);
    const double x_expand_limit = std::max(seed.span, 1.0);
    std::vector<Eigen::Vector2d> sorted_region = region;
    std::sort(sorted_region.begin(), sorted_region.end(),
              [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                  if (a.x() == b.x()) return a.y() < b.y();
                  return a.x() < b.x();
              });
    const double median_x_step = median_positive_x_step(sorted_region);
    const double x_gap_limit = median_x_step > 0.0
                                       ? std::max(2.5 * median_x_step, 1e-9)
                                       : std::numeric_limits<double>::max();
    std::vector<Eigen::Vector2d> expanded;
    expanded.reserve(sorted_region.size());

    for (const auto& pt : sorted_region) {
        const double residual =
                std::abs(pt.y() - line_y_at_x(seed.line, pt.x()));
        if (residual > residual_limit) continue;

        const double dx_to_seed =
                pt.x() < seed.x_min
                        ? seed.x_min - pt.x()
                        : (pt.x() > seed.x_max ? pt.x() - seed.x_max : 0.0);
        if (dx_to_seed > x_expand_limit) continue;
        expanded.push_back(pt);
    }

    if (expanded.size() < seed.points.size()) return seed.points;

    std::vector<std::vector<Eigen::Vector2d>> components;
    std::vector<Eigen::Vector2d> component;
    for (int i = 0; i < expanded.size(); ++i) {
        if (i > 0 && expanded[i].x() - expanded[i - 1].x() > x_gap_limit) {
            if (!component.empty()) components.push_back(component);
            component.clear();
        }
        component.push_back(expanded[i]);
    }
    if (!component.empty()) components.push_back(component);
    if (components.size() > 1) {
        auto best_component = std::max_element(
                components.begin(), components.end(),
                [&seed](const std::vector<Eigen::Vector2d>& a,
                        const std::vector<Eigen::Vector2d>& b) {
                    auto overlap_score =
                            [&seed](const std::vector<Eigen::Vector2d>& c) {
                                if (c.empty())
                                    return -std::numeric_limits<double>::max();
                                auto [min_it, max_it] = std::minmax_element(
                                        c.begin(), c.end(),
                                        [](const Eigen::Vector2d& p,
                                           const Eigen::Vector2d& q) {
                                            return p.x() < q.x();
                                        });
                                const double overlap = std::max(
                                        0.0, std::min(max_it->x(), seed.x_max) -
                                                     std::max(min_it->x(),
                                                              seed.x_min));
                                return overlap +
                                       0.01 * (max_it->x() - min_it->x());
                            };
                    return overlap_score(a) < overlap_score(b);
                });
        if (best_component != components.end()) expanded = *best_component;
    }

    if (expanded.size() < seed.points.size()) return seed.points;
    auto candidate = make_surface_candidate(expanded);
    if (std::abs(line_slope(candidate.line)) >
        std::max(kMinSurfaceSlopeLimit, seed_slope * 2.0)) {
        return seed.points;
    }
    if (candidate.rms > std::max(residual_limit, seed.rms + 1.0)) {
        return seed.points;
    }
    return expanded;
}

RightSurfaceSearch build_right_surface_search(
        const std::vector<Eigen::Vector2d>& right_region, double edge_x) {
    RightSurfaceSearch search;
    search.edge_x = edge_x;
    if (right_region.empty()) return search;

    std::vector<Eigen::Vector2d> pts = right_region;
    std::sort(pts.begin(), pts.end(),
              [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                  if (a.x() == b.x()) return a.y() < b.y();
                  return a.x() < b.x();
              });

    std::vector<double> abs_slopes;
    abs_slopes.reserve(pts.size());
    for (int i = 0; i < pts.size() - 1; ++i) {
        const double dx = pts[i + 1].x() - pts[i].x();
        if (std::abs(dx) < 1e-12) continue;
        abs_slopes.push_back(std::abs((pts[i + 1].y() - pts[i].y()) / dx));
    }
    double slope_threshold = kMinSurfaceSlopeLimit * 5.0;
    if (!abs_slopes.empty()) {
        std::nth_element(abs_slopes.begin(),
                         abs_slopes.begin() + abs_slopes.size() / 2,
                         abs_slopes.end());
        slope_threshold = std::max(slope_threshold,
                                   abs_slopes[abs_slopes.size() / 2] * 8.0);
    }

    size_t vertical_end = 0;
    for (int i = 0; i < pts.size() - 1; ++i) {
        const double dx = pts[i + 1].x() - pts[i].x();
        if (std::abs(dx) < 1e-12) continue;
        const double slope = std::abs((pts[i + 1].y() - pts[i].y()) / dx);
        if (slope >= slope_threshold) {
            vertical_end = i + 1;
        } else if (vertical_end > 0) {
            break;
        }
    }

    for (size_t i = 0; i <= vertical_end && i < pts.size(); ++i) {
        search.vertical_points.push_back(pts[i]);
    }

    auto valley_it = std::min_element(
            pts.begin(), pts.end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.y() < b.y();
            });
    const size_t valley_idx = valley_it == pts.end()
                                      ? pts.size()
                                      : std::distance(pts.begin(), valley_it);
    if (valley_it != pts.end()) search.valley_x = valley_it->x();

    const size_t valley_margin =
            std::max<size_t>(3, static_cast<size_t>(pts.size() * 0.05));
    const size_t valley_begin =
            valley_idx > valley_margin ? valley_idx - valley_margin : 0;
    const size_t valley_end =
            std::min(pts.size(), valley_idx + valley_margin + 1);

    for (size_t i = 0; i < pts.size(); ++i) {
        const bool in_vertical = i <= vertical_end;
        const bool in_valley = i >= valley_begin && i < valley_end;
        if (in_valley) search.valley_points.push_back(pts[i]);
        if (!in_vertical && !in_valley) search.search_region.push_back(pts[i]);
    }

    return search;
}

std::vector<Eigen::Vector2d> collect_adjacent_right_surface(
        const std::vector<Eigen::Vector2d>& right_region) {
    if (right_region.size() < kMinSurfacePoints) return {};

    std::vector<Eigen::Vector2d> pts = right_region;
    std::sort(pts.begin(), pts.end(),
              [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                  if (a.x() == b.x()) return a.y() < b.y();
                  return a.x() < b.x();
              });

    std::vector<double> abs_slopes;
    abs_slopes.reserve(pts.size());
    for (int i = 0; i < pts.size() - 1; ++i) {
        const double dx = pts[i + 1].x() - pts[i].x();
        if (std::abs(dx) < 1e-12) continue;
        abs_slopes.push_back(std::abs((pts[i + 1].y() - pts[i].y()) / dx));
    }
    if (abs_slopes.empty()) return {};
    std::nth_element(abs_slopes.begin(),
                     abs_slopes.begin() + abs_slopes.size() / 2,
                     abs_slopes.end());
    const double vertical_slope_threshold =
            std::max(kMinSurfaceSlopeLimit * 5.0,
                     abs_slopes[abs_slopes.size() / 2] * 8.0);

    std::vector<Eigen::Vector2d> adjacent;
    adjacent.push_back(pts.front());
    for (int i = 0; i < pts.size() - 1; ++i) {
        const double dx = pts[i + 1].x() - pts[i].x();
        if (std::abs(dx) < 1e-12) break;
        const double slope = std::abs((pts[i + 1].y() - pts[i].y()) / dx);
        if (slope >= vertical_slope_threshold &&
            adjacent.size() >= kMinSurfacePoints) {
            break;
        }
        adjacent.push_back(pts[i + 1]);
    }

    if (adjacent.size() < kMinSurfacePoints) return {};

    auto candidates = collect_surface_candidates(adjacent);
    if (candidates.empty()) return {};

    auto best_it = std::min_element(
            candidates.begin(), candidates.end(),
            [](const SurfaceCandidate& a, const SurfaceCandidate& b) {
                if (a.x_min == b.x_min) return a.span > b.span;
                return a.x_min < b.x_min;
            });
    return best_it != candidates.end() ? best_it->points
                                       : std::vector<Eigen::Vector2d>{};
}

void draw_line_on_plot(cv::Mat& image,
                       const std::pair<Eigen::Vector2d, Eigen::Vector2d>& line,
                       double x_min,
                       double x_max,
                       double y_min,
                       double y_max,
                       const cv::Scalar& color,
                       int thickness = 1,
                       int y_shift = 0) {
    const double x_span = std::max(x_max - x_min, 1e-9);
    const double y_span = std::max(y_max - y_min, 1e-9);
    int line_x_left = static_cast<int>((line.first.x() - x_min) / x_span * 800);
    int line_x_right =
            static_cast<int>((line.second.x() - x_min) / x_span * 800);
    int line_y_left =
            static_cast<int>(500 - (line.first.y() - y_min) / y_span * 500);
    int line_y_right =
            static_cast<int>(500 - (line.second.y() - y_min) / y_span * 500);
    cv::line(image, cv::Point(line_x_left, line_y_left + y_shift),
             cv::Point(line_x_right, line_y_right + y_shift), color, thickness);
}

cv::Point point_on_plot(const Eigen::Vector2d& pt,
                        double x_min,
                        double x_max,
                        double y_min,
                        double y_max,
                        int y_shift = 0) {
    const double x_span = std::max(x_max - x_min, 1e-9);
    const double y_span = std::max(y_max - y_min, 1e-9);
    int x = static_cast<int>((pt.x() - x_min) / x_span * 800);
    int y = static_cast<int>(500 - (pt.y() - y_min) / y_span * 500);
    return cv::Point(x, y + y_shift);
}

void draw_measurement_overlay(
        cv::Mat& image,
        const std::vector<std::vector<Eigen::Vector2d>>& intersections,
        double x_min,
        double x_max,
        double y_min,
        double y_max,
        int y_shift = 0) {
    if (intersections.size() <= 2 || intersections[2].size() < 2) return;

    const Eigen::Vector2d left_boundary = intersections[2][0];
    const Eigen::Vector2d right_boundary = intersections[2][1];
    const cv::Point left_px =
            point_on_plot(left_boundary, x_min, x_max, y_min, y_max, y_shift);
    const cv::Point right_px =
            point_on_plot(right_boundary, x_min, x_max, y_min, y_max, y_shift);
    const cv::Point right_height_px = point_on_plot(
            Eigen::Vector2d(right_boundary.x(), left_boundary.y()), x_min,
            x_max, y_min, y_max, y_shift);

    cv::line(image, left_px, right_height_px, cv::Scalar(0, 0, 0), 2);
    cv::line(image, right_height_px, right_px, cv::Scalar(128, 0, 128), 1);
    cv::drawMarker(image, left_px, cv::Scalar(0, 0, 0), cv::MARKER_SQUARE, 14,
                   2);
    cv::drawMarker(image, right_px, cv::Scalar(0, 0, 0), cv::MARKER_SQUARE, 14,
                   2);

    std::ostringstream label;
    const double signed_height = right_boundary.y() - left_boundary.y();
    label << std::fixed << std::setprecision(3)
          << "W=" << std::abs(right_boundary.x() - left_boundary.x())
          << " H=" << std::abs(signed_height) << " S=" << signed_height;
    cv::putText(image, label.str(), cv::Point(8, image.rows - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1,
                cv::LINE_AA);
}

double trimmed_mean(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t begin = 0;
    size_t end = values.size();
    if (values.size() >= 5) {
        const size_t trim = values.size() / 5;
        begin = trim;
        end = values.size() - trim;
    }
    double sum = 0.0;
    for (size_t i = begin; i < end; ++i) sum += values[i];
    return sum / static_cast<double>(end - begin);
}

double median_value(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 1) return values[mid];
    return 0.5 * (values[mid - 1] + values[mid]);
}

double median_absolute_deviation(const std::vector<double>& values,
                                 double median) {
    std::vector<double> deviations;
    deviations.reserve(values.size());
    for (double value : values) deviations.push_back(std::abs(value - median));
    return median_value(deviations);
}

std::vector<SliceMeasurement> collect_slice_measurements(
        const SliceLineSegments& corners,
        const std::vector<double>& lht_width,
        bool use_lht_width,
        const std::vector<std::vector<Eigen::Vector2d>>* left_pts = nullptr,
        const std::vector<std::vector<Eigen::Vector2d>>* right_pts = nullptr) {
    std::vector<SliceMeasurement> measurements;
    measurements.reserve(corners.size());

    for (int i = 0; i < corners.size(); ++i) {
        SliceMeasurement measurement;
        measurement.index = i;
        measurement.left_boundary = corners[i].first;
        measurement.right_boundary = corners[i].second;

        if (std::isnan(corners[i].first.x()) ||
            std::isnan(corners[i].second.x())) {
            measurement.reject_reason = "invalid_corner";
            measurements.push_back(measurement);
            continue;
        }

        if (corners[i].second.x() <= corners[i].first.x()) {
            measurement.reject_reason = "right_before_left";
            measurements.push_back(measurement);
            continue;
        }

        measurement.width = use_lht_width && i < lht_width.size()
                                    ? lht_width[i]
                                    : std::abs(corners[i].second.x() -
                                               corners[i].first.x());
        set_measurement_height(measurement,
                               corners[i].second.y() - corners[i].first.y());
        if (measurement.width == -255.0 || measurement.width <= 0.0 ||
            !std::isfinite(measurement.width) ||
            !std::isfinite(measurement.height)) {
            measurement.reject_reason = "invalid_measurement";
            measurements.push_back(measurement);
            continue;
        }

        // Store surface point sets if provided
        if (left_pts != nullptr && i < left_pts->size())
            measurement.left_surface_pts = (*left_pts)[i];
        if (right_pts != nullptr && i < right_pts->size())
            measurement.right_surface_pts = (*right_pts)[i];

        measurement.accepted = true;
        measurements.push_back(measurement);
    }

    std::vector<double> widths;
    std::vector<double> heights;
    for (const auto& measurement : measurements) {
        if (!measurement.accepted) continue;
        widths.push_back(measurement.width);
        heights.push_back(measurement.height_abs);
    }
    if (widths.size() < 8) return measurements;

    const double width_median = median_value(widths);
    const double height_median = median_value(heights);
    const double width_sigma =
            1.4826 * median_absolute_deviation(widths, width_median);
    const double height_sigma =
            1.4826 * median_absolute_deviation(heights, height_median);
    const double width_limit =
            std::max({3.0 * width_sigma, std::abs(width_median) * 0.2, 1.0});
    const double height_limit =
            std::max({3.0 * height_sigma, std::abs(height_median) * 0.2, 1.0});

    for (auto& measurement : measurements) {
        if (!measurement.accepted) continue;
        const bool width_outlier =
                std::abs(measurement.width - width_median) > width_limit;
        const bool height_outlier =
                std::abs(measurement.height_abs - height_median) > height_limit;
        if (width_outlier || height_outlier) {
            measurement.accepted = false;
            if (width_outlier && height_outlier) {
                measurement.reject_reason = "width_height_outlier";
            } else if (width_outlier) {
                measurement.reject_reason = "width_outlier";
            } else {
                measurement.reject_reason = "height_outlier";
            }
        }
    }

    return measurements;
}

void annotate_slice_quality(std::vector<SliceMeasurement>& measurements,
                            const geometry::PointCloud::Ptr& cloud) {
    if (!cloud || cloud->width_ == 0 || cloud->y_slices_.empty()) return;
    const size_t expected_points = cloud->width_;
    constexpr double kMinValidSliceRatio = 0.5;
    for (auto& measurement : measurements) {
        if (measurement.index < 0 ||
            measurement.index >= static_cast<int>(cloud->y_slices_.size())) {
            continue;
        }
        measurement.expected_points = expected_points;
        measurement.valid_points = cloud->y_slices_[measurement.index].size();
        measurement.valid_ratio =
                static_cast<double>(measurement.valid_points) /
                static_cast<double>(expected_points);
        if (measurement.accepted &&
            measurement.valid_ratio < kMinValidSliceRatio) {
            measurement.accepted = false;
            measurement.reject_reason = "invalid_height_slice";
        }
    }
}

void fill_result_from_measurements(
        const std::vector<SliceMeasurement>& measurements,
        double& gap_step,
        double& step_width,
        std::vector<std::vector<double>>* temp_res) {
    std::vector<double> widths;
    std::vector<double> heights;
    for (const auto& measurement : measurements) {
        if (!measurement.accepted) continue;
        widths.push_back(measurement.width);
        heights.push_back(measurement.height_abs);
        if (temp_res != nullptr) {
            (*temp_res)[0].emplace_back(measurement.width);
            (*temp_res)[1].emplace_back(measurement.height_abs);
        }
    }

    if (!widths.empty()) {
        gap_step = trimmed_mean(heights);
        step_width = trimmed_mean(widths);
    } else {
        gap_step = 0.0;
        step_width = 0.0;
    }
}

void write_slice_measurements_csv(
        const std::string& debug_path,
        const std::vector<SliceMeasurement>& measurements) {
    ProfileScope profile("write_slice_measurements_csv");
    if (debug_path.empty()) return;
    std::string path = debug_path;
    if (!path.empty() && path.back() != '/' && path.back() != '\\') path += "/";
    path += "slice_metrics.csv";

    std::ofstream ofs(path);
    if (!ofs.is_open()) return;
    ofs << "slice,width,height_abs,signed_height,left_x,left_y,right_x,right_y,"
           "valid_points,expected_points,valid_ratio,"
           "left_residual,right_residual,accepted,reject_reason\n";
    for (const auto& measurement : measurements) {
        ofs << measurement.index << "," << measurement.width << ","
            << measurement.height_abs << "," << measurement.signed_height << ","
            << measurement.left_boundary.x() << ","
            << measurement.left_boundary.y() << ","
            << measurement.right_boundary.x() << ","
            << measurement.right_boundary.y() << "," << measurement.valid_points
            << "," << measurement.expected_points << ","
            << measurement.valid_ratio << "," << measurement.left_residual
            << "," << measurement.right_residual << ","
            << (measurement.accepted ? 1 : 0) << ","
            << measurement.reject_reason << "\n";
    }
}

std::string sanitize_filename_token(const std::string& text) {
    std::string token;
    token.reserve(text.size());
    for (unsigned char ch : text) {
        if (std::isalnum(ch) || ch == '_' || ch == '-') {
            token.push_back(static_cast<char>(ch));
        } else {
            token.push_back('_');
        }
    }
    return token.empty() ? "rejected" : token;
}

void remove_stale_rejected_debug_images(const std::string& path, int index) {
    std::vector<std::string> filenames;
    if (!utility::filesystem::ListFilesInDirectory(path, filenames)) return;

    const std::string prefix =
            path + "group_pts" + std::to_string(index) + "__REJECTED_";
    const std::string suffix = ".jpg";
    for (const auto& filename : filenames) {
        if (filename.size() < prefix.size() + suffix.size()) continue;
        if (filename.compare(0, prefix.size(), prefix) != 0) continue;
        if (filename.compare(filename.size() - suffix.size(), suffix.size(),
                             suffix) != 0) {
            continue;
        }
        utility::filesystem::RemoveFile(filename);
    }
}

void mark_rejected_debug_images(
        const std::string& debug_path,
        const std::vector<SliceMeasurement>& measurements) {
    ProfileScope profile("mark_rejected_debug_images");
    if (debug_path.empty()) return;

    std::string path = debug_path;
    if (path.back() != '/' && path.back() != '\\') path += "/";

    for (const auto& measurement : measurements) {
        remove_stale_rejected_debug_images(path, measurement.index);
        if (measurement.accepted) continue;

        const std::string src =
                path + "group_pts" + std::to_string(measurement.index) + ".jpg";
        if (!utility::filesystem::FileExists(src)) continue;

        const std::string reason =
                sanitize_filename_token(measurement.reject_reason);
        const std::string dst = path + "group_pts" +
                                std::to_string(measurement.index) +
                                "__REJECTED_" + reason + ".jpg";
        if (utility::filesystem::FileExists(dst)) {
            utility::filesystem::RemoveFile(dst);
        }
        std::rename(src.c_str(), dst.c_str());
    }
}

std::vector<std::vector<Eigen::Vector2d>> select_surface_groups(
        std::vector<SurfaceCandidate>& candidates) {
    if (candidates.size() < 2) return {};

    std::sort(candidates.begin(), candidates.end(),
              [](const SurfaceCandidate& a, const SurfaceCandidate& b) {
                  return a.span > b.span;
              });
    const size_t max_candidates = std::min<size_t>(candidates.size(), 8);

    int best_i = -1;
    int best_j = -1;
    double best_score = -std::numeric_limits<double>::max();
    for (int i = 0; i < max_candidates; ++i) {
        for (int j = i + 1; j < max_candidates; ++j) {
            const double x_gap =
                    std::abs(candidates[i].x_center - candidates[j].x_center);
            if (x_gap <= 0.0) continue;
            const double residual_penalty =
                    candidates[i].rms + candidates[j].rms;
            const double slope_penalty =
                    3.0 * (std::abs(line_slope(candidates[i].line)) +
                           std::abs(line_slope(candidates[j].line)));
            const double score = candidates[i].span + candidates[j].span +
                                 0.25 * x_gap - residual_penalty -
                                 slope_penalty;
            if (score > best_score) {
                best_score = score;
                best_i = i;
                best_j = j;
            }
        }
    }

    if (best_i < 0 || best_j < 0) return {};

    std::vector<SurfaceCandidate> selected{candidates[best_i],
                                           candidates[best_j]};
    std::sort(selected.begin(), selected.end(),
              [](const SurfaceCandidate& a, const SurfaceCandidate& b) {
                  return a.x_center < b.x_center;
              });
    return {selected[0].points, selected[1].points};
}

std::vector<SurfaceCandidate> collect_surface_candidates(
        const std::vector<Eigen::Vector2d>& sampled_pts) {
    if (sampled_pts.size() < 3) return {};

    std::vector<Eigen::Vector2d> pts = sampled_pts;
    std::sort(pts.begin(), pts.end(),
              [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                  if (a.x() == b.x()) return a.y() < b.y();
                  return a.x() < b.x();
              });

    std::vector<SurfaceCandidate> candidates;
    std::vector<Eigen::Vector2d> segment;
    double prev_slope = 0.0;
    bool has_prev_slope = false;
    int unstable_gap = 0;
    const double slope_limit = estimate_surface_slope_limit(pts);
    const double slope_delta_limit =
            std::max(kMinSurfaceSlopeLimit, std::min(slope_limit * 1.5, 0.8));
    const double median_x_step = median_positive_x_step(pts);
    const double x_gap_limit = median_x_step > 0.0
                                       ? std::max(2.5 * median_x_step, 1e-9)
                                       : std::numeric_limits<double>::max();
    const double min_surface_span =
            std::max(kMinStableSurfaceSpan, 6.0 * median_x_step);

    auto flush_segment = [&]() {
        if (segment.size() >= kMinSurfacePoints) {
            auto candidate = make_surface_candidate(segment);
            const double slope = std::abs(line_slope(candidate.line));
            const double rms_limit =
                    std::max(0.2, candidate.span * slope_limit * 0.05);
            if (candidate.span >= min_surface_span && slope <= slope_limit &&
                candidate.rms <= rms_limit) {
                candidates.push_back(candidate);
            }
        }
        segment.clear();
        has_prev_slope = false;
        unstable_gap = 0;
    };

    for (int i = 0; i < pts.size() - 1; ++i) {
        const double dx = pts[i + 1].x() - pts[i].x();
        if (std::abs(dx) < 1e-12) {
            flush_segment();
            continue;
        }
        if (dx > x_gap_limit) {
            if (segment.empty()) segment.push_back(pts[i]);
            flush_segment();
            continue;
        }

        const double slope = (pts[i + 1].y() - pts[i].y()) / dx;
        const bool slope_ok = std::abs(slope) <= slope_limit;
        const bool delta_ok = !has_prev_slope ||
                              std::abs(slope - prev_slope) <= slope_delta_limit;
        if (slope_ok && delta_ok) {
            if (unstable_gap > 0) flush_segment();
            if (segment.empty()) segment.push_back(pts[i]);
            segment.push_back(pts[i + 1]);
            prev_slope = slope;
            has_prev_slope = true;
            unstable_gap = 0;
        } else if (!segment.empty() && unstable_gap < kMaxSurfaceGap) {
            unstable_gap++;
        } else {
            flush_segment();
        }
    }
    flush_segment();

    return candidates;
}

std::vector<SurfaceCandidate> collect_platform_candidates(
        const std::vector<Eigen::Vector2d>& sampled_pts) {
    if (sampled_pts.size() < kMinSurfacePoints) return {};

    std::vector<Eigen::Vector2d> pts = sampled_pts;
    std::sort(pts.begin(), pts.end(),
              [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                  if (a.x() == b.x()) return a.y() < b.y();
                  return a.x() < b.x();
              });

    const size_t n = pts.size();
    const double median_x_step = median_positive_x_step(pts);
    const double x_gap_limit = median_x_step > 0.0
                                       ? std::max(2.5 * median_x_step, 1e-9)
                                       : std::numeric_limits<double>::max();
    const double min_span =
            std::max(kMinStableSurfaceSpan, 6.0 * median_x_step);
    const double max_span = std::max(30.0, min_span * 6.0);

    // Precompute prefix sums for O(1) sliding window stats
    std::vector<double> pref_x(n + 1, 0.0), pref_y(n + 1, 0.0);
    std::vector<double> pref_xx(n + 1, 0.0), pref_xy(n + 1, 0.0),
            pref_yy(n + 1, 0.0);
    for (size_t i = 0; i < n; ++i) {
        pref_x[i + 1] = pref_x[i] + pts[i].x();
        pref_y[i + 1] = pref_y[i] + pts[i].y();
        pref_xx[i + 1] = pref_xx[i] + pts[i].x() * pts[i].x();
        pref_xy[i + 1] = pref_xy[i] + pts[i].x() * pts[i].y();
        pref_yy[i + 1] = pref_yy[i] + pts[i].y() * pts[i].y();
    }

    // O(1) window stats helper: returns (slope, intercept, rms) or NaN if
    // invalid
    auto window_stats = [&](size_t start,
                            size_t end) -> std::tuple<double, double, double> {
        size_t m = end - start + 1;
        double sx = pref_x[end + 1] - pref_x[start];
        double sy = pref_y[end + 1] - pref_y[start];
        double sxx = pref_xx[end + 1] - pref_xx[start];
        double sxy = pref_xy[end + 1] - pref_xy[start];
        double syy = pref_yy[end + 1] - pref_yy[start];
        double mx = sx / m, my = sy / m;
        double var_x = sxx - sx * mx;
        if (var_x <= 1e-12)
            return {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0};
        double cov_xy = sxy - sx * my;
        double slope = cov_xy / var_x;
        double intercept = my - slope * mx;
        double rms_sq = (syy - 2.0 * intercept * sy - 2.0 * slope * sxy +
                         intercept * intercept * m +
                         2.0 * slope * intercept * sx + slope * slope * sxx) /
                        m;
        if (rms_sq < 0.0) rms_sq = 0.0;
        return {slope, intercept, std::sqrt(rms_sq)};
    };

    auto [y_min_it, y_max_it] = std::minmax_element(
            pts.begin(), pts.end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.y() < b.y();
            });
    const double x_span_total =
            std::max(pts.back().x() - pts.front().x(), 1e-9);
    const double y_span_total = std::max(y_max_it->y() - y_min_it->y(), 1e-9);

    struct PlatformWindowCandidate {
        size_t start = 0;
        size_t end = 0;
        double slope = 0.0;
        double intercept = 0.0;
        double rms = 0.0;
        double x_min = 0.0;
        double x_max = 0.0;
        double x_center = 0.0;
        double y_center = 0.0;
        double span = 0.0;
        double score = 0.0;
    };
    std::vector<PlatformWindowCandidate> windows;
    for (size_t start = 0; start < n; ++start) {
        for (size_t end = start + kMinSurfacePoints - 1; end < n; ++end) {
            if (end > start && pts[end].x() - pts[end - 1].x() > x_gap_limit)
                break;

            double span = pts[end].x() - pts[start].x();
            if (span < min_span) continue;
            if (span > max_span) break;

            auto [slope, intercept, rms] = window_stats(start, end);
            if (std::isnan(slope)) continue;
            if (std::abs(slope) > kMaxPlatformSlopeLimit) continue;
            double roughness = rms / std::max(span, 1e-9);
            if (roughness > 35.0) continue;

            PlatformWindowCandidate candidate;
            candidate.start = start;
            candidate.end = end;
            candidate.slope = slope;
            candidate.intercept = intercept;
            candidate.rms = rms;
            candidate.x_min = pts[start].x();
            candidate.x_max = pts[end].x();
            candidate.x_center = 0.5 * (candidate.x_min + candidate.x_max);
            candidate.span = span;
            candidate.y_center =
                    (pref_y[end + 1] - pref_y[start]) / (end - start + 1);

            const double x_norm =
                    (candidate.x_center - pts.front().x()) / x_span_total;
            const double y_norm =
                    (candidate.y_center - y_min_it->y()) / y_span_total;
            const double span_score = 4.0 * std::min(candidate.span, 12.0);
            candidate.score = span_score + 18.0 * x_norm + 20.0 * y_norm -
                              5.0 * std::abs(slope) - roughness;
            windows.push_back(candidate);
        }
    }

    constexpr size_t kMaxPlatformCandidates = 32;
    if (windows.size() > kMaxPlatformCandidates) {
        std::partial_sort(windows.begin(),
                          windows.begin() + kMaxPlatformCandidates,
                          windows.end(),
                          [](const PlatformWindowCandidate& a,
                             const PlatformWindowCandidate& b) {
                              return a.score > b.score;
                          });
        windows.resize(kMaxPlatformCandidates);
    }

    std::vector<SurfaceCandidate> candidates;
    candidates.reserve(windows.size());
    for (const auto& window : windows) {
        SurfaceCandidate candidate;
        candidate.points.assign(pts.begin() + window.start,
                                pts.begin() + window.end + 1);
        candidate.x_min = window.x_min;
        candidate.x_max = window.x_max;
        candidate.x_center = window.x_center;
        candidate.y_center = window.y_center;
        candidate.span = window.span;
        candidate.rms = window.rms;
        candidate.line = std::make_pair(
                Eigen::Vector2d(
                        candidate.x_min,
                        window.slope * candidate.x_min + window.intercept),
                Eigen::Vector2d(
                        candidate.x_max,
                        window.slope * candidate.x_max + window.intercept));
        candidates.push_back(std::move(candidate));
    }
    return candidates;
}

std::vector<Eigen::Vector2d> select_right_platform_surface(
        const std::vector<Eigen::Vector2d>& points) {
    auto candidates = collect_platform_candidates(points);
    if (candidates.empty()) return {};

    auto [x_min_it, x_max_it] = std::minmax_element(
            points.begin(), points.end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.x() < b.x();
            });
    auto [y_min_it, y_max_it] = std::minmax_element(
            points.begin(), points.end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.y() < b.y();
            });
    const double x_span = std::max(x_max_it->x() - x_min_it->x(), 1e-9);
    const double y_span = std::max(y_max_it->y() - y_min_it->y(), 1e-9);

    auto best_it = std::max_element(
            candidates.begin(), candidates.end(),
            [&](const SurfaceCandidate& a, const SurfaceCandidate& b) {
                auto score = [&](const SurfaceCandidate& candidate) {
                    const double slope = std::abs(line_slope(candidate.line));
                    const double roughness =
                            candidate.rms / std::max(candidate.span, 1e-9);
                    const double x_norm =
                            (candidate.x_center - x_min_it->x()) / x_span;
                    const double y_norm =
                            (candidate.y_center - y_min_it->y()) / y_span;
                    const double span_score =
                            4.0 * std::min(candidate.span, 12.0);
                    return span_score + 18.0 * x_norm + 20.0 * y_norm -
                           5.0 * slope - roughness;
                };
                return score(a) < score(b);
            });

    return best_it != candidates.end() ? best_it->points
                                       : std::vector<Eigen::Vector2d>{};
}

std::vector<std::vector<Eigen::Vector2d>> split_by_step_edge(
        const std::vector<Eigen::Vector2d>& sampled_pts) {
    if (sampled_pts.size() < 8) return {};

    std::vector<Eigen::Vector2d> pts = sampled_pts;
    std::sort(pts.begin(), pts.end(),
              [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                  if (a.x() == b.x()) return a.y() < b.y();
                  return a.x() < b.x();
              });

    std::vector<double> abs_slopes;
    abs_slopes.reserve(pts.size());
    std::vector<double> x_steps;
    x_steps.reserve(pts.size());
    int edge_idx = -1;
    double max_abs_slope = -1.0;
    double max_x_step = -1.0;
    int max_x_step_idx = -1;
    for (int i = 0; i < pts.size() - 1; ++i) {
        const double dx = pts[i + 1].x() - pts[i].x();
        if (std::abs(dx) < 1e-12) continue;
        x_steps.push_back(dx);
        if (dx > max_x_step) {
            max_x_step = dx;
            max_x_step_idx = i;
        }
        const double slope = (pts[i + 1].y() - pts[i].y()) / dx;
        const double abs_slope = std::abs(slope);
        abs_slopes.push_back(abs_slope);
        if (abs_slope > max_abs_slope) {
            max_abs_slope = abs_slope;
            edge_idx = i;
        }
    }
    if (edge_idx < 0 || abs_slopes.empty()) return {};

    std::nth_element(abs_slopes.begin(),
                     abs_slopes.begin() + abs_slopes.size() / 2,
                     abs_slopes.end());
    const double median_abs_slope = abs_slopes[abs_slopes.size() / 2];
    const double edge_slope_threshold =
            std::max(kMinSurfaceSlopeLimit * 5.0, median_abs_slope * 8.0);
    std::nth_element(x_steps.begin(), x_steps.begin() + x_steps.size() / 2,
                     x_steps.end());
    const double median_x_step = x_steps[x_steps.size() / 2];
    const bool has_x_gap = max_x_step > std::max(2.0 * median_x_step, 1e-9);
    const bool has_slope_edge = max_abs_slope >= edge_slope_threshold;
    if (!has_x_gap && !has_slope_edge) return {};
    if (has_x_gap) edge_idx = max_x_step_idx;

    const double edge_x = pts[edge_idx].x();
    const double x_span = std::max(pts.back().x() - pts.front().x(), 1e-9);
    const double edge_margin = x_span * 0.005;

    std::vector<Eigen::Vector2d> left_region;
    std::vector<Eigen::Vector2d> right_region;
    for (const auto& pt : pts) {
        if (pt.x() < edge_x - edge_margin) {
            left_region.push_back(pt);
        } else if (pt.x() > edge_x + edge_margin) {
            right_region.push_back(pt);
        }
    }
    if (left_region.size() < kMinSurfacePoints ||
        right_region.size() < kMinSurfacePoints) {
        return {};
    }

    auto right_search = build_right_surface_search(right_region, edge_x);
    const auto& right_candidates_region = right_search.search_region.empty()
                                                  ? right_region
                                                  : right_search.search_region;
    return {left_region, right_candidates_region};
}

std::vector<Eigen::Vector2d> filter_surface_group(
        const std::vector<Eigen::Vector2d>& points,
        bool prefer_stable_subsegment) {
    if (points.size() < 8) return points;

    std::vector<Eigen::Vector2d> base_points = points;
    if (prefer_stable_subsegment) {
        auto platform = select_right_platform_surface(points);
        if (platform.size() >= kMinSurfacePoints) return platform;

        auto candidates = collect_surface_candidates(points);
        if (candidates.empty()) {
            return {};
        } else {
            auto best_it = std::max_element(
                    candidates.begin(), candidates.end(),
                    [](const SurfaceCandidate& a, const SurfaceCandidate& b) {
                        auto score = [](const SurfaceCandidate& candidate) {
                            return candidate.span -
                                   8.0 * std::abs(line_slope(candidate.line)) -
                                   5.0 * candidate.rms;
                        };
                        return score(a) < score(b);
                    });
            if (best_it->points.size() >= kMinSurfacePoints) {
                return best_it->points;
            }
            return {};
        }
    }

    const auto line = fit_line_segment_robust(base_points);
    double sum_sq_residual = 0.0;
    for (const auto& pt : base_points) {
        const double residual = pt.y() - line_y_at_x(line, pt.x());
        sum_sq_residual += residual * residual;
    }
    const double rms = std::sqrt(sum_sq_residual /
                                 static_cast<double>(base_points.size()));
    double threshold = std::max(3.0 * rms, 0.03);
    if (prefer_stable_subsegment &&
        base_points.size() >= 2 * kMinSurfacePoints) {
        std::vector<double> residuals;
        residuals.reserve(base_points.size());
        for (const auto& pt : base_points) {
            residuals.push_back(std::abs(pt.y() - line_y_at_x(line, pt.x())));
        }
        std::sort(residuals.begin(), residuals.end());
        const size_t keep_count =
                std::max<size_t>(kMinSurfacePoints, residuals.size() * 6 / 10);
        threshold = std::min(threshold, residuals[keep_count - 1]);
    }

    std::vector<Eigen::Vector2d> filtered;
    filtered.reserve(base_points.size());
    for (const auto& pt : base_points) {
        const double residual = std::abs(pt.y() - line_y_at_x(line, pt.x()));
        if (residual <= threshold) filtered.push_back(pt);
    }
    return filtered.size() >= kMinSurfacePoints ? filtered : base_points;
}

std::vector<std::vector<Eigen::Vector2d>> extract_surface_candidates(
        const std::vector<Eigen::Vector2d>& sampled_pts) {
    auto step_groups = split_by_step_edge(sampled_pts);
    if (step_groups.size() >= 2) return step_groups;

    auto candidates = collect_surface_candidates(sampled_pts);
    return select_surface_groups(candidates);
}

std::vector<std::vector<Eigen::Vector2d>> group_horizontal_by_height(
        std::vector<Eigen::Vector2d>& horiz_pts) {
    if (horiz_pts.size() < 2) return {};

    std::sort(horiz_pts.begin(), horiz_pts.end(),
              [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                  if (a.y() == b.y()) return a.x() < b.x();
                  return a.y() < b.y();
              });

    const size_t min_side =
            std::min<size_t>(5, std::max<size_t>(2, horiz_pts.size() / 20));
    size_t split_idx = 0;
    double best_gap = -1.0;
    for (size_t i = min_side; i + min_side <= horiz_pts.size(); ++i) {
        const double gap = horiz_pts[i].y() - horiz_pts[i - 1].y();
        if (gap > best_gap) {
            best_gap = gap;
            split_idx = i;
        }
    }

    const double y_range = horiz_pts.back().y() - horiz_pts.front().y();
    const double min_gap = std::max(0.05, y_range * 0.1);
    if (split_idx == 0 || best_gap < min_gap) {
        return {horiz_pts};
    }

    std::vector<std::vector<Eigen::Vector2d>> groups(2);
    groups[0].assign(horiz_pts.begin(), horiz_pts.begin() + split_idx);
    groups[1].assign(horiz_pts.begin() + split_idx, horiz_pts.end());
    return groups;
}
// ========== 3D Plane Fitting and Consistency Filter ==========

// Fit plane z = a*x + b*y + c using least squares
Eigen::Vector3d fit_plane_ls(const std::vector<Eigen::Vector3d>& pts) {
    const size_t n = pts.size();
    Eigen::MatrixXd A(n, 3);
    Eigen::VectorXd b_vec(n);
    for (size_t i = 0; i < n; ++i) {
        A(i, 0) = pts[i].x();
        A(i, 1) = pts[i].y();
        A(i, 2) = 1.0;
        b_vec(i) = pts[i].z();
    }
    return A.colPivHouseholderQr().solve(b_vec);
}

struct RobustPlaneResult {
    Eigen::Vector3d coeff;  // (a, b, c) for z = a*x + b*y + c
    std::vector<double> residuals;
    double rms = 0.0;
    double mad = 0.0;
};

double plane_z_at(const Eigen::Vector3d& coeff, double x, double y) {
    return coeff(0) * x + coeff(1) * y + coeff(2);
}

std::string format_double(double value, int precision = 3) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

void draw_final_measurement_summary(
        const std::string& output_path,
        const std::vector<SliceMeasurement>& measurements) {
    ProfileScope profile("draw_final_measurement_summary");
    if (measurements.empty()) return;

    std::vector<double> accepted_widths;
    std::vector<double> accepted_heights;
    std::vector<double> accepted_signed_heights;
    std::map<std::string, int> reject_counts;
    for (const auto& measurement : measurements) {
        if (measurement.accepted) {
            accepted_widths.push_back(measurement.width);
            accepted_heights.push_back(measurement.height_abs);
            accepted_signed_heights.push_back(measurement.signed_height);
        } else {
            reject_counts[measurement.reject_reason.empty()
                                  ? "unknown"
                                  : measurement.reject_reason]++;
        }
    }

    const double final_width = trimmed_mean(accepted_widths);
    const double final_height = trimmed_mean(accepted_heights);
    const double final_signed_height = trimmed_mean(accepted_signed_heights);
    const int accepted_count = static_cast<int>(accepted_widths.size());
    const int total_count = static_cast<int>(measurements.size());

    cv::Mat image(900, 1200, CV_8UC3, cv::Scalar(255, 255, 255));
    const cv::Scalar text_color(30, 30, 30);
    cv::putText(image, "Final measurement summary", cv::Point(24, 36),
                cv::FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv::LINE_AA);
    cv::putText(image,
                "accepted: " + std::to_string(accepted_count) + " / " +
                        std::to_string(total_count),
                cv::Point(24, 70), cv::FONT_HERSHEY_SIMPLEX, 0.58, text_color,
                1, cv::LINE_AA);
    cv::putText(image, "step_width: " + format_double(final_width),
                cv::Point(260, 70), cv::FONT_HERSHEY_SIMPLEX, 0.58, text_color,
                1, cv::LINE_AA);
    cv::putText(image, "gap_step_abs: " + format_double(final_height),
                cv::Point(500, 70), cv::FONT_HERSHEY_SIMPLEX, 0.58, text_color,
                1, cv::LINE_AA);
    cv::putText(image, "signed_height: " + format_double(final_signed_height),
                cv::Point(760, 70), cv::FONT_HERSHEY_SIMPLEX, 0.48, text_color,
                1, cv::LINE_AA);
    cv::putText(image,
                "green=used  red=rejected  purple=signed height  black=width",
                cv::Point(24, 100), cv::FONT_HERSHEY_SIMPLEX, 0.48,
                cv::Scalar(80, 80, 80), 1, cv::LINE_AA);

    int reason_y = 126;
    for (const auto& [reason, count] : reject_counts) {
        cv::putText(image, reason + ": " + std::to_string(count),
                    cv::Point(24, reason_y), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
        reason_y += 20;
        if (reason_y > 160) break;
    }

    auto draw_chart = [&](const cv::Rect& rect, const std::string& title,
                          const std::vector<double>& values,
                          double final_value) {
        cv::rectangle(image, rect, cv::Scalar(210, 210, 210), 1);
        cv::putText(image, title, cv::Point(rect.x + 10, rect.y + 24),
                    cv::FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1, cv::LINE_AA);
        if (values.empty() || values.size() != measurements.size()) return;

        double v_min = std::numeric_limits<double>::max();
        double v_max = -std::numeric_limits<double>::max();
        for (int i = 0; i < values.size(); ++i) {
            if (!measurements[i].accepted || !std::isfinite(values[i]))
                continue;
            v_min = std::min(v_min, values[i]);
            v_max = std::max(v_max, values[i]);
        }
        if (v_min > v_max) {
            for (double value : values) {
                if (!std::isfinite(value)) continue;
                v_min = std::min(v_min, value);
                v_max = std::max(v_max, value);
            }
        }
        if (v_min > v_max) return;
        if (std::abs(v_max - v_min) < 1e-9) {
            v_max += 1.0;
            v_min -= 1.0;
        }

        const int left = rect.x + 44;
        const int right = rect.x + rect.width - 16;
        const int top = rect.y + 42;
        const int bottom = rect.y + rect.height - 30;
        auto to_point = [&](int i, double value) {
            const double x_ratio =
                    static_cast<double>(i) /
                    static_cast<double>(std::max<int>(1, values.size() - 1));
            const double y_ratio =
                    std::clamp((value - v_min) / (v_max - v_min), 0.0, 1.0);
            return cv::Point(
                    static_cast<int>(left + x_ratio * (right - left)),
                    static_cast<int>(bottom - y_ratio * (bottom - top)));
        };

        cv::line(image, cv::Point(left, bottom), cv::Point(right, bottom),
                 cv::Scalar(230, 230, 230), 1);
        cv::line(image, cv::Point(left, top), cv::Point(left, bottom),
                 cv::Scalar(230, 230, 230), 1);
        cv::putText(image, format_double(v_max, 1),
                    cv::Point(rect.x + 4, top + 5), cv::FONT_HERSHEY_SIMPLEX,
                    0.34, cv::Scalar(110, 110, 110), 1, cv::LINE_AA);
        cv::putText(image, format_double(v_min, 1),
                    cv::Point(rect.x + 4, bottom + 5), cv::FONT_HERSHEY_SIMPLEX,
                    0.34, cv::Scalar(110, 110, 110), 1, cv::LINE_AA);

        if (std::isfinite(final_value)) {
            const cv::Point mean_left = to_point(0, final_value);
            const cv::Point mean_right =
                    to_point(static_cast<int>(values.size() - 1), final_value);
            cv::line(image, mean_left, mean_right, cv::Scalar(40, 40, 40), 1,
                     cv::LINE_AA);
            cv::putText(image, "final " + format_double(final_value, 2),
                        cv::Point(rect.x + rect.width - 170, rect.y + 24),
                        cv::FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1,
                        cv::LINE_AA);
        }

        for (int i = 1; i < values.size(); ++i) {
            if (!measurements[i].accepted || !measurements[i - 1].accepted)
                continue;
            if (!std::isfinite(values[i]) || !std::isfinite(values[i - 1]))
                continue;
            cv::line(image, to_point(i - 1, values[i - 1]),
                     to_point(i, values[i]), cv::Scalar(0, 145, 0), 1,
                     cv::LINE_AA);
        }
        for (int i = 0; i < values.size(); ++i) {
            if (!std::isfinite(values[i])) continue;
            const cv::Scalar color = measurements[i].accepted
                                             ? cv::Scalar(0, 145, 0)
                                             : cv::Scalar(0, 0, 230);
            cv::circle(image, to_point(i, values[i]),
                       measurements[i].accepted ? 2 : 3, color, -1,
                       cv::LINE_AA);
        }
    };

    std::vector<double> widths(measurements.size());
    std::vector<double> heights(measurements.size());
    for (int i = 0; i < measurements.size(); ++i) {
        widths[i] = measurements[i].width;
        heights[i] = measurements[i].height_abs;
    }

    draw_chart(cv::Rect(24, 180, 552, 245), "width by slice", widths,
               final_width);
    draw_chart(cv::Rect(624, 180, 552, 245), "height_abs by slice", heights,
               final_height);

    const cv::Rect geom_rect(24, 465, 1152, 390);
    cv::rectangle(image, geom_rect, cv::Scalar(210, 210, 210), 1);
    cv::putText(image,
                "Measured geometry from fitted surfaces at accepted boundaries",
                cv::Point(geom_rect.x + 10, geom_rect.y + 26),
                cv::FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1, cv::LINE_AA);

    double x_min = std::numeric_limits<double>::max();
    double x_max = -std::numeric_limits<double>::max();
    double z_min = std::numeric_limits<double>::max();
    double z_max = -std::numeric_limits<double>::max();
    for (const auto& measurement : measurements) {
        if (!measurement.accepted) continue;
        x_min = std::min({x_min, measurement.left_boundary.x(),
                          measurement.right_boundary.x()});
        x_max = std::max({x_max, measurement.left_boundary.x(),
                          measurement.right_boundary.x()});
        z_min = std::min({z_min, measurement.left_boundary.y(),
                          measurement.right_boundary.y()});
        z_max = std::max({z_max, measurement.left_boundary.y(),
                          measurement.right_boundary.y()});
    }

    if (x_min <= x_max && z_min <= z_max) {
        if (std::abs(x_max - x_min) < 1e-9) {
            x_min -= 1.0;
            x_max += 1.0;
        }
        if (std::abs(z_max - z_min) < 1e-9) {
            z_min -= 1.0;
            z_max += 1.0;
        }

        const int left = geom_rect.x + 58;
        const int right = geom_rect.x + geom_rect.width - 28;
        const int top = geom_rect.y + 52;
        const int bottom = geom_rect.y + geom_rect.height - 34;
        auto geom_point = [&](double x, double z) {
            const double x_ratio = (x - x_min) / (x_max - x_min);
            const double z_ratio = (z - z_min) / (z_max - z_min);
            return cv::Point(
                    static_cast<int>(left + x_ratio * (right - left)),
                    static_cast<int>(bottom - z_ratio * (bottom - top)));
        };

        cv::line(image, cv::Point(left, bottom), cv::Point(right, bottom),
                 cv::Scalar(230, 230, 230), 1);
        cv::line(image, cv::Point(left, top), cv::Point(left, bottom),
                 cv::Scalar(230, 230, 230), 1);
        cv::putText(image, "x", cv::Point(right - 8, bottom + 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(110, 110, 110),
                    1, cv::LINE_AA);
        cv::putText(image, "z", cv::Point(left - 22, top + 6),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(110, 110, 110),
                    1, cv::LINE_AA);

        std::vector<cv::Point> left_curve;
        std::vector<cv::Point> right_curve;
        int accepted_seen = 0;
        const int draw_stride = std::max(1, accepted_count / 24);
        for (const auto& measurement : measurements) {
            if (!measurement.accepted) continue;
            const cv::Point left_pt = geom_point(measurement.left_boundary.x(),
                                                 measurement.left_boundary.y());
            const cv::Point right_pt =
                    geom_point(measurement.right_boundary.x(),
                               measurement.right_boundary.y());
            left_curve.push_back(left_pt);
            right_curve.push_back(right_pt);

            if (accepted_seen % draw_stride == 0) {
                const cv::Point width_right =
                        geom_point(measurement.right_boundary.x(),
                                   measurement.left_boundary.y());
                cv::line(image, left_pt, width_right, cv::Scalar(30, 30, 30), 1,
                         cv::LINE_AA);
                cv::line(image, width_right, right_pt, cv::Scalar(160, 40, 160),
                         1, cv::LINE_AA);
            }
            accepted_seen++;
        }

        for (int i = 1; i < left_curve.size(); ++i) {
            cv::line(image, left_curve[i - 1], left_curve[i],
                     cv::Scalar(0, 0, 220), 2, cv::LINE_AA);
            cv::line(image, right_curve[i - 1], right_curve[i],
                     cv::Scalar(220, 0, 0), 2, cv::LINE_AA);
        }
        for (const auto& pt : left_curve)
            cv::circle(image, pt, 2, cv::Scalar(0, 0, 220), -1, cv::LINE_AA);
        for (const auto& pt : right_curve)
            cv::circle(image, pt, 2, cv::Scalar(220, 0, 0), -1, cv::LINE_AA);

        cv::putText(image, "left fitted surface boundary",
                    cv::Point(geom_rect.x + 18,
                              geom_rect.y + geom_rect.height - 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(0, 0, 220), 1,
                    cv::LINE_AA);
        cv::putText(image, "right fitted surface boundary",
                    cv::Point(geom_rect.x + 250,
                              geom_rect.y + geom_rect.height - 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(220, 0, 0), 1,
                    cv::LINE_AA);
        cv::putText(image,
                    "measurement lines sampled: " +
                            std::to_string((accepted_count + draw_stride - 1) /
                                           draw_stride),
                    cv::Point(geom_rect.x + 520,
                              geom_rect.y + geom_rect.height - 14),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42, cv::Scalar(80, 80, 80), 1,
                    cv::LINE_AA);
    }

    cv::imwrite(output_path, image);
}

void remove_legacy_3d_debug_artifacts(const std::string& debug_path) {
    if (debug_path.empty()) return;

    std::string path = debug_path;
    if (path.back() != '/' && path.back() != '\\') path += "/";
    const char* filenames[] = {
            "slice_metrics_3d.csv",        "left_3d_plane.tiff",
            "right_3d_plane.tiff",         "left_3d_plane_full.tiff",
            "right_3d_plane_full.tiff",    "left_3d_plane.ply",
            "right_3d_plane.ply",          "fitted_3d_planes.ply",
            "left_3d_plane_full.ply",      "right_3d_plane_full.ply",
            "fitted_3d_planes_full.ply",   "left_residual_vs_slice.png",
            "right_residual_vs_slice.png",
    };
    for (const char* filename : filenames) {
        std::remove((path + filename).c_str());
    }
}

void write_3d_filter_debug_snapshot(
        const std::string& debug_path,
        const std::vector<SliceMeasurement>& measurements) {
    if (debug_path.empty()) return;

    std::string path = debug_path;
    if (path.back() != '/' && path.back() != '\\') path += "/";
    utility::filesystem::MakeDirectoryHierarchy(path);
    remove_legacy_3d_debug_artifacts(path);

    draw_final_measurement_summary(path + "final_measurement_summary.png",
                                   measurements);
}

// Robust plane fit: MAD outlier rejection + one re-fit pass
RobustPlaneResult robust_plane_fit(const std::vector<Eigen::Vector3d>& pts) {
    RobustPlaneResult result;
    if (pts.size() < 6) {
        result.coeff = fit_plane_ls(pts);
        result.residuals.resize(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            double zp = result.coeff(0) * pts[i].x() +
                        result.coeff(1) * pts[i].y() + result.coeff(2);
            result.residuals[i] = std::abs(pts[i].z() - zp);
        }
        return result;
    }
    result.coeff = fit_plane_ls(pts);
    result.residuals.resize(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        double zp = result.coeff(0) * pts[i].x() +
                    result.coeff(1) * pts[i].y() + result.coeff(2);
        result.residuals[i] = std::abs(pts[i].z() - zp);
    }
    double med = median_value(result.residuals);
    result.mad = median_absolute_deviation(result.residuals, med);
    const double threshold = med + 3.0 * 1.4826 * result.mad;

    std::vector<Eigen::Vector3d> inliers;
    inliers.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        if (result.residuals[i] <= threshold) inliers.push_back(pts[i]);
    }
    if (inliers.size() >= std::max<size_t>(6, pts.size() / 3)) {
        result.coeff = fit_plane_ls(inliers);
        double sum_sq = 0.0;
        for (size_t i = 0; i < pts.size(); ++i) {
            double zp = result.coeff(0) * pts[i].x() +
                        result.coeff(1) * pts[i].y() + result.coeff(2);
            result.residuals[i] = std::abs(pts[i].z() - zp);
            sum_sq += result.residuals[i] * result.residuals[i];
        }
        result.rms = std::sqrt(sum_sq / pts.size());
        med = median_value(result.residuals);
        result.mad = median_absolute_deviation(result.residuals, med);
    }
    return result;
}

// Per-slice residual computation and 3D+continuity filtering
// Modifies measurements in-place
std::vector<SliceMeasurement> filter_slices_by_3d_consistency(
        std::vector<SliceMeasurement> measurements,
        const Eigen::Vector3d& trans_mat,
        const std::string& debug_path) {
    ProfileScope profile("filter_slices_by_3d_consistency");
    if (measurements.empty()) return measurements;
    const int n = static_cast<int>(measurements.size());

    // Build 3D point clouds from accepted slices
    std::vector<Eigen::Vector3d> left_pts_3d, right_pts_3d;
    for (int i = 0; i < n; ++i) {
        auto& m = measurements[i];
        if (!m.accepted) continue;
        if (m.left_surface_pts.empty() || m.right_surface_pts.empty()) {
            m.accepted = false;
            m.reject_reason = "insufficient_surface_points";
            continue;
        }
        double y_phys = static_cast<double>(m.index) * trans_mat.y();
        for (const auto& pt : m.left_surface_pts)
            left_pts_3d.emplace_back(pt.x(), y_phys, pt.y());
        for (const auto& pt : m.right_surface_pts)
            right_pts_3d.emplace_back(pt.x(), y_phys, pt.y());
    }
    if (left_pts_3d.size() < 6 || right_pts_3d.size() < 6) {
        write_3d_filter_debug_snapshot(debug_path, measurements);
        return measurements;
    }

    // Fit planes
    RobustPlaneResult left_plane = robust_plane_fit(left_pts_3d);
    RobustPlaneResult right_plane = robust_plane_fit(right_pts_3d);

    // Compute per-slice mean residuals
    for (int i = 0; i < n; ++i) {
        auto& m = measurements[i];
        if (!m.accepted) continue;
        double y_phys = static_cast<double>(m.index) * trans_mat.y();
        double sum_l = 0.0;
        for (const auto& pt : m.left_surface_pts) {
            double zp = plane_z_at(left_plane.coeff, pt.x(), y_phys);
            sum_l += std::abs(pt.y() - zp);
        }
        m.left_residual = sum_l / m.left_surface_pts.size();
        double sum_r = 0.0;
        for (const auto& pt : m.right_surface_pts) {
            double zp = plane_z_at(right_plane.coeff, pt.x(), y_phys);
            sum_r += std::abs(pt.y() - zp);
        }
        m.right_residual = sum_r / m.right_surface_pts.size();

        const double left_z =
                plane_z_at(left_plane.coeff, m.left_boundary.x(), y_phys);
        const double right_z =
                plane_z_at(right_plane.coeff, m.right_boundary.x(), y_phys);
        if (std::isfinite(left_z) && std::isfinite(right_z)) {
            m.left_boundary.y() = left_z;
            m.right_boundary.y() = right_z;
            set_measurement_height(m, right_z - left_z);
        }
    }

    // Collect statistics from accepted slices
    std::vector<double> lr, rr, ws, hs;
    for (const auto& m : measurements) {
        if (!m.accepted) continue;
        lr.push_back(m.left_residual);
        rr.push_back(m.right_residual);
        ws.push_back(m.width);
        hs.push_back(m.height_abs);
    }
    if (lr.size() < 7) {
        write_3d_filter_debug_snapshot(debug_path, measurements);
        return measurements;
    }

    double lr_m = median_value(lr),
           lr_mad = median_absolute_deviation(lr, lr_m);
    double rr_m = median_value(rr),
           rr_mad = median_absolute_deviation(rr, rr_m);
    double lr_limit = lr_m + 4.0 * 1.4826 * lr_mad;
    double rr_limit = rr_m + 4.0 * 1.4826 * rr_mad;
    double w_m = median_value(ws), w_mad = median_absolute_deviation(ws, w_m);
    double h_m = median_value(hs), h_mad = median_absolute_deviation(hs, h_m);
    const double global_w_limit = std::max(3.0 * 1.4826 * w_mad, w_m * 0.3);
    const double global_h_limit =
            std::max(3.0 * 1.4826 * h_mad, std::abs(h_m) * 0.3);

    std::vector<char> local_continuity_candidates(n, false);
    for (int i = 0; i < n; ++i) {
        const auto& m = measurements[i];
        local_continuity_candidates[i] = m.accepted &&
                                         m.left_residual <= lr_limit &&
                                         m.right_residual <= rr_limit;
    }

    auto local_scalar_outlier = [&](int i, auto getter, double min_limit) {
        constexpr int kLocalRadius = 4;
        std::vector<double> values;
        values.reserve(2 * kLocalRadius + 1);
        for (int j = std::max(0, i - kLocalRadius);
             j <= std::min(n - 1, i + kLocalRadius); ++j) {
            if (!local_continuity_candidates[j]) continue;
            values.push_back(getter(measurements[j]));
        }
        if (values.size() < 5) return false;
        const double median = median_value(values);
        const double mad = median_absolute_deviation(values, median);
        const double limit = std::max(4.0 * 1.4826 * mad, min_limit);
        return std::abs(getter(measurements[i]) - median) > limit;
    };

    // Apply filter
    for (int i = 0; i < n; ++i) {
        auto& m = measurements[i];
        if (!m.accepted) continue;
        if (m.left_residual > lr_limit) {
            m.accepted = false;
            m.reject_reason = "left_surface_residual_outlier";
            continue;
        }
        if (m.right_residual > rr_limit) {
            m.accepted = false;
            m.reject_reason = "right_surface_residual_outlier";
            continue;
        }
        // Width continuity against a local robust window. This avoids
        // order-dependent rejection cascades from single-neighbor checks.
        const double local_w_limit = std::max(2.0, w_m * 0.4);
        if (local_scalar_outlier(
                    i, [](const SliceMeasurement& s) { return s.width; },
                    std::min(global_w_limit, local_w_limit))) {
            m.accepted = false;
            m.reject_reason = "width_local_jump";
            continue;
        }
        // Height continuity
        const double local_h_limit = std::max(2.0, std::abs(h_m) * 0.25);
        if (local_scalar_outlier(
                    i, [](const SliceMeasurement& s) { return s.height_abs; },
                    std::min(global_h_limit, local_h_limit))) {
            m.accepted = false;
            m.reject_reason = "height_local_jump";
            continue;
        }
        // x-boundary continuity. Compare to the local trend instead of a
        // single adjacent slice so smooth diagonal seams are preserved.
        const double boundary_limit = std::max(2.0, w_m * 0.5);
        if (local_scalar_outlier(
                    i,
                    [](const SliceMeasurement& s) {
                        return s.left_boundary.x();
                    },
                    boundary_limit) ||
            local_scalar_outlier(
                    i,
                    [](const SliceMeasurement& s) {
                        return s.right_boundary.x();
                    },
                    boundary_limit)) {
            m.accepted = false;
            m.reject_reason = "boundary_local_jump";
            continue;
        }
        if (m.right_surface_pts.size() < kMinSurfacePoints) {
            m.accepted = false;
            m.reject_reason = "short_right_support";
            continue;
        }
    }

    // Debug output
    if (!debug_path.empty()) {
        std::string path = debug_path;
        if (path.back() != '/' && path.back() != '\\') path += "/";
        utility::filesystem::MakeDirectoryHierarchy(path);
        write_3d_filter_debug_snapshot(path, measurements);
        // Overview plots: rejected slices as red dots, accepted as green lines
        auto draw_overview = [&](const std::string& fname,
                                 const std::vector<double>& vals) {
            if (vals.empty() || n < 2) return;
            // Find range from accepted slices only
            double vmin = std::numeric_limits<double>::max();
            double vmax = -std::numeric_limits<double>::max();
            for (int i = 0; i < n; ++i) {
                if (!measurements[i].accepted || !std::isfinite(vals[i]))
                    continue;
                vmin = std::min(vmin, vals[i]);
                vmax = std::max(vmax, vals[i]);
            }
            if (vmin > vmax) {
                vmin = 0.0;
                vmax = 1.0;
            }
            double vspan = std::max(vmax - vmin, 1e-9);
            cv::Mat img(400, 800, CV_8UC3, cv::Scalar(255, 255, 255));
            // Draw rejected slices as red dots
            for (int i = 0; i < n; ++i) {
                if (measurements[i].accepted) continue;
                if (!std::isfinite(vals[i])) continue;
                int x = i * 780 / std::max(n - 1, 1) + 10;
                int y = static_cast<int>(380 - (vals[i] - vmin) / vspan * 360);
                cv::circle(img, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
            }
            // Draw accepted slices as green lines connecting adjacent accepted
            for (int i = 1; i < n; ++i) {
                if (!measurements[i].accepted || !measurements[i - 1].accepted)
                    continue;
                if (!std::isfinite(vals[i]) || !std::isfinite(vals[i - 1]))
                    continue;
                int x1 = (i - 1) * 780 / (n - 1) + 10;
                int x2 = i * 780 / (n - 1) + 10;
                int y1 = static_cast<int>(380 -
                                          (vals[i - 1] - vmin) / vspan * 360);
                int y2 = static_cast<int>(380 - (vals[i] - vmin) / vspan * 360);
                cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2),
                         cv::Scalar(0, 128, 0), 1);
            }
            cv::imwrite(path + fname, img);
        };
        std::vector<double> wv(n), hv(n);
        for (int i = 0; i < n; ++i) {
            wv[i] = measurements[i].width;
            hv[i] = measurements[i].height;
        }
        draw_overview("width_vs_slice.png", wv);
        draw_overview("height_vs_slice.png", hv);
    }
    return measurements;
}

// ========== Raw-grid fast path ==========
std::vector<std::vector<Eigen::Vector2d>> detect_missing_gap_platforms(
        const std::vector<Eigen::Vector2d>& sorted_pts,
        std::string* fallback_reason) {
    if (sorted_pts.size() < 2 * kMinSurfacePoints) return {};

    std::vector<double> x_steps;
    x_steps.reserve(sorted_pts.size() - 1);
    double max_gap = -1.0;
    size_t max_gap_idx = 0;
    for (size_t i = 1; i < sorted_pts.size(); ++i) {
        const double dx = sorted_pts[i].x() - sorted_pts[i - 1].x();
        if (dx <= 1e-12) return {};
        x_steps.push_back(dx);
        if (dx > max_gap) {
            max_gap = dx;
            max_gap_idx = i - 1;
        }
    }
    if (x_steps.empty()) return {};

    const double median_dx = median_value(x_steps);
    const double missing_gap_threshold =
            std::max(3.0 * median_dx, median_dx + 1e-9);
    if (max_gap <= missing_gap_threshold) return {};

    auto [y_min_it, y_max_it] = std::minmax_element(
            sorted_pts.begin(), sorted_pts.end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.y() < b.y();
            });
    const double z_range = std::abs(y_max_it->y() - y_min_it->y());
    const double max_acceptable_rms = std::max(10.0, z_range * 0.025);

    std::vector<Eigen::Vector2d> left_region(
            sorted_pts.begin(), sorted_pts.begin() + max_gap_idx + 1);
    std::vector<Eigen::Vector2d> right_region(
            sorted_pts.begin() + max_gap_idx + 1, sorted_pts.end());
    if (left_region.size() < kMinSurfacePoints ||
        right_region.size() < kMinSurfacePoints) {
        if (fallback_reason) *fallback_reason = "missing_gap_too_few_points";
        return {};
    }

    auto enough_span = [](const std::vector<Eigen::Vector2d>& pts) {
        if (pts.size() < kMinSurfacePoints) return false;
        return pts.back().x() - pts.front().x() >= kMinStableSurfaceSpan;
    };
    if (!enough_span(left_region) || !enough_span(right_region)) {
        if (fallback_reason) *fallback_reason = "missing_gap_insufficient_span";
        return {};
    }

    auto candidate_quality_ok = [max_acceptable_rms](
                                        const SurfaceCandidate& candidate) {
        const double slope = std::abs(line_slope(candidate.line));
        const double roughness = candidate.rms / std::max(candidate.span, 1e-9);
        return candidate.points.size() >= kMinSurfacePoints &&
               candidate.span >= kMinStableSurfaceSpan &&
               slope <= kMaxPlatformSlopeLimit &&
               candidate.rms <= max_acceptable_rms && roughness <= 35.0;
    };

    auto adjacent_surface =
            [&](const std::vector<Eigen::Vector2d>& region,
                bool use_right_edge) -> std::vector<Eigen::Vector2d> {
        const double min_span =
                std::max(kMinStableSurfaceSpan, 6.0 * median_dx);
        const double max_span = std::max(30.0, min_span);
        size_t start = 0;
        size_t end = region.size() - 1;
        if (use_right_edge) {
            start = end;
            while (start > 0 &&
                   region[end].x() - region[start - 1].x() <= max_span) {
                --start;
            }
        } else {
            end = start;
            while (end + 1 < region.size() &&
                   region[end + 1].x() - region[start].x() <= max_span) {
                ++end;
            }
        }

        std::vector<Eigen::Vector2d> local(region.begin() + start,
                                           region.begin() + end + 1);
        if (local.size() < kMinSurfacePoints ||
            local.back().x() - local.front().x() < min_span) {
            return {};
        }
        SurfaceCandidate candidate = make_surface_candidate(local);
        if (!candidate_quality_ok(candidate)) return {};
        return robust_line_fit_inliers(local);
    };

    std::vector<Eigen::Vector2d> left_pts = adjacent_surface(left_region, true);
    std::vector<Eigen::Vector2d> right_pts =
            adjacent_surface(right_region, false);
    if (left_pts.empty() || right_pts.empty()) {
        if (fallback_reason)
            *fallback_reason = "missing_gap_no_reliable_surface";
        return {};
    }

    auto left_max_it = std::max_element(
            left_pts.begin(), left_pts.end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.x() < b.x();
            });
    auto right_min_it = std::min_element(
            right_pts.begin(), right_pts.end(),
            [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                return a.x() < b.x();
            });
    if (right_min_it->x() - left_max_it->x() < missing_gap_threshold) {
        if (fallback_reason) *fallback_reason = "missing_gap_boundary_overlap";
        return {};
    }

    if (left_pts.size() < kMinSurfacePoints ||
        right_pts.size() < kMinSurfacePoints) {
        if (fallback_reason)
            *fallback_reason = "missing_gap_insufficient_inliers";
        return {};
    }

    return {left_pts, right_pts};
}

// Detects left/right platform surfaces directly from raw ordered TIFF points
// without B-spline upsampling. Returns empty vector on any failure to
// trigger automatic fallback to the B-spline path.
std::vector<std::vector<Eigen::Vector2d>> fast_path_detect_platforms(
        const std::vector<Eigen::Vector2d>& raw_pts,
        std::string* fallback_reason) {
    constexpr double kFastMinPoints = 20;
    constexpr double kFastMinSpan = 3.0;
    constexpr double kFastReliableMinSpan = 6.0;
    constexpr int kSmoothWindow = 3;

    if (raw_pts.size() < kFastMinPoints) {
        if (fallback_reason) *fallback_reason = "too_few_points";
        return {};
    }

    // Sort by x for ordered processing
    std::vector<Eigen::Vector2d> pts = raw_pts;
    std::sort(pts.begin(), pts.end(),
              [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                  if (a.x() == b.x()) return a.y() < b.y();
                  return a.x() < b.x();
              });

    auto missing_gap_groups =
            detect_missing_gap_platforms(pts, fallback_reason);
    if (!missing_gap_groups.empty()) return missing_gap_groups;
    const bool has_missing_gap_reject =
            fallback_reason != nullptr &&
            fallback_reason->rfind("missing_gap_", 0) == 0;

    // Verify regular x-spacing
    std::vector<double> x_steps;
    x_steps.reserve(pts.size());
    for (size_t i = 1; i < pts.size(); ++i) {
        double dx = pts[i].x() - pts[i - 1].x();
        if (dx <= 1e-12) {
            if (fallback_reason && !has_missing_gap_reject)
                *fallback_reason = "irregular_x_spacing";
            return {};
        }
        x_steps.push_back(dx);
    }
    double median_dx = median_value(x_steps);
    for (double dx : x_steps) {
        if (dx > 3.0 * median_dx || dx < median_dx * 0.3) {
            if (fallback_reason && !has_missing_gap_reject)
                *fallback_reason = "irregular_x_spacing";
            return {};
        }
    }

    // Light smoothing: moving average
    std::vector<Eigen::Vector2d> smoothed(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        double sum_x = 0.0, sum_y = 0.0;
        int count = 0;
        for (int j = -kSmoothWindow / 2; j <= kSmoothWindow / 2; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx >= 0 && idx < static_cast<int>(pts.size())) {
                sum_x += pts[idx].x();
                sum_y += pts[idx].y();
                ++count;
            }
        }
        smoothed[i] = Eigen::Vector2d(sum_x / count, sum_y / count);
    }

    // Adaptive slope threshold from local slope distribution
    std::vector<double> abs_slopes;
    for (size_t i = 1; i < smoothed.size(); ++i) {
        double dx = smoothed[i].x() - smoothed[i - 1].x();
        if (std::abs(dx) < 1e-12) continue;
        abs_slopes.push_back(
                std::abs((smoothed[i].y() - smoothed[i - 1].y()) / dx));
    }
    if (abs_slopes.empty()) {
        if (fallback_reason) *fallback_reason = "no_valid_slopes";
        return {};
    }
    double median_slope = median_value(abs_slopes);
    double slope_threshold = std::max(0.35, median_slope * 3.0);

    struct EdgeSpan {
        bool valid = false;
        double left_x = 0.0;
    };
    EdgeSpan primary_edge;
    {
        std::vector<double> local_slopes(smoothed.size() - 1, 0.0);
        size_t max_slope_idx = 0;
        double max_slope = -1.0;
        for (size_t i = 1; i < smoothed.size(); ++i) {
            const double dx = smoothed[i].x() - smoothed[i - 1].x();
            if (std::abs(dx) < 1e-12) continue;
            const double slope =
                    std::abs((smoothed[i].y() - smoothed[i - 1].y()) / dx);
            local_slopes[i - 1] = slope;
            if (slope > max_slope) {
                max_slope = slope;
                max_slope_idx = i - 1;
            }
        }

        const double edge_slope_threshold =
                std::max(slope_threshold * 2.0, median_slope * 8.0);
        if (max_slope >= edge_slope_threshold) {
            size_t run_begin = max_slope_idx;
            while (run_begin > 0 &&
                   local_slopes[run_begin - 1] >= edge_slope_threshold) {
                --run_begin;
            }
            size_t run_end = max_slope_idx;
            while (run_end + 1 < local_slopes.size() &&
                   local_slopes[run_end + 1] >= edge_slope_threshold) {
                ++run_end;
            }
            primary_edge.valid = true;
            primary_edge.left_x = pts[run_begin].x();
        }
    }

    // Compute local slopes and flag flat regions
    std::vector<bool> is_flat(pts.size(), false);
    for (size_t i = 1; i < smoothed.size(); ++i) {
        double dx = smoothed[i].x() - smoothed[i - 1].x();
        double slope = std::abs((smoothed[i].y() - smoothed[i - 1].y()) / dx);
        if (slope <= slope_threshold) {
            is_flat[i] = true;
            is_flat[i - 1] = true;
        }
    }

    // Extract contiguous flat segments
    struct FlatSegment {
        size_t start, end;
        double span;
    };
    std::vector<FlatSegment> segments;
    size_t seg_start = 0;
    bool in_segment = is_flat[0];
    for (size_t i = 1; i < is_flat.size(); ++i) {
        if (is_flat[i] && !in_segment) {
            seg_start = i;
            in_segment = true;
        } else if (!is_flat[i] && in_segment) {
            double span = pts[i - 1].x() - pts[seg_start].x();
            if (span >= kFastMinSpan &&
                (i - 1 - seg_start + 1) >= kMinSurfacePoints)
                segments.push_back({seg_start, i - 1, span});
            in_segment = false;
        }
    }
    if (in_segment) {
        size_t last = is_flat.size() - 1;
        double span = pts[last].x() - pts[seg_start].x();
        if (span >= kFastMinSpan && (last - seg_start + 1) >= kMinSurfacePoints)
            segments.push_back({seg_start, last, span});
    }

    if (segments.size() < 2) {
        if (fallback_reason) *fallback_reason = "too_few_segments";
        return {};
    }

    // Adaptive RMS threshold: scale with Z-range
    double z_range = 0.0;
    {
        auto [ymin, ymax] = std::minmax_element(
                pts.begin(), pts.end(),
                [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                    return a.y() < b.y();
                });
        z_range = std::abs(ymax->y() - ymin->y());
    }
    double max_acceptable_rms = std::max(10.0, z_range * 0.025);
    double min_reliable_span = std::max(kFastReliableMinSpan, median_dx * 6.0);

    // Score segments: prefer long, low-RMS segments. Very short flat tails
    // are left for the B-spline fallback instead of being accepted as planes.
    struct ScoredSegment {
        FlatSegment seg;
        std::pair<Eigen::Vector2d, Eigen::Vector2d> line;
        double x_min;
        double x_max;
        double x_center;
        double rms;
        double slope;
        size_t point_count;
        double score;
    };
    std::vector<ScoredSegment> scored;
    for (const auto& seg : segments) {
        if (seg.span < min_reliable_span) continue;

        std::vector<Eigen::Vector2d> pts_seg(pts.begin() + seg.start,
                                             pts.begin() + seg.end + 1);
        auto line = fit_line_segment(pts_seg);
        double sum_sq = 0.0;
        for (const auto& pt : pts_seg) {
            double r = pt.y() - line_y_at_x(line, pt.x());
            sum_sq += r * r;
        }
        double rms = std::sqrt(sum_sq / pts_seg.size());
        double slope = std::abs(line_slope(line));
        if (rms > max_acceptable_rms) continue;
        if (slope > kMaxPlatformSlopeLimit) continue;

        const double x_min = line.first.x();
        const double x_max = line.second.x();
        const double x_center = 0.5 * (x_min + x_max);
        double score = 2.0 * seg.span - 6.0 * rms - 2.0 * slope;
        scored.push_back(
                {seg, line, x_min, x_max, x_center, rms, slope,
                 pts_seg.size(), score});
    }
    if (scored.size() < 2) {
        if (fallback_reason) *fallback_reason = "too_few_scored_segments";
        return {};
    }

    // Prefer a transition/gap-first interpretation when the slice contains
    // two credible support surfaces separated by a high-change or depressed
    // band. This rejects short valley-floor fragments before the legacy
    // edge-proximity selector can choose them as a reference surface.
    {
        const double valid_x_span = pts.back().x() - pts.front().x();
        const double reference_min_span =
                std::max({12.0 * median_dx, min_reliable_span,
                          std::min(valid_x_span * 0.08, 30.0 * median_dx)});
        const size_t reference_min_points =
                std::max<size_t>(8, 2 * kMinSurfacePoints);
        const double reference_slope_limit =
                std::clamp(std::max(0.6, median_slope * 6.0), 0.6, 2.5);
        const double transition_slope_threshold =
                std::max({0.5, slope_threshold * 1.5, median_slope * 5.0});
        const double min_transition_height =
                std::max(0.35, z_range * 0.03);

        auto is_reference_like = [&](const ScoredSegment& candidate) {
            return candidate.seg.span >= reference_min_span &&
                   candidate.point_count >= reference_min_points &&
                   candidate.slope <= reference_slope_limit &&
                   candidate.rms <= max_acceptable_rms;
        };

        const ScoredSegment* transition_left = nullptr;
        const ScoredSegment* transition_right = nullptr;
        double best_transition_score = -std::numeric_limits<double>::max();

        for (const auto& left : scored) {
            if (!is_reference_like(left)) continue;
            for (const auto& right : scored) {
                if (&left == &right || !is_reference_like(right)) continue;
                if (left.x_max + 2.0 * median_dx > right.x_min) continue;

                const double gap_width = right.x_min - left.x_max;
                bool has_intermediate_reference = false;
                for (const auto& middle : scored) {
                    if (&middle == &left || &middle == &right) continue;
                    if (!is_reference_like(middle)) continue;
                    if (middle.x_center > left.x_max &&
                        middle.x_center < right.x_min) {
                        has_intermediate_reference = true;
                        break;
                    }
                }
                if (has_intermediate_reference) continue;

                double max_gap_slope = 0.0;
                double valley_depth = 0.0;
                int gap_point_count = 0;

                const double left_y = line_y_at_x(left.line, left.x_max);
                const double right_y = line_y_at_x(right.line, right.x_min);
                const double transition_height = std::abs(right_y - left_y);

                const size_t gap_begin = left.seg.end + 1;
                const size_t gap_end =
                        right.seg.start > 0 ? right.seg.start - 1 : 0;
                if (gap_begin <= gap_end && gap_end < smoothed.size()) {
                    for (size_t i = gap_begin; i <= gap_end; ++i) {
                        ++gap_point_count;
                        const double x = pts[i].x();
                        const double t = gap_width > 1e-12
                                                 ? (x - left.x_max) / gap_width
                                                 : 0.0;
                        const double bridge_y =
                                left_y + std::clamp(t, 0.0, 1.0) *
                                                 (right_y - left_y);
                        valley_depth =
                                std::max(valley_depth, bridge_y - pts[i].y());

                        if (i > gap_begin) {
                            const double dx =
                                    smoothed[i].x() - smoothed[i - 1].x();
                            if (std::abs(dx) > 1e-12) {
                                max_gap_slope = std::max(
                                        max_gap_slope,
                                        std::abs((smoothed[i].y() -
                                                  smoothed[i - 1].y()) /
                                                 dx));
                            }
                        }
                    }
                }

                const bool has_missing_band =
                        gap_point_count == 0 && gap_width >= 3.0 * median_dx;
                const bool has_step_transition =
                        max_gap_slope >= transition_slope_threshold ||
                        transition_height >= min_transition_height;
                const bool has_valley_transition =
                        valley_depth >= min_transition_height;
                if (!has_missing_band && !has_step_transition &&
                    !has_valley_transition) {
                    continue;
                }

                const double support_span =
                        std::min(left.seg.span, 80.0 * median_dx) +
                        std::min(right.seg.span, 80.0 * median_dx);
                const double transition_score =
                        20.0 * max_gap_slope + 6.0 * valley_depth +
                        3.0 * transition_height + 0.08 * support_span +
                        0.01 * right.x_max - 0.05 * gap_width -
                        4.0 * (left.rms + right.rms);
                if (transition_score > best_transition_score) {
                    best_transition_score = transition_score;
                    transition_left = &left;
                    transition_right = &right;
                }
            }
        }

        if (transition_left && transition_right) {
            std::vector<Eigen::Vector2d> left_pts(
                    pts.begin() + transition_left->seg.start,
                    pts.begin() + transition_left->seg.end + 1);
            std::vector<Eigen::Vector2d> right_pts(
                    pts.begin() + transition_right->seg.start,
                    pts.begin() + transition_right->seg.end + 1);

            left_pts = robust_line_fit_inliers(left_pts);
            right_pts = robust_line_fit_inliers(right_pts);
            if (left_pts.size() >= kMinSurfacePoints &&
                right_pts.size() >= kMinSurfacePoints) {
                return {left_pts, right_pts};
            }
        }
    }

    // Measurement support should describe the local surfaces adjacent to the
    // main step transition. For the right side we keep the existing rightmost
    // reliable support preference because it avoids short low valley artifacts;
    // for the left side, choose the reliable segment closest to the transition
    // instead of the leftmost stable fragment.
    const ScoredSegment* left_best = nullptr;
    const ScoredSegment* right_best = nullptr;
    double best_left_edge_distance = std::numeric_limits<double>::max();
    double best_right_x = -std::numeric_limits<double>::max();

    for (const auto& candidate : scored) {
        if (primary_edge.valid &&
            candidate.x_center <= primary_edge.left_x + median_dx) {
            const double edge_distance =
                    std::max(0.0, primary_edge.left_x - candidate.x_max);
            if (edge_distance < best_left_edge_distance ||
                (std::abs(edge_distance - best_left_edge_distance) < 1e-9 &&
                 (!left_best || candidate.score > left_best->score))) {
                best_left_edge_distance = edge_distance;
                left_best = &candidate;
            }
        }
        if (candidate.x_max > best_right_x ||
            (std::abs(candidate.x_max - best_right_x) < 1e-9 &&
             (!right_best || candidate.score > right_best->score))) {
            best_right_x = candidate.x_max;
            right_best = &candidate;
        }
    }
    if (!left_best) {
        double best_left_x = std::numeric_limits<double>::max();
        for (const auto& candidate : scored) {
            if (candidate.x_min < best_left_x ||
                (std::abs(candidate.x_min - best_left_x) < 1e-9 &&
                 (!left_best || candidate.score > left_best->score))) {
                best_left_x = candidate.x_min;
                left_best = &candidate;
            }
        }
    }

    if (!left_best || !right_best || left_best == right_best) {
        if (fallback_reason) *fallback_reason = "no_confident_platform_pair";
        return {};
    }

    // Verify enough x-separation between left and right
    if (right_best->x_min - left_best->x_max < median_dx * 2.0) {
        if (fallback_reason) *fallback_reason = "insufficient_x_separation";
        return {};
    }

    // Verify segment quality with adaptive thresholds
    if (left_best->rms > max_acceptable_rms) {
        if (fallback_reason) *fallback_reason = "left_high_rms";
        return {};
    }
    if (right_best->rms > max_acceptable_rms) {
        if (fallback_reason) *fallback_reason = "right_high_rms";
        return {};
    }

    // Verify segment slopes are stable (not near-vertical)
    if (left_best->slope > kMaxPlatformSlopeLimit ||
        right_best->slope > kMaxPlatformSlopeLimit) {
        if (fallback_reason) *fallback_reason = "platform_too_steep";
        return {};
    }

    // Materialize inlier points for both segments
    std::vector<Eigen::Vector2d> left_pts(pts.begin() + left_best->seg.start,
                                          pts.begin() + left_best->seg.end + 1);
    std::vector<Eigen::Vector2d> right_pts(
            pts.begin() + right_best->seg.start,
            pts.begin() + right_best->seg.end + 1);

    // Apply robust line fit filtering for cleaner inliers
    left_pts = robust_line_fit_inliers(left_pts);
    right_pts = robust_line_fit_inliers(right_pts);

    if (left_pts.size() < kMinSurfacePoints ||
        right_pts.size() < kMinSurfacePoints) {
        if (fallback_reason) *fallback_reason = "insufficient_inliers";
        return {};
    }

    return {left_pts, right_pts};
}

}  // namespace

void GapStepDetection::detect_gap_step(
        std::shared_ptr<geometry::PointCloud> cloud,
        Eigen::Vector3d transformation_matrix,
        bool debug_mode) {
    // debug mode
    if (debug_mode) {
        utility::filesystem::MakeDirectory("./bspline");
    }

    // LOG_DEBUG("Slice along Y-axis");
    //  slice along y axis
    slice_along_y(cloud, transformation_matrix);

    // bspline interpolation
    double height_threshold = 0.01;
    lineSegments corners;
    bspline_interpolation(cloud, height_threshold, corners, debug_mode);

    // calculate the gap step result
    double gap_step = 0.0, step_width = 0.0;
    calculate_gap_step(corners, gap_step, step_width);
}

void GapStepDetection::detect_gap_step_dll(
        std::shared_ptr<geometry::PointCloud> cloud,
        Eigen::Vector3d transformation_matrix,
        double& gap_step,
        double& step_width,
        bool debug_mode) {
    // debug mode
    if (debug_mode) {
        utility::filesystem::MakeDirectory("./bspline");
    }
    // std::cout << "1" << std::endl;

    // LOG_DEBUG("Slice along Y-axis");
    //  slice along y axis
    slice_along_y(cloud, transformation_matrix);
    // std::cout << "2" << std::endl;

    // bspline interpolation
    double height_threshold = 0.01;
    lineSegments corners;
    bspline_interpolation(cloud, height_threshold, corners, debug_mode);
    // std::cout << "3" << std::endl;

    // calculate the gap step result
    // double gap_step = 0.0, step_width = 0.0;
    calculate_gap_step(corners, gap_step, step_width);
    // std::cout << "4" << std::endl;
}

void GapStepDetection::detect_gap_step_dll_plot(
        std::shared_ptr<geometry::PointCloud> cloud,
        Eigen::Vector3d transformation_matrix,
        double& gap_step,
        double& step_width,
        double& height_threshold,
        std::vector<std::vector<double>>& temp_res,
        std::string& debug_path,
        bool debug_mode) {
    // debug mode
    if (debug_mode) {
        utility::filesystem::MakeDirectory_dll(
                debug_path);  //"C:\\Users\\Administrator\\Desktop\\res\\bspline"
        ensure_trailing_path_separator(debug_path);
    }
    // std::cout << "1" << std::endl;

    // LOG_DEBUG("Slice along Y-axis");
    //  slice along y axis
    slice_along_y(cloud, transformation_matrix);
    // std::cout << "2" << std::endl;

    std::vector<double> LHT_width;
    bool LHT = false;
    lineSegments corners;
    std::vector<std::vector<Eigen::Vector2d>> left_surface, right_surface;
    bspline_interpolation_dll2(cloud, height_threshold, corners, LHT_width,
                               debug_path, LHT, debug_mode, &left_surface,
                               &right_surface);

    auto measurements = collect_slice_measurements(
            corners, LHT_width, LHT, &left_surface, &right_surface);
    measurements = filter_slices_by_3d_consistency(
            measurements, transformation_matrix, debug_mode ? debug_path : "");
    fill_result_from_measurements(measurements, gap_step, step_width,
                                  &temp_res);
    if (debug_mode) {
        write_slice_measurements_csv(debug_path, measurements);
        mark_rejected_debug_images(debug_path, measurements);
    }
    // std::cout << "4" << std::endl;
}
bool GapStepDetection::detect_gap_step_dll_plot2(
        std::shared_ptr<geometry::PointCloud> cloud,
        Eigen::Vector3d transformation_matrix,
        double& gap_step,
        double& step_width,
        double& height_threshold,
        std::vector<std::vector<double>>& temp_res,
        std::string& debug_path,
        bool LHT,
        bool debug_mode) {
    if (!cloud || cloud->points_.empty()) {
        LOG_ERROR("detect_gap_step_dll_plot2: input cloud is empty");
        gap_step = -1.0;
        step_width = -1.0;
        return false;
    }
    if (cloud->source_point_count_ > 0) {
        const double invalid_ratio =
                static_cast<double>(cloud->invalid_point_count_) /
                static_cast<double>(cloud->source_point_count_);
        constexpr double kMaxInvalidInputRatio = 0.25;
        if (invalid_ratio > kMaxInvalidInputRatio) {
            LOG_ERROR(
                    "detect_gap_step_dll_plot2: invalid TIFF pixel ratio {} "
                    "is above limit {}",
                    invalid_ratio, kMaxInvalidInputRatio);
            gap_step = -1.0;
            step_width = -1.0;
            return false;
        }
    }
    gap_step = -1.0;
    step_width = -1.0;

    try {
        detect_gap_step_dll_plot2_impl(cloud, transformation_matrix, gap_step,
                                       step_width, height_threshold, temp_res,
                                       debug_path, LHT, debug_mode);
        return true;
    } catch (...) {
        LOG_ERROR(
                "detect_gap_step_dll_plot2: caught a crash, "
                "returning error values");
        return false;
    }
}

void GapStepDetection::detect_gap_step_dll_plot2_impl(
        std::shared_ptr<geometry::PointCloud> cloud,
        Eigen::Vector3d transformation_matrix,
        double& gap_step,
        double& step_width,
        double& height_threshold,
        std::vector<std::vector<double>>& temp_res,
        std::string& debug_path,
        bool LHT,
        bool debug_mode) {
    ProfileScope profile("detect_gap_step_dll_plot2_impl");
    // debug mode
    if (debug_mode) {
        utility::filesystem::MakeDirectory_dll(debug_path);
        ensure_trailing_path_separator(debug_path);
    }

    // slice along y axis
    slice_along_y(cloud, transformation_matrix);

    // bspline interpolation with surface point extraction
    std::vector<double> LHT_width;
    lineSegments corners;
    std::vector<std::vector<Eigen::Vector2d>> left_surface, right_surface;
    bspline_interpolation_dll2(cloud, height_threshold, corners, LHT_width,
                               debug_path, LHT, debug_mode, &left_surface,
                               &right_surface);

    // Collect measurements with surface points
    auto measurements = collect_slice_measurements(
            corners, LHT_width, LHT, &left_surface, &right_surface);
    annotate_slice_quality(measurements, cloud);

    // 3D consistency filter
    {
        std::string filter_debug_path = debug_mode ? debug_path : "";
        measurements = filter_slices_by_3d_consistency(
                measurements, transformation_matrix, filter_debug_path);
    }

    // Fill results from filtered measurements
    fill_result_from_measurements(measurements, gap_step, step_width,
                                  &temp_res);

    if (debug_mode) {
        write_slice_measurements_csv(debug_path, measurements);
        mark_rejected_debug_images(debug_path, measurements);
    }
}

void GapStepDetection::slice_along_y(geometry::PointCloud::Ptr cloud,
                                     Eigen::Vector3d transformation_matrix) {
    ProfileScope profile("slice_along_y");
    bool has_normals = cloud->HasNormals();
    // std::cout << transformation_matrix << std::endl;
    // std::cout << "has points: " << cloud->points_.size() << std::endl;
    if (has_normals) {
        Eigen::Vector3d min_bound = cloud->GetMinBound();
        Eigen::Vector3d max_bound = cloud->GetMaxBound();
        // int num_slice = (int)(((max_bound.y() - min_bound.y()) /
        //                        transformation_matrix.y()) +
        //                       1);
        int num_slice = static_cast<int>((max_bound.y() - min_bound.y()) /
                                                 transformation_matrix.y() +
                                         0.5) +
                        1;
        std::vector<double> y_slice_peaks(num_slice, 0);
        cloud->y_slice_peaks = y_slice_peaks;
        cloud->y_slices_.resize(num_slice);
        cloud->ny_slices_.resize(num_slice);
        cloud->y_slice_idxs.resize(num_slice);
        for (size_t j = 0; j < cloud->points_.size(); j++) {
            auto pt = cloud->points_[j];
            auto n = cloud->normals_[j];
            int slice_idx =
                    (int)((pt.y() - min_bound.y()) / transformation_matrix.y());
            if (slice_idx < 0) slice_idx = 0;
            if (slice_idx >= num_slice) slice_idx = num_slice - 1;
            if (cloud->y_slice_peaks[slice_idx] <= pt.z()) {
                cloud->y_slice_peaks[slice_idx] = pt.z();
            }
            cloud->y_slices_[slice_idx].emplace_back(
                    Eigen::Vector2d(pt.x(), pt.z()));
            cloud->ny_slices_[slice_idx].emplace_back(
                    Eigen::Vector3d(n.x(), n.y(), n.z()));
            cloud->y_slice_idxs[slice_idx].emplace_back(j);
        }
    } else {
        Eigen::Vector3d min_bound = cloud->GetMinBound();
        Eigen::Vector3d max_bound = cloud->GetMaxBound();
        int num_slice = static_cast<int>((max_bound.y() - min_bound.y()) /
                                                 transformation_matrix.y() +
                                         0.5) +
                        1;

        std::vector<double> y_slice_peaks(num_slice, 0);
        cloud->y_slice_peaks = y_slice_peaks;
        cloud->y_slices_.resize(num_slice);

        for (size_t j = 0; j < cloud->points_.size(); j++) {
            auto pt = cloud->points_[j];
            int slice_idx =
                    (int)((pt.y() - min_bound.y()) / transformation_matrix.y());
            if (slice_idx < 0) slice_idx = 0;
            if (slice_idx >= num_slice) slice_idx = num_slice - 1;
            if (cloud->y_slice_peaks[slice_idx] <= pt.z()) {
                cloud->y_slice_peaks[slice_idx] = pt.z();
            }
            cloud->y_slices_[slice_idx].emplace_back(
                    Eigen::Vector2d(pt.x(), pt.z()));
        }
    }
}

void GapStepDetection::bspline_interpolation(geometry::PointCloud::Ptr cloud,
                                             double height_threshold,
                                             lineSegments& corners,
                                             bool debug_mode) {
    // use common part to fit a curve
    core::PlaneDetection plane_detector;
    std::vector<double> step_height;
    step_height.resize(cloud->y_slices_.size());
    corners.resize(cloud->y_slices_.size());
    std::fill(corners.begin(), corners.end(), invalid_corner());
    std::fill(step_height.begin(), step_height.end(), -255.0);

#pragma omp parallel for
    for (int i = 0; i < cloud->y_slices_.size(); i++) {
        // Cubic B-spline needs at least degree+1 = 4 control points
        if (cloud->y_slices_[i].size() < 4) continue;
        int sampled_pts = adaptive_sample_count(cloud->y_slices_[i].size());
        std::vector<Eigen::Vector2d> resampled_pts =
                plane_detector.resample_a_curve(cloud->y_slices_[i],
                                                sampled_pts, i, false);
        plane_detector.fit_a_curve(resampled_pts, sampled_pts, i, debug_mode);

        // compute the derivative
        std::vector<std::vector<Eigen::Vector2d>> groups =
                group_by_derivative(resampled_pts);

        std::vector<std::vector<Eigen::Vector2d>> filter_groups =
                statistics_filter(groups);

        lineSegments lines = line_segment(filter_groups);
        if (lines.size() < 2) {
            step_height[i] = -255.0;
            continue;
        }
        step_height[i] = std::abs(lines[0].first.y() - lines[1].first.y());

        std::vector<std::vector<Eigen::Vector2d>> intersections;
        compute_step_width(resampled_pts, lines, intersections,
                           height_threshold);
        // put two corners corresponding to the slice to container
        if (intersections.size() > 2 && intersections[2].size() >= 2) {
            corners[i] =
                    std::make_pair(intersections[2][0], intersections[2][1]);
        } else {
            corners[i] = invalid_corner();
        }

        if (debug_mode) {
            plot_clusters(resampled_pts, filter_groups, lines, intersections,
                          i);
        }
    }

    // return sampled_map;
}

void GapStepDetection::bspline_interpolation_dll(
        geometry::PointCloud::Ptr cloud,
        double height_threshold,
        lineSegments& corners,
        std::string& debug_path,
        bool debug_mode) {
    // use common part to fit a curve
    core::PlaneDetection plane_detector;
    std::vector<double> step_height;
    step_height.resize(cloud->y_slices_.size());
    corners.resize(cloud->y_slices_.size());
    std::fill(corners.begin(), corners.end(), invalid_corner());
    std::fill(step_height.begin(), step_height.end(), -255.0);

#pragma omp parallel for
    for (int i = 0; i < cloud->y_slices_.size(); i++) {
        // Cubic B-spline needs at least degree+1 = 4 control points
        if (cloud->y_slices_[i].size() < 4) continue;
        int sampled_pts = adaptive_sample_count(cloud->y_slices_[i].size());
        std::vector<Eigen::Vector2d> resampled_pts =
                plane_detector.resample_a_curve(cloud->y_slices_[i],
                                                sampled_pts, i, debug_mode);
        // compute the derivative
        std::vector<std::vector<Eigen::Vector2d>> groups =
                group_by_derivative(resampled_pts);

        std::vector<std::vector<Eigen::Vector2d>> filter_groups =
                statistics_filter(groups);

        lineSegments lines = line_segment(filter_groups);
        if (lines.size() < 2) {
            step_height[i] = -255.0;
            continue;
        }
        step_height[i] = std::abs(lines[0].first.y() - lines[1].first.y());

        std::vector<std::vector<Eigen::Vector2d>> intersections;
        compute_step_width(resampled_pts, lines, intersections,
                           height_threshold);
        // put two corners corresponding to the slice to container
        if (intersections.size() > 2 && intersections[2].size() >= 2) {
            corners[i] =
                    std::make_pair(intersections[2][0], intersections[2][1]);
        } else {
            corners[i] = invalid_corner();
        }

        if (debug_mode) {
            plot_clusters_dll(resampled_pts, groups, lines, intersections,
                              debug_path, i);
        }
    }

    // return sampled_map;
}

void GapStepDetection::bspline_interpolation_dll2(
        geometry::PointCloud::Ptr cloud,
        double height_threshold,
        lineSegments& corners,
        std::vector<double>& LHT_width,
        std::string& debug_path,
        bool LHT,
        bool debug_mode,
        std::vector<std::vector<Eigen::Vector2d>>* left_surface,
        std::vector<std::vector<Eigen::Vector2d>>* right_surface) {
    ProfileScope profile("bspline_interpolation_dll2");
    // use common part to fit a curve
    core::PlaneDetection plane_detector;
    std::vector<double> step_height;
    size_t n_slices = cloud->y_slices_.size();
    step_height.resize(n_slices);
    corners.resize(n_slices);
    LHT_width.resize(n_slices);
    std::fill(corners.begin(), corners.end(), invalid_corner());
    std::fill(step_height.begin(), step_height.end(), -255.0);
    std::fill(LHT_width.begin(), LHT_width.end(), -255.0);

    if (left_surface) left_surface->resize(n_slices);
    if (right_surface) right_surface->resize(n_slices);

    const bool collect_profile = profile_enabled();
    std::atomic<long long> resample_us_sum{0};
    std::atomic<long long> group_us_sum{0};
    std::atomic<long long> filter_us_sum{0};
    std::atomic<long long> line_us_sum{0};
    std::atomic<long long> measure_us_sum{0};
    std::atomic<long long> plot_us_sum{0};
    std::atomic<int> fast_hit_count{0};
    std::atomic<int> fast_miss_count{0};
    std::vector<std::string> fast_miss_reasons(collect_profile ? n_slices : 0);

#pragma omp parallel for
    for (int i = 0; i < cloud->y_slices_.size(); i++) {
        // Cubic B-spline needs at least degree+1 = 4 control points
        if (cloud->y_slices_[i].size() < 4) continue;

        std::vector<Eigen::Vector2d> limit_pts;
        std::vector<std::vector<Eigen::Vector2d>> filter_groups;
        bool used_fast_path = false;
        std::vector<Eigen::Vector2d>
                resampled_pts;  // populated only for bspline path

        // --- Fast path: try raw-grid detection first ---
        {
            std::string fallback_reason;
            auto fast_groups = fast_path_detect_platforms(
                    cloud->y_slices_[i],
                    collect_profile ? &fallback_reason : nullptr);
            if (!fast_groups.empty()) {
                filter_groups = std::move(fast_groups);
                // Compute limit_pts from filter_groups for downstream use
                for (int g = 0; g < 2 && g < filter_groups.size(); ++g) {
                    if (filter_groups[g].empty()) continue;
                    auto [min_it, max_it] = std::minmax_element(
                            filter_groups[g].begin(), filter_groups[g].end(),
                            [](const Eigen::Vector2d& a,
                               const Eigen::Vector2d& b) {
                                return a.x() < b.x();
                            });
                    limit_pts.push_back(*min_it);
                    limit_pts.push_back(*max_it);
                }
                used_fast_path = true;
                if (collect_profile)
                    fast_hit_count.fetch_add(1, std::memory_order_relaxed);
            } else {
                if (collect_profile) {
                    fast_miss_count.fetch_add(1, std::memory_order_relaxed);
                    fast_miss_reasons[i] = fallback_reason;
                }
            }
        }

        if (!used_fast_path) {
            // --- B-spline fallback path ---
            int sampled_pts = adaptive_sample_count(cloud->y_slices_[i].size());
            ProfileClock::time_point step_start;
            if (collect_profile) step_start = ProfileClock::now();
            resampled_pts = plane_detector.resample_a_curve(
                    cloud->y_slices_[i], sampled_pts, i, false);
            if (collect_profile)
                resample_us_sum.fetch_add(elapsed_us(step_start),
                                          std::memory_order_relaxed);
            if (collect_profile) step_start = ProfileClock::now();
            auto groups = group_by_derivative_dll(resampled_pts);
            if (collect_profile)
                group_us_sum.fetch_add(elapsed_us(step_start),
                                       std::memory_order_relaxed);

            if (collect_profile) step_start = ProfileClock::now();
            filter_groups = statistics_filter(groups, limit_pts);
            if (collect_profile)
                filter_us_sum.fetch_add(elapsed_us(step_start),
                                        std::memory_order_relaxed);
        }

        double left_height_threshold = height_threshold,
               right_height_threshold = height_threshold;
        ProfileClock::time_point step_start;
        if (collect_profile) step_start = ProfileClock::now();
        lineSegments lines = line_segment(filter_groups);
        if (collect_profile)
            line_us_sum.fetch_add(elapsed_us(step_start),
                                  std::memory_order_relaxed);
        // Store surface points for 3D filtering
        if (left_surface && filter_groups.size() > 0)
            (*left_surface)[i] = filter_groups[0];
        if (right_surface && filter_groups.size() > 1)
            (*right_surface)[i] = filter_groups[1];
        if (lines.size() < 2) {
            step_height[i] = -255.0;
            LHT_width[i] = -255.0;
            if (debug_mode) {
                auto& plot_pts =
                        used_fast_path ? cloud->y_slices_[i] : resampled_pts;
                plot_clusters_dll(plot_pts, filter_groups, lines, {}, limit_pts,
                                  debug_path, i);
            }
            continue;
        }
        step_height[i] = std::abs(lines[0].first.y() - lines[1].first.y());
        std::vector<std::vector<Eigen::Vector2d>> intersections;
        std::vector<double> temp_width;
        if (collect_profile) step_start = ProfileClock::now();
        compute_step_width_dll(cloud->y_slices_[i], resampled_pts, lines,
                               intersections, temp_width, left_height_threshold,
                               right_height_threshold, limit_pts, LHT);
        if (!temp_width.empty()) {
            double max_val =
                    *std::max_element(temp_width.begin(), temp_width.end());
            LHT_width[i] = max_val;
        } else {
            LHT_width[i] = -255.0;
        }
        // put two corners corresponding to the slice to container
        if (intersections.size() > 2 && intersections[2].size() >= 2) {
            corners[i] =
                    std::make_pair(intersections[2][0], intersections[2][1]);
        } else {
            corners[i] = invalid_corner();
        }
        if (collect_profile)
            measure_us_sum.fetch_add(elapsed_us(step_start),
                                     std::memory_order_relaxed);

        if (debug_mode) {
            if (collect_profile) step_start = ProfileClock::now();
            auto& plot_pts =
                    used_fast_path ? cloud->y_slices_[i] : resampled_pts;
            if (LHT) {
                plot_clusters_dll_lht(cloud->y_slices_[i], plot_pts,
                                      filter_groups, lines, height_threshold,
                                      intersections, limit_pts, debug_path, i);
            } else {
                plot_clusters_dll(plot_pts, filter_groups, lines, intersections,
                                  limit_pts, debug_path, i);
            }
            if (collect_profile)
                plot_us_sum.fetch_add(elapsed_us(step_start),
                                      std::memory_order_relaxed);
        }
    }

    if (profile_enabled()) {
        std::cerr << "[profile] bspline_interpolation_dll2: "
                  << "fast_hit=" << fast_hit_count.load()
                  << " fast_miss=" << fast_miss_count.load() << std::endl;
        std::cerr << "[profile] bspline_interpolation_dll2.thread_sum: "
                  << "resample=" << resample_us_sum.load() / 1000.0 << " ms, "
                  << "group=" << group_us_sum.load() / 1000.0 << " ms, "
                  << "filter=" << filter_us_sum.load() / 1000.0 << " ms, "
                  << "line=" << line_us_sum.load() / 1000.0 << " ms, "
                  << "measure=" << measure_us_sum.load() / 1000.0 << " ms, "
                  << "plot=" << plot_us_sum.load() / 1000.0 << " ms"
                  << std::endl;
        if (fast_miss_count.load() > 0) {
            std::map<std::string, int> reason_counts;
            for (const auto& r : fast_miss_reasons)
                if (!r.empty()) reason_counts[r]++;
            std::cerr << "[profile] fast_path fallback reasons:";
            for (const auto& [reason, count] : reason_counts)
                std::cerr << " " << reason << "=" << count;
            std::cerr << std::endl;
        }
    }

    // return sampled_map;
}
std::vector<std::vector<Eigen::Vector2d>> GapStepDetection::group_by_derivative(
        std::vector<Eigen::Vector2d>& sampled_pts) {
    // Need at least 3 points for derivative computation
    if (sampled_pts.size() < 3) {
        return {};
    }
    std::vector<std::vector<Eigen::Vector2d>> surface_groups =
            extract_surface_candidates(sampled_pts);
    if (surface_groups.size() >= 2) return surface_groups;

    std::vector<Eigen::Vector2d> horiz_pts;
    for (int i = 0; i < sampled_pts.size(); i++) {
        double derivative;
        if (i == 0) {
            derivative = (sampled_pts[i + 1](1) - sampled_pts[i](1)) /
                         (sampled_pts[i + 1](0) - sampled_pts[i](0));
        } else if (i == sampled_pts.size() - 1) {
            derivative = (sampled_pts[i](1) - sampled_pts[i - 1](1)) /
                         (sampled_pts[i](0) - sampled_pts[i - 1](0));
        } else {
            derivative = (sampled_pts[i + 1](1) - sampled_pts[i - 1](1)) /
                         (sampled_pts[i + 1](0) - sampled_pts[i - 1](0));
        }
        if (derivative > -0.15 && derivative < 0.15) {
            horiz_pts.emplace_back(sampled_pts[i]);
        }
    }
    return group_horizontal_by_height(horiz_pts);
}

std::vector<std::vector<Eigen::Vector2d>>
GapStepDetection::group_by_derivative_dll(
        std::vector<Eigen::Vector2d>& sampled_pts
        /*        Eigen::Vector2d& max_derivative_point*/) {
    // Need at least 3 points for derivative computation
    if (sampled_pts.size() < 3) {
        return {};
    }
    std::vector<std::vector<Eigen::Vector2d>> surface_groups =
            extract_surface_candidates(sampled_pts);
    if (surface_groups.size() >= 2) return surface_groups;

    std::vector<Eigen::Vector2d> horiz_pts;
    for (int i = 0; i < sampled_pts.size(); i++) {
        double derivative;
        if (i == 0) {
            derivative = (sampled_pts[i + 1](1) - sampled_pts[i](1)) /
                         (sampled_pts[i + 1](0) - sampled_pts[i](0));
        } else if (i == sampled_pts.size() - 1) {
            derivative = (sampled_pts[i](1) - sampled_pts[i - 1](1)) /
                         (sampled_pts[i](0) - sampled_pts[i - 1](0));
        } else {
            derivative = (sampled_pts[i + 1](1) - sampled_pts[i - 1](1)) /
                         (sampled_pts[i + 1](0) - sampled_pts[i - 1](0));
        }
        if (derivative > -0.15 && derivative < 0.15) {
            horiz_pts.emplace_back(sampled_pts[i]);
        }
    }
    return group_horizontal_by_height(horiz_pts);
}

std::vector<std::vector<Eigen::Vector2d>> GapStepDetection::statistics_filter(
        std::vector<std::vector<Eigen::Vector2d>>& clusters) {
    std::vector<std::vector<Eigen::Vector2d>> filter_group_pts;
    for (int i = 0; i < clusters.size(); i++) {
        if (clusters[i].empty()) {
            filter_group_pts.push_back({});
            continue;
        }
        filter_group_pts.push_back(filter_surface_group(clusters[i], i == 1));
    }
    return filter_group_pts;
}

std::vector<std::vector<Eigen::Vector2d>> GapStepDetection::statistics_filter(
        std::vector<std::vector<Eigen::Vector2d>>& clusters,
        std::vector<Eigen::Vector2d>& limit_pts) {
    std::vector<std::vector<Eigen::Vector2d>> filter_group_pts;
    for (int i = 0; i < clusters.size(); i++) {
        if (clusters[i].empty()) {
            filter_group_pts.push_back({});
            continue;
        }
        filter_group_pts.push_back(filter_surface_group(clusters[i], i == 1));
    }
    std::vector<Eigen::Vector2d> temp_pts;

    for (int i = 0; i < 2 && i < filter_group_pts.size(); ++i) {
        if (filter_group_pts[i].empty()) continue;
        auto [min_it, max_it] = std::minmax_element(
                filter_group_pts[i].begin(), filter_group_pts[i].end(),
                [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                    return a.x() < b.x();
                });

        temp_pts.push_back(*min_it);  // left boundary
        temp_pts.push_back(*max_it);  // right boundary
    }
    // Need at least 4 temp_pts (2 clusters x 2 boundary points) for
    // the ordering logic below
    if (temp_pts.size() >= 4) {
        if (temp_pts[1].x() > temp_pts[2].x()) {
            limit_pts.push_back(temp_pts[3]);
            limit_pts.push_back(temp_pts[0]);
        } else {
            limit_pts.push_back(temp_pts[1]);
            limit_pts.push_back(temp_pts[2]);
        }
    }
    return filter_group_pts;
}

void GapStepDetection::plot_clusters(
        std::vector<Eigen::Vector2d>& resampled_pts,
        std::vector<std::vector<Eigen::Vector2d>>& clusters,
        lineSegments& line_segs,
        std::vector<std::vector<Eigen::Vector2d>> intersections,
        int img_id) {
    cv::Mat bg = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
    bg.setTo(cv::Scalar(255, 255, 255));
    std::vector<double> x_vec;
    std::vector<double> y_vec;

    for (int i = 0; i < resampled_pts.size(); i++) {
        x_vec.push_back(resampled_pts[i].x());
        y_vec.push_back(resampled_pts[i].y());
    }

    double x_min = *std::min_element(x_vec.begin(), x_vec.end());
    double x_max = *std::max_element(x_vec.begin(), x_vec.end());
    double y_min = *std::min_element(y_vec.begin(), y_vec.end());
    double y_max = *std::max_element(y_vec.begin(), y_vec.end());

    for (size_t i = 0; i < x_vec.size(); ++i) {
        int x = static_cast<int>((x_vec[i] - x_min) / (x_max - x_min) * 800);
        int y = static_cast<int>(500 -
                                 (y_vec[i] - y_min) / (y_max - y_min) * 500);
        cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
    }

    for (int i = 0; i < clusters.size() && i < line_segs.size(); i++) {
        if (i == 0) {
            double left_x = line_segs[i].first.x();
            double right_x = line_segs[i].second.x();
            double left_y = line_segs[i].first.y();
            double right_y = line_segs[i].second.y();
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
                // std::cout << pt.x() << " " << pt.y() << std::endl;
                // std::cout << x << " " << y << std::endl;
            }
            int line_x_left =
                    static_cast<int>((left_x - x_min) / (x_max - x_min) * 800);
            int line_x_right =
                    static_cast<int>((right_x - x_min) / (x_max - x_min) * 800);
            int line_y_left = static_cast<int>(
                    500 - (left_y - y_min) / (y_max - y_min) * 500);
            int line_y_right = static_cast<int>(
                    500 - (right_y - y_min) / (y_max - y_min) * 500);
            cv::line(bg, cv::Point(line_x_left, line_y_left),
                     cv::Point(line_x_right, line_y_right),
                     cv::Scalar(0, 0, 255), 1);
        } else {
            // i == 1
            double left_x = line_segs[i].first.x();
            double right_x = line_segs[i].second.x();
            double left_y = line_segs[i].first.y();
            double right_y = line_segs[i].second.y();
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), -1);
            }
            int line_x_left =
                    static_cast<int>((left_x - x_min) / (x_max - x_min) * 800);
            int line_x_right =
                    static_cast<int>((right_x - x_min) / (x_max - x_min) * 800);
            int line_y_left = static_cast<int>(
                    500 - (left_y - y_min) / (y_max - y_min) * 500);
            int line_y_right = static_cast<int>(
                    500 - (right_y - y_min) / (y_max - y_min) * 500);
            cv::line(bg, cv::Point(line_x_left, line_y_left),
                     cv::Point(line_x_right, line_y_right),
                     cv::Scalar(255, 0, 0), 1);
        }
    }

    for (int i = 0; i < intersections.size(); i++) {
        if (i == 0) {
            int tmp_y = 0;
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(0, 0, 255),
                               cv::MARKER_STAR, 10);
                tmp_y = y;
                // std::cout << "Line: " << std::endl;
                // std::cout << pt.x() << " " << pt.y() << std::endl;
                // std::cout << x << " " << y << std::endl;
            }

        } else if (i == 1) {
            int tmp_y = 0;
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(255, 0, 0),
                               cv::MARKER_STAR, 10);
                tmp_y = y;
            }
        } else {
            // egde points i== 2
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(255, 0, 0),
                               cv::MARKER_DIAMOND, 10);
            }
        }
    }
    draw_measurement_overlay(bg, intersections, x_min, x_max, y_min, y_max);
    cv::imwrite("./bspline/group_pts" + std::to_string(img_id) + ".jpg", bg);
}

void GapStepDetection::plot_clusters_dll(
        std::vector<Eigen::Vector2d>& resampled_pts,
        std::vector<std::vector<Eigen::Vector2d>>& clusters,
        lineSegments& line_segs,
        std::vector<std::vector<Eigen::Vector2d>> intersections,
        std::vector<Eigen::Vector2d>& limit_pts,
        std::string& debug_path,
        int img_id) {
    // cv::Mat bg = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
    cv::Mat bg = cv::Mat::zeros(510, 810, CV_8UC3);  // 创建一个空白图像
    bg.setTo(cv::Scalar(255, 255, 255));
    std::vector<double> x_vec;
    std::vector<double> y_vec;

    for (int i = 0; i < resampled_pts.size(); i++) {
        x_vec.push_back(resampled_pts[i].x());
        y_vec.push_back(resampled_pts[i].y());
    }

    double x_min = *std::min_element(x_vec.begin(), x_vec.end());
    double x_max = *std::max_element(x_vec.begin(), x_vec.end());
    double y_min = *std::min_element(y_vec.begin(), y_vec.end());
    double y_max = *std::max_element(y_vec.begin(), y_vec.end());
    const bool has_limit_pair = limit_pts.size() >= 2;
    const bool has_mid_low = limit_pts.size() >= 3;

    for (size_t i = 0; i < x_vec.size(); ++i) {
        int x = static_cast<int>((x_vec[i] - x_min) / (x_max - x_min) * 800);
        int y = static_cast<int>(500 -
                                 (y_vec[i] - y_min) / (y_max - y_min) * 500);
        if (has_limit_pair &&
            ((x_vec[i] == limit_pts[0].x() && y_vec[i] == limit_pts[0].y()) ||
             (x_vec[i] == limit_pts[1].x() && y_vec[i] == limit_pts[1].y()))) {
            // cv::circle(bg, cv::Point(x, y), 8, cv::Scalar(0, 0, 0),
            //            -1);
            cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(0, 0, 0),
                           cv::MARKER_SQUARE, 15);
        } else if (has_mid_low && x_vec[i] == limit_pts[2].x() &&
                   y_vec[i] == limit_pts[2].y()) {
            cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(0, 0, 255),
                           cv::MARKER_SQUARE, 10);
        } else {
            cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
        }
        // cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
    }
    // origin lines
    for (int i = 0; i < clusters.size() && i < line_segs.size(); i++) {
        if (i == 0) {
            double left_x = line_segs[i].first.x();
            double right_x = line_segs[i].second.x();
            double left_y = line_segs[i].first.y();
            double right_y = line_segs[i].second.y();
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
                // std::cout << pt.x() << " " << pt.y() << std::endl;
                // std::cout << x << " " << y << std::endl;
            }
            int line_x_left =
                    static_cast<int>((left_x - x_min) / (x_max - x_min) * 800);
            int line_x_right =
                    static_cast<int>((right_x - x_min) / (x_max - x_min) * 800);
            int line_y_left = static_cast<int>(
                    500 - (left_y - y_min) / (y_max - y_min) * 500);
            int line_y_right = static_cast<int>(
                    500 - (right_y - y_min) / (y_max - y_min) * 500);
            cv::line(bg, cv::Point(line_x_left, line_y_left),
                     cv::Point(line_x_right, line_y_right),
                     cv::Scalar(0, 0, 255), 1);
        } else {
            // i == 1
            double left_x = line_segs[i].first.x();
            double right_x = line_segs[i].second.x();
            double left_y = line_segs[i].first.y();
            double right_y = line_segs[i].second.y();
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), -1);
            }
            int line_x_left =
                    static_cast<int>((left_x - x_min) / (x_max - x_min) * 800);
            int line_x_right =
                    static_cast<int>((right_x - x_min) / (x_max - x_min) * 800);
            int line_y_left = static_cast<int>(
                    500 - (left_y - y_min) / (y_max - y_min) * 500);
            int line_y_right = static_cast<int>(
                    500 - (right_y - y_min) / (y_max - y_min) * 500);
            cv::line(bg, cv::Point(line_x_left, line_y_left),
                     cv::Point(line_x_right, line_y_right),
                     cv::Scalar(255, 0, 0), 1);
        }
    }
    // moved lines
    for (int i = 0; i < intersections.size(); i++) {
        if (i == 0) {
            int tmp_y = 0;
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(0, 0, 255),
                               cv::MARKER_STAR, 10);
                tmp_y = y;
                // std::cout << "Line: " << std::endl;
                // std::cout << pt.x() << " " << pt.y() << std::endl;
                // std::cout << x << " " << y << std::endl;
            }

        } else if (i == 1) {
            int tmp_y = 0;
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(255, 0, 0),
                               cv::MARKER_STAR, 10);
                tmp_y = y;
            }
        } else {
            // egde points i== 2
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(255, 0, 0),
                               cv::MARKER_DIAMOND, 10);
            }
        }
    }
    // std::cout << "debug_path:"
    //           << debug_path + "group_pts" + std::to_string(img_id) + ".jpg"
    //           << std::endl;
    draw_measurement_overlay(bg, intersections, x_min, x_max, y_min, y_max);
    cv::imwrite(debug_path + "group_pts" + std::to_string(img_id) + ".jpg", bg);
}

void GapStepDetection::plot_clusters_dll_lht(
        std::vector<Eigen::Vector2d>& cloud_pts,
        std::vector<Eigen::Vector2d>& resampled_pts,
        std::vector<std::vector<Eigen::Vector2d>>& clusters,
        lineSegments& line_segs,
        double& height_threshold,
        std::vector<std::vector<Eigen::Vector2d>> intersections,
        std::vector<Eigen::Vector2d>& limit_pts,
        std::string& debug_path,
        int img_id) {
    (void)cloud_pts;
    (void)height_threshold;
    plot_clusters_dll(resampled_pts, clusters, line_segs, intersections,
                      limit_pts, debug_path, img_id);
    return;

    // cv::Mat bg = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
    cv::Mat bg = cv::Mat::zeros(510, 810, CV_8UC3);  // 创建一个空白图像
    bg.setTo(cv::Scalar(255, 255, 255));
    if (cloud_pts.size() < 2) {
        plot_clusters_dll(resampled_pts, clusters, line_segs, intersections,
                          limit_pts, debug_path, img_id);
        return;
    }
    std::vector<double> x_vec;
    std::vector<double> y_vec;
    // sorted by prev step
    double maxWidth = (-1) * std::numeric_limits<double>::max();
    double stepX = cloud_pts[1][0] - cloud_pts[0][0];
    //无效区域标志点
    double x_start = 0.0, x_end = 0.0;
    double y_start = 0.0, y_end = 0.0;
    // Eigen::Vector2d start_pt_prev;
    // Eigen::Vector2d end_pt_after;
    double bottom_value = -10.0;  //如果没有交点，需要减小这个值
    int fig_shift = 2;
    bool flagC = false;

    for (int i = 0; i < cloud_pts.size() - 1; i++) {
        auto pt = cloud_pts[i];
        auto next_pt = cloud_pts[i + 1];
        if ((next_pt.x() - pt.x()) > 2 * stepX) {
            if ((next_pt.x() - pt.x()) > maxWidth) {
                flagC = true;
                maxWidth = next_pt.x() - pt.x();
                x_start = pt.x();
                x_end = next_pt.x();
                y_start = pt.y();
                y_end = next_pt.y();
                // if (i >= 1) {
                //     start_pt_prev = cloud_pts[i - 1];
                // } else {
                //     start_pt_prev = pt;
                // }
                // if (i >= 1) {
                //     start_pt_prev = cloud_pts[i - 1];
                // } else {
                //     start_pt_prev = pt;
                // }
            }
        }
    }
    //所有点都是连续的，画普通 debug 图
    if (!flagC) {
        plot_clusters_dll(resampled_pts, clusters, line_segs, intersections,
                          limit_pts, debug_path, img_id);
        return;
    }

    for (int i = 0; i < resampled_pts.size(); i++) {
        x_vec.push_back(resampled_pts[i].x());
        y_vec.push_back(resampled_pts[i].y());
    }

    double x_min = *std::min_element(x_vec.begin(), x_vec.end());
    double x_max = *std::max_element(x_vec.begin(), x_vec.end());
    double y_min = *std::min_element(y_vec.begin(), y_vec.end());
    double y_max = *std::max_element(y_vec.begin(), y_vec.end());

    // moved lines
    //动态计算平移后的直线与生成点之间的交点
    double left_height = line_segs[0].first.y() - height_threshold;
    double right_height = line_segs[1].first.y() - height_threshold;
    Eigen::Vector2d left_intersection, right_intersection;

    //平移后的高度小于等于y_start/y_end，该模式下不存在这种情况，画普通 debug 图
    if ((left_height - y_start) >= 0 || (right_height - y_end) >= 0) {
        plot_clusters_dll(resampled_pts, clusters, line_segs, intersections,
                          limit_pts, debug_path, img_id);
        return;
    }
    //求交点
    while ((y_start - left_height) * (bottom_value - left_height) >= 0 ||
           (bottom_value - right_height) * (y_end - right_height) >= 0) {
        bottom_value -= 1;
    }
    if ((y_start - left_height) * (bottom_value - left_height) < 0) {
        double denominator = bottom_value - y_start;
        double t = 0.0;
        if (denominator != 0) {
            t = (left_height - y_start) / denominator;
        }
        double u = x_start + t * (x_start + stepX - x_start);
        left_intersection = Eigen::Vector2d(u, left_height);
        // left_intersections.push_back(Eigen::Vector2d(u, left_height));
    }

    if ((bottom_value - right_height) * (y_end - right_height) < 0) {
        double denominator = y_end - bottom_value;
        double t = 0.0;
        if (denominator != 0) {
            t = (right_height - bottom_value) / denominator;
        }
        double u = x_end - stepX + t * stepX;
        right_intersection = Eigen::Vector2d(u, right_height);
        // left_intersections.push_back(Eigen::Vector2d(u, left_height));
    }
    //真实数据转图像数据后画图
    int x_left = static_cast<int>((left_intersection.x() - x_min) /
                                  (x_max - x_min) * 800);
    int y_left = static_cast<int>(500 - (left_intersection.y() - bottom_value) /
                                                (y_max - bottom_value) * 500);
    cv::drawMarker(bg, cv::Point(x_left, y_left + fig_shift),
                   cv::Scalar(0, 0, 255), cv::MARKER_STAR, 10);

    int x_right = static_cast<int>((right_intersection.x() - x_min) /
                                   (x_max - x_min) * 800);
    int y_right =
            static_cast<int>(500 - (right_intersection.y() - bottom_value) /
                                           (y_max - bottom_value) * 500);
    cv::drawMarker(bg, cv::Point(x_right, y_right + fig_shift),
                   cv::Scalar(255, 0, 0), cv::MARKER_STAR, 10);

    const bool has_limit_pair = limit_pts.size() >= 2;

    //原始点
    for (size_t i = 0; i < x_vec.size(); ++i) {
        if (x_vec[i] > (x_start - stepX) && x_vec[i] < (x_end + stepX)) {
            y_vec[i] = bottom_value;
        }
        int x = static_cast<int>((x_vec[i] - x_min) / (x_max - x_min) * 800);
        int y = static_cast<int>(500 - (y_vec[i] - bottom_value) /
                                               (y_max - bottom_value) * 500);
        if (has_limit_pair &&
            ((x_vec[i] == limit_pts[0].x() && y_vec[i] == limit_pts[0].y()) ||
             (x_vec[i] == limit_pts[1].x() && y_vec[i] == limit_pts[1].y()))) {
            // cv::circle(bg, cv::Point(x, y), 8, cv::Scalar(0, 0, 0),
            //            -1);
            cv::drawMarker(bg, cv::Point(x, y + fig_shift), cv::Scalar(0, 0, 0),
                           cv::MARKER_SQUARE, 15);
        } else {
            cv::circle(bg, cv::Point(x, y + fig_shift), 2,
                       cv::Scalar(0, 255, 0), -1);
        }
        // cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
    }
    // origin lines
    for (int i = 0; i < clusters.size() && i < line_segs.size(); i++) {
        if (i == 0) {
            double left_x = line_segs[i].first.x();
            double right_x = line_segs[i].second.x();
            double left_y = line_segs[i].first.y();
            double right_y = line_segs[i].second.y();
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - bottom_value) /
                                                       (y_max - bottom_value) *
                                                       500);
                cv::circle(bg, cv::Point(x, y + fig_shift), 2,
                           cv::Scalar(0, 0, 255), -1);
                // std::cout << pt.x() << " " << pt.y() << std::endl;
                // std::cout << x << " " << y << std::endl;
            }
            int line_x_left =
                    static_cast<int>((left_x - x_min) / (x_max - x_min) * 800);
            int line_x_right =
                    static_cast<int>((right_x - x_min) / (x_max - x_min) * 800);
            int line_y_left = static_cast<int>(
                    500 -
                    (left_y - bottom_value) / (y_max - bottom_value) * 500);
            int line_y_right = static_cast<int>(
                    500 -
                    (right_y - bottom_value) / (y_max - bottom_value) * 500);
            cv::line(bg, cv::Point(line_x_left, line_y_left + fig_shift),
                     cv::Point(line_x_right, line_y_right + fig_shift),
                     cv::Scalar(0, 0, 255), 1);
        } else {
            // i == 1
            double left_x = line_segs[i].first.x();
            double right_x = line_segs[i].second.x();
            double left_y = line_segs[i].first.y();
            double right_y = line_segs[i].second.y();
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - bottom_value) /
                                                       (y_max - bottom_value) *
                                                       500);
                cv::circle(bg, cv::Point(x, y + fig_shift), 2,
                           cv::Scalar(255, 0, 0), -1);
            }
            int line_x_left =
                    static_cast<int>((left_x - x_min) / (x_max - x_min) * 800);
            int line_x_right =
                    static_cast<int>((right_x - x_min) / (x_max - x_min) * 800);
            int line_y_left = static_cast<int>(
                    500 -
                    (left_y - bottom_value) / (y_max - bottom_value) * 500);
            int line_y_right = static_cast<int>(
                    500 -
                    (right_y - bottom_value) / (y_max - bottom_value) * 500);
            cv::line(bg, cv::Point(line_x_left, line_y_left + fig_shift),
                     cv::Point(line_x_right, line_y_right + fig_shift),
                     cv::Scalar(255, 0, 0), 1);
        }
    }
    draw_measurement_overlay(bg, intersections, x_min, x_max, bottom_value,
                             y_max, fig_shift);
    cv::imwrite(debug_path + "group_pts" + std::to_string(img_id) + ".jpg", bg);
}

void GapStepDetection::plot_clusters_dll(
        std::vector<Eigen::Vector2d>& resampled_pts,
        std::vector<std::vector<Eigen::Vector2d>>& clusters,
        lineSegments& line_segs,
        std::vector<std::vector<Eigen::Vector2d>> intersections,
        std::string& debug_path,
        int img_id) {
    cv::Mat bg = cv::Mat::zeros(500, 800, CV_8UC3);  // 创建一个空白图像
    bg.setTo(cv::Scalar(255, 255, 255));
    std::vector<double> x_vec;
    std::vector<double> y_vec;

    for (int i = 0; i < resampled_pts.size(); i++) {
        x_vec.push_back(resampled_pts[i].x());
        y_vec.push_back(resampled_pts[i].y());
    }

    double x_min = *std::min_element(x_vec.begin(), x_vec.end());
    double x_max = *std::max_element(x_vec.begin(), x_vec.end());
    double y_min = *std::min_element(y_vec.begin(), y_vec.end());
    double y_max = *std::max_element(y_vec.begin(), y_vec.end());

    for (size_t i = 0; i < x_vec.size(); ++i) {
        int x = static_cast<int>((x_vec[i] - x_min) / (x_max - x_min) * 800);
        int y = static_cast<int>(500 -
                                 (y_vec[i] - y_min) / (y_max - y_min) * 500);
        cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 255, 0), -1);
    }

    for (int i = 0; i < clusters.size() && i < line_segs.size(); i++) {
        if (i == 0) {
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
                // std::cout << pt.x() << " " << pt.y() << std::endl;
                // std::cout << x << " " << y << std::endl;
            }
            draw_line_on_plot(bg, line_segs[i], x_min, x_max, y_min, y_max,
                              cv::Scalar(0, 0, 255), 2);
        } else {
            // i == 1
            for (auto pt : clusters[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::circle(bg, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), -1);
            }
            draw_line_on_plot(bg, line_segs[i], x_min, x_max, y_min, y_max,
                              cv::Scalar(255, 0, 0), 2);
        }
    }

    for (int i = 0; i < intersections.size(); i++) {
        if (i == 0) {
            int tmp_y = 0;
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(0, 0, 255),
                               cv::MARKER_STAR, 10);
                tmp_y = y;
                // std::cout << "Line: " << std::endl;
                // std::cout << pt.x() << " " << pt.y() << std::endl;
                // std::cout << x << " " << y << std::endl;
            }

        } else if (i == 1) {
            int tmp_y = 0;
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(255, 0, 0),
                               cv::MARKER_STAR, 10);
                tmp_y = y;
            }
        } else {
            // egde points i== 2
            for (auto pt : intersections[i]) {
                int x = static_cast<int>((pt.x() - x_min) / (x_max - x_min) *
                                         800);
                int y = static_cast<int>(500 - (pt.y() - y_min) /
                                                       (y_max - y_min) * 500);
                cv::drawMarker(bg, cv::Point(x, y), cv::Scalar(255, 0, 0),
                               cv::MARKER_DIAMOND, 10);
            }
        }
    }
    // std::cout << "debug_path:"
    //           << debug_path + "group_pts" + std::to_string(img_id) + ".jpg"
    //           << std::endl;
    draw_measurement_overlay(bg, intersections, x_min, x_max, y_min, y_max);
    cv::imwrite(debug_path + "group_pts" + std::to_string(img_id) + ".jpg", bg);
}

GapStepDetection::lineSegments GapStepDetection::line_segment(
        std::vector<std::vector<Eigen::Vector2d>>& pt_groups) {
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> res;
    for (int i = 0; i < pt_groups.size(); i++) {
        if (pt_groups[i].size() < kMinSurfacePoints) continue;
        pt_groups[i] = robust_line_fit_inliers(pt_groups[i]);
        res.push_back(fit_line_segment(pt_groups[i]));
    }
    if (res.size() >= 2 && res[0].second.x() > res[1].first.x()) {
        std::swap(res[1], res[0]);
        if (pt_groups.size() >= 2) std::swap(pt_groups[1], pt_groups[0]);
    }
    return res;
}
// fix width calculate error
GapStepDetection::lineSegments GapStepDetection::line_segment_dll(
        std::vector<std::vector<Eigen::Vector2d>>& pt_groups,
        std::vector<std::vector<Eigen::Vector2d>>& filter_pt_groups,
        double& left_height_threshold,
        double& right_height_threshold) {
    std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>> res;
    std::vector<std::vector<double>> distance;
    std::vector<double> threshold_res;
    threshold_res.resize(filter_pt_groups.size());
    distance.resize(filter_pt_groups.size());
    for (int i = 0; i < filter_pt_groups.size(); i++) {
        if (filter_pt_groups[i].empty()) {
            res.push_back(std::make_pair(Eigen::Vector2d(0, 0),
                                         Eigen::Vector2d(0, 0)));
            continue;
        }
        filter_pt_groups[i] = robust_line_fit_inliers(filter_pt_groups[i]);
        res.push_back(fit_line_segment(filter_pt_groups[i]));
        // calculate every point distance to line
        for (const auto& dpt : pt_groups[i]) {
            const double line_y = line_y_at_x(res.back(), dpt.x());
            double dis = std::abs(dpt.y() - line_y);
            if (dpt.y() > line_y) {
                dis = -dis;
            }
            distance[i].push_back(dis);
        }
    }
    for (int i = 0; i < distance.size(); i++) {
        if (distance[i].empty()) {
            threshold_res[i] = 0.0;
            continue;
        }
        double sum =
                std::accumulate(distance[i].begin(), distance[i].end(), 0.0);
        threshold_res[i] = sum / distance[i].size();
    }
    if (res.size() >= 2 && res[0].second.x() > res[1].first.x()) {
        std::swap(res[1], res[0]);
        if (pt_groups.size() >= 2) std::swap(pt_groups[1], pt_groups[0]);
        if (filter_pt_groups.size() >= 2)
            std::swap(filter_pt_groups[1], filter_pt_groups[0]);
        std::swap(threshold_res[1], threshold_res[0]);
    }
    if (threshold_res.size() >= 2) {
        left_height_threshold = threshold_res[0];
        right_height_threshold = threshold_res[1];
    }
    return res;
}

void GapStepDetection::compute_step_width(
        std::vector<Eigen::Vector2d>& resampled_pts,
        lineSegments& line_segs,
        std::vector<std::vector<Eigen::Vector2d>>& intersections,
        double height_threshold) {
    (void)resampled_pts;
    (void)height_threshold;
    std::pair<Eigen::Vector2d, Eigen::Vector2d> left_line = line_segs[0];
    std::pair<Eigen::Vector2d, Eigen::Vector2d> right_line = line_segs[1];
    intersections.emplace_back(std::vector<Eigen::Vector2d>{left_line.second});
    intersections.emplace_back(std::vector<Eigen::Vector2d>{right_line.first});
    intersections.emplace_back(
            std::vector<Eigen::Vector2d>{left_line.second, right_line.first});
}
void GapStepDetection::compute_step_width_dll(
        std::vector<Eigen::Vector2d>& cloud_pts,
        std::vector<Eigen::Vector2d>& resampled_pts,
        lineSegments& line_segs,
        std::vector<std::vector<Eigen::Vector2d>>& intersections,
        std::vector<double>& temp_width,
        double& left_height_threshold,
        double& right_height_threshold,
        // Eigen::Vector2d& max_derivative_point,
        std::vector<Eigen::Vector2d>& limit_pts,
        bool LHT) {
    (void)cloud_pts;
    (void)resampled_pts;
    (void)left_height_threshold;
    (void)right_height_threshold;
    (void)LHT;

    std::pair<Eigen::Vector2d, Eigen::Vector2d> left_line = line_segs[0];
    std::pair<Eigen::Vector2d, Eigen::Vector2d> right_line = line_segs[1];
    Eigen::Vector2d left_boundary = left_line.second;
    Eigen::Vector2d right_boundary = right_line.first;
    if (limit_pts.size() >= 2) {
        const double split_x =
                0.5 * (left_line.second.x() + right_line.first.x());
        bool found_left_limit = false;
        bool found_right_limit = false;
        double left_limit_x = left_boundary.x();
        double right_limit_x = right_boundary.x();
        for (const auto& pt : limit_pts) {
            if (pt.x() <= split_x &&
                (!found_left_limit || pt.x() > left_limit_x)) {
                left_limit_x = pt.x();
                found_left_limit = true;
            }
            if (pt.x() >= split_x &&
                (!found_right_limit || pt.x() < right_limit_x)) {
                right_limit_x = pt.x();
                found_right_limit = true;
            }
        }
        if (found_left_limit && found_right_limit &&
            left_limit_x < right_limit_x) {
            left_boundary = Eigen::Vector2d(
                    left_limit_x, line_y_at_x(left_line, left_limit_x));
            right_boundary = Eigen::Vector2d(
                    right_limit_x, line_y_at_x(right_line, right_limit_x));
        }
    }

    intersections.emplace_back(std::vector<Eigen::Vector2d>{left_boundary});
    intersections.emplace_back(std::vector<Eigen::Vector2d>{right_boundary});
    intersections.emplace_back(
            std::vector<Eigen::Vector2d>{left_boundary, right_boundary});

    limit_pts.clear();
    limit_pts.push_back(left_boundary);
    limit_pts.push_back(right_boundary);
    temp_width.push_back(std::abs(right_boundary.x() - left_boundary.x()));
}

void GapStepDetection::calculate_gap_step(lineSegments& corners,
                                          double& gap_step,
                                          double& step_width) {
    auto measurements = collect_slice_measurements(corners, {}, false);
    fill_result_from_measurements(measurements, gap_step, step_width, nullptr);
    const int accepted_count =
            std::count_if(measurements.begin(), measurements.end(),
                          [](const SliceMeasurement& measurement) {
                              return measurement.accepted;
                          });
    if (accepted_count == 0) {
        LOG_WARN("calculate_gap_step: all slices are invalid");
    }
    LOG_INFO("calculate_gap_step: accepted {}/{} slices", accepted_count,
             measurements.size());
    LOG_INFO("gap step: {} step width: {}", gap_step, step_width);
}

void GapStepDetection::calculate_gap_step_dll_plot(
        lineSegments& corners,
        std::vector<double>& LHT_width,
        double& gap_step,
        double& step_width,
        std::vector<std::vector<double>>& temp_res,
        bool LHT) {
    auto measurements = collect_slice_measurements(corners, LHT_width, LHT);
    fill_result_from_measurements(measurements, gap_step, step_width,
                                  &temp_res);
    const int accepted_count =
            std::count_if(measurements.begin(), measurements.end(),
                          [](const SliceMeasurement& measurement) {
                              return measurement.accepted;
                          });
    if (accepted_count == 0) {
        LOG_WARN("calculate_gap_step_dll_plot: all slices invalid");
    }
    LOG_INFO("calculate_gap_step_dll_plot: accepted {}/{} slices",
             accepted_count, measurements.size());
    LOG_INFO("gap step: {} step width: {}", gap_step, step_width);
}

#ifdef HYMSON3D_TESTING
std::vector<std::vector<Eigen::Vector2d>>
GapStepDetection::test_group_by_derivative_dll(
        std::vector<Eigen::Vector2d>& sampled_pts) {
    return group_by_derivative_dll(sampled_pts);
}

std::vector<std::vector<Eigen::Vector2d>>
GapStepDetection::test_filtered_groups_dll(
        std::vector<Eigen::Vector2d>& sampled_pts) {
    auto groups = group_by_derivative_dll(sampled_pts);
    std::vector<Eigen::Vector2d> limit_pts;
    return statistics_filter(groups, limit_pts);
}

std::vector<double> GapStepDetection::test_group_line_slopes(
        std::vector<Eigen::Vector2d>& sampled_pts) {
    std::vector<double> slopes;
    auto groups = group_by_derivative_dll(sampled_pts);
    std::vector<Eigen::Vector2d> limit_pts;
    auto filtered_groups = statistics_filter(groups, limit_pts);
    auto lines = line_segment(filtered_groups);
    for (const auto& line : lines) {
        const double dx = line.second.x() - line.first.x();
        slopes.push_back(dx == 0.0 ? 0.0
                                   : (line.second.y() - line.first.y()) / dx);
    }
    return slopes;
}

std::vector<std::vector<Eigen::Vector2d>>
GapStepDetection::test_fast_path_detect_platforms(
        std::vector<Eigen::Vector2d>& raw_pts) {
    std::string fallback_reason;
    return fast_path_detect_platforms(raw_pts, &fallback_reason);
}

std::pair<Eigen::Vector2d, Eigen::Vector2d>
GapStepDetection::test_compute_step_boundaries(
        const std::vector<Eigen::Vector2d>& left_pts,
        const std::vector<Eigen::Vector2d>& right_pts,
        std::vector<Eigen::Vector2d> limit_pts) {
    std::vector<std::vector<Eigen::Vector2d>> groups{left_pts, right_pts};
    auto lines = line_segment(groups);
    std::vector<Eigen::Vector2d> cloud_pts;
    std::vector<Eigen::Vector2d> resampled_pts;
    std::vector<std::vector<Eigen::Vector2d>> intersections;
    std::vector<double> temp_width;
    double left_height_threshold = 0.0;
    double right_height_threshold = 0.0;
    compute_step_width_dll(cloud_pts, resampled_pts, lines, intersections,
                           temp_width, left_height_threshold,
                           right_height_threshold, limit_pts, true);
    if (intersections.size() <= 2 || intersections[2].size() < 2) {
        return invalid_corner();
    }
    return {intersections[2][0], intersections[2][1]};
}

int GapStepDetection::test_mark_rejected_debug_images(
        const std::string& debug_dir) {
    utility::filesystem::DeleteDirectory(debug_dir);
    utility::filesystem::MakeDirectoryHierarchy(debug_dir);

    std::string path = debug_dir;
    if (!path.empty() && path.back() != '/' && path.back() != '\\') path += "/";

    cv::Mat image(16, 16, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::imwrite(path + "group_pts0.jpg", image);
    cv::imwrite(path + "group_pts1.jpg", image);

    std::vector<SliceMeasurement> measurements(2);
    measurements[0].index = 0;
    measurements[0].accepted = true;
    measurements[1].index = 1;
    measurements[1].accepted = false;
    measurements[1].reject_reason = "width_outlier";

    mark_rejected_debug_images(path, measurements);

    const std::string accepted_path = path + "group_pts0.jpg";
    const std::string rejected_original_path = path + "group_pts1.jpg";
    const std::string rejected_marked_path =
            path + "group_pts1__REJECTED_width_outlier.jpg";

    if (!utility::filesystem::FileExists(accepted_path)) {
        std::cerr << "accepted debug image should keep original name"
                  << std::endl;
        return 1;
    }
    if (utility::filesystem::FileExists(rejected_original_path)) {
        std::cerr << "rejected debug image should not keep unmarked name"
                  << std::endl;
        return 1;
    }
    if (!utility::filesystem::FileExists(rejected_marked_path)) {
        std::cerr << "rejected debug image should be marked in filename"
                  << std::endl;
        return 1;
    }

    measurements[0].accepted = false;
    measurements[0].reject_reason = "right_surface_residual_outlier";
    measurements[1].accepted = true;
    cv::imwrite(rejected_original_path, image);
    mark_rejected_debug_images(path, measurements);

    if (utility::filesystem::FileExists(rejected_marked_path)) {
        std::cerr << "stale rejected marker should be removed when slice is "
                     "accepted"
                  << std::endl;
        return 1;
    }
    if (!utility::filesystem::FileExists(rejected_original_path)) {
        std::cerr << "newly accepted debug image should keep original name"
                  << std::endl;
        return 1;
    }
    if (!utility::filesystem::FileExists(
                path +
                "group_pts0__REJECTED_right_surface_residual_outlier.jpg")) {
        std::cerr << "newly rejected debug image should be marked" << std::endl;
        return 1;
    }

    return 0;
}

// Test: 3D consistency filter with synthetic slices
int GapStepDetection::test_3d_consistency_filter(const std::string& debug_dir) {
    const int n_slices = 20;
    const double y_step = 0.03;  // transformation_matrix.y()
    Eigen::Vector3d trans_mat(0.01, y_step, 0.001);

    // Helper: build a slice with known geometry
    auto make_slice = [](int idx, double left_x0, double left_z0,
                         double right_x0, double right_z0, double width,
                         double height, int n_left_pts = 10,
                         int n_right_pts = 8) {
        SliceMeasurement m;
        m.index = idx;
        m.left_boundary = Eigen::Vector2d(left_x0, left_z0);
        m.right_boundary = Eigen::Vector2d(right_x0, right_z0);
        m.width = width;
        set_measurement_height(m, height);
        // Generate surface points near the boundaries
        for (int k = 0; k < n_left_pts; ++k) {
            double x = left_x0 - 0.5 + k * 1.0 / n_left_pts;
            m.left_surface_pts.emplace_back(x, left_z0 + 0.001 * (k % 3 - 1));
        }
        for (int k = 0; k < n_right_pts; ++k) {
            double x = right_x0 - 0.5 + k * 1.0 / n_right_pts;
            m.right_surface_pts.emplace_back(x, right_z0 + 0.001 * (k % 3 - 1));
        }
        m.accepted = true;
        return m;
    };

    // ----- Test 1: All normal slices, all should be accepted -----
    {
        std::vector<SliceMeasurement> measurements;
        for (int i = 0; i < n_slices; ++i) {
            measurements.push_back(
                    make_slice(i, 10.0, 5.0, 15.0, 8.0, 5.0, 3.0));
        }
        auto filtered = filter_slices_by_3d_consistency(
                measurements, trans_mat,
                debug_dir.empty() ? "" : debug_dir + "/test1_all_normal");
        int accepted_count = 0;
        for (const auto& m : filtered)
            if (m.accepted) accepted_count++;
        if (accepted_count < n_slices) {
            std::cerr << "TEST1 FAIL: expected all " << n_slices
                      << " slices accepted, got " << accepted_count
                      << std::endl;
            return 1;
        }
    }

    // ----- Test 2: Some slices have right surface jumping to valley -----
    {
        std::vector<SliceMeasurement> measurements;
        for (int i = 0; i < n_slices; ++i) {
            auto m = make_slice(i, 10.0, 5.0, 15.0, 8.0, 5.0, 3.0);
            // Slice 5-7: right surface dropped to valley (much lower z)
            if (i >= 5 && i <= 7) {
                m.right_surface_pts.clear();
                for (int k = 0; k < 8; ++k) {
                    m.right_surface_pts.emplace_back(15.0 + k * 0.1,
                                                     -5.0 + 0.001 * k);
                }
                m.right_boundary = Eigen::Vector2d(15.0, -5.0);
            }
            measurements.push_back(m);
        }
        auto filtered = filter_slices_by_3d_consistency(
                measurements, trans_mat,
                debug_dir.empty() ? "" : debug_dir + "/test2_valley_jump");
        for (int i = 5; i <= 7; ++i) {
            if (filtered[i].accepted) {
                std::cerr << "TEST2 FAIL: slice " << i
                          << " with valley jump should be rejected"
                          << std::endl;
                return 1;
            }
        }
        if (!filtered[0].accepted || !filtered[10].accepted) {
            std::cerr << "TEST2 FAIL: normal slices should still be accepted"
                      << std::endl;
            return 1;
        }
    }

    // ----- Test 3: Width jump on some slices -----
    {
        std::vector<SliceMeasurement> measurements;
        for (int i = 0; i < n_slices; ++i) {
            double w = (i == 10) ? 20.0 : 5.0;  // sudden width jump at slice 10
            measurements.push_back(
                    make_slice(i, 10.0, 5.0, 10.0 + w, 8.0, w, 3.0));
        }
        auto filtered = filter_slices_by_3d_consistency(
                measurements, trans_mat,
                debug_dir.empty() ? "" : debug_dir + "/test3_width_jump");
        if (filtered[10].accepted) {
            std::cerr << "TEST3 FAIL: slice with width jump should be rejected"
                      << std::endl;
            return 1;
        }
        if (!filtered[0].accepted || !filtered[15].accepted) {
            std::cerr << "TEST3 FAIL: normal slices should be accepted"
                      << std::endl;
            return 1;
        }
    }

    // ----- Test 4: Equal height, gap preserved (gap shouldn't cause rejection)
    // -----
    {
        std::vector<SliceMeasurement> measurements;
        for (int i = 0; i < n_slices; ++i) {
            // Left and right at same height but with a gap
            measurements.push_back(
                    make_slice(i, 10.0, 5.0, 20.0, 5.0, 10.0, 0.0));
        }
        auto filtered = filter_slices_by_3d_consistency(
                measurements, trans_mat,
                debug_dir.empty() ? "" : debug_dir + "/test4_equal_height_gap");
        int accepted_count = 0;
        for (const auto& m : filtered)
            if (m.accepted) accepted_count++;
        if (accepted_count < n_slices) {
            std::cerr << "TEST4 FAIL: equal-height slices with gap should all "
                         "be accepted, got "
                      << accepted_count << "/" << n_slices << std::endl;
            return 1;
        }
    }

    // ----- Test 5: Right surface missing points on a few slices -----
    {
        std::vector<SliceMeasurement> measurements;
        for (int i = 0; i < n_slices; ++i) {
            auto m = make_slice(i, 10.0, 5.0, 15.0, 8.0, 5.0, 3.0);
            if (i == 12 || i == 13) {
                // Very few right surface points
                m.right_surface_pts.clear();
                m.right_surface_pts.emplace_back(15.0, 8.0);
            }
            measurements.push_back(m);
        }
        auto filtered = filter_slices_by_3d_consistency(
                measurements, trans_mat,
                debug_dir.empty() ? "" : debug_dir + "/test5_short_right");
        if (filtered[12].accepted || filtered[13].accepted) {
            std::cerr << "TEST5 FAIL: slices with short right support should "
                         "be rejected"
                      << std::endl;
            return 1;
        }
        if (!filtered[0].accepted || !filtered[10].accepted) {
            std::cerr << "TEST5 FAIL: normal slices should be accepted"
                      << std::endl;
            return 1;
        }
    }

    // ----- Test 6: Height should come from fitted surfaces, not noisy boundary
    // -----
    {
        std::vector<SliceMeasurement> measurements;
        for (int i = 0; i < n_slices; ++i) {
            auto m = make_slice(i, 10.0, 5.0, 15.0, 8.0, 5.0, 3.0);
            if (i == 8) {
                m.right_boundary.y() = 80.0;
                set_measurement_height(
                        m, m.right_boundary.y() - m.left_boundary.y());
            }
            measurements.push_back(m);
        }
        auto filtered = filter_slices_by_3d_consistency(
                measurements, trans_mat,
                debug_dir.empty() ? "" : debug_dir + "/test6_plane_height");
        if (!filtered[8].accepted) {
            std::cerr
                    << "TEST6 FAIL: noisy boundary slice should be corrected, "
                       "not rejected"
                    << std::endl;
            return 1;
        }
        if (std::abs(filtered[8].height - 3.0) > 0.05) {
            std::cerr << "TEST6 FAIL: expected fitted-surface height near 3.0, "
                         "got "
                      << filtered[8].height << std::endl;
            return 1;
        }
        if (!debug_dir.empty()) {
            const std::string plane_debug_dir =
                    debug_dir + "/test6_plane_height/";
            std::ifstream ifs(plane_debug_dir +
                              "final_measurement_summary.png");
            if (!ifs.good()) {
                std::cerr << "TEST6 FAIL: final measurement summary image was "
                             "not written"
                          << std::endl;
                return 1;
            }
            if (utility::filesystem::FileExists(plane_debug_dir +
                                                "slice_metrics_3d.csv") ||
                utility::filesystem::FileExists(plane_debug_dir +
                                                "left_3d_plane.tiff") ||
                utility::filesystem::FileExists(plane_debug_dir +
                                                "right_3d_plane.tiff") ||
                utility::filesystem::FileExists(plane_debug_dir +
                                                "left_3d_plane.ply") ||
                utility::filesystem::FileExists(plane_debug_dir +
                                                "right_3d_plane.ply") ||
                utility::filesystem::FileExists(plane_debug_dir +
                                                "fitted_3d_planes.ply") ||
                utility::filesystem::FileExists(plane_debug_dir +
                                                "left_residual_vs_slice.png") ||
                utility::filesystem::FileExists(
                        plane_debug_dir + "right_residual_vs_slice.png")) {
                std::cerr << "TEST6 FAIL: 3D debug artifacts should not be "
                             "written"
                          << std::endl;
                return 1;
            }
        }
    }

    // ----- Test 7: Debug surface points should match robust line-fit inliers
    // -----
    {
        std::vector<std::vector<Eigen::Vector2d>> groups(2);
        for (int x = 0; x <= 30; ++x) {
            groups[0].emplace_back(static_cast<double>(x), 0.02 * x);
        }
        groups[0].emplace_back(70.0, 80.0);
        for (int x = 90; x <= 120; ++x) {
            groups[1].emplace_back(static_cast<double>(x), 10.0 - 0.01 * x);
        }
        groups[1].emplace_back(50.0, -75.0);

        auto lines = line_segment(groups);
        if (lines.size() != 2 || groups[0].empty() || groups[1].empty()) {
            std::cerr << "TEST7 FAIL: expected two fitted lines" << std::endl;
            return 1;
        }
        auto left_max_it = std::max_element(
                groups[0].begin(), groups[0].end(),
                [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                    return a.x() < b.x();
                });
        auto right_min_it = std::min_element(
                groups[1].begin(), groups[1].end(),
                [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
                    return a.x() < b.x();
                });
        if (left_max_it->x() != lines[0].second.x() ||
            right_min_it->x() != lines[1].first.x()) {
            std::cerr << "TEST7 FAIL: debug points and measurement endpoints "
                         "should come from the same robust inlier set"
                      << std::endl;
            return 1;
        }
        if (left_max_it->x() >= 70.0 || right_min_it->x() <= 50.0) {
            std::cerr << "TEST7 FAIL: robust outliers should be removed from "
                         "debug/reference point groups"
                      << std::endl;
            return 1;
        }
    }

    // ----- Test 8: No 3D debug artifacts should be written -----
    if (!debug_dir.empty()) {
        std::vector<SliceMeasurement> measurements;
        for (int i = 0; i < n_slices; ++i) {
            SliceMeasurement m;
            m.index = i;
            m.left_boundary = Eigen::Vector2d(12.0, 5.0);
            m.right_boundary = Eigen::Vector2d(32.0, 8.0);
            m.width = 20.0;
            set_measurement_height(m, 3.0);
            for (int x = 10; x <= 14; ++x) {
                m.left_surface_pts.emplace_back(static_cast<double>(x), 5.0);
            }
            for (int x = 30; x <= 34; ++x) {
                m.right_surface_pts.emplace_back(static_cast<double>(x), 8.0);
            }
            m.accepted = true;
            measurements.push_back(m);
        }

        const std::string debug_path = debug_dir + "/test8_no_3d_debug/";
        auto filtered = filter_slices_by_3d_consistency(measurements, trans_mat,
                                                        debug_path);
        int accepted_count = 0;
        for (const auto& m : filtered)
            if (m.accepted) accepted_count++;
        if (accepted_count != n_slices) {
            std::cerr << "TEST8 FAIL: synthetic slices should remain accepted"
                      << std::endl;
            return 1;
        }
        if (utility::filesystem::FileExists(debug_path +
                                            "slice_metrics_3d.csv") ||
            utility::filesystem::FileExists(debug_path +
                                            "left_3d_plane.tiff") ||
            utility::filesystem::FileExists(debug_path +
                                            "right_3d_plane.tiff") ||
            utility::filesystem::FileExists(debug_path + "left_3d_plane.ply") ||
            utility::filesystem::FileExists(debug_path +
                                            "right_3d_plane.ply") ||
            utility::filesystem::FileExists(debug_path +
                                            "fitted_3d_planes.ply") ||
            utility::filesystem::FileExists(debug_path +
                                            "left_residual_vs_slice.png") ||
            utility::filesystem::FileExists(debug_path +
                                            "right_residual_vs_slice.png")) {
            std::cerr << "TEST8 FAIL: 3D debug artifacts should not be written"
                      << std::endl;
            return 1;
        }
    }

    // ----- Test 9: Final reported height is absolute, signed height is kept
    // for direction analysis -----
    {
        std::vector<SliceMeasurement> measurements;
        for (int i = 0; i < n_slices; ++i) {
            auto m = make_slice(i, 10.0, 8.0, 15.0, 5.0, 5.0, -3.0);
            measurements.push_back(m);
        }
        double gap_step = 0.0;
        double step_width = 0.0;
        std::vector<std::vector<double>> temp_res(2);
        fill_result_from_measurements(measurements, gap_step, step_width,
                                      &temp_res);
        if (std::abs(gap_step - 3.0) > 1e-6) {
            std::cerr << "TEST9 FAIL: final gap_step should use absolute "
                         "height, got "
                      << gap_step << std::endl;
            return 1;
        }
        if (temp_res.size() < 2 || temp_res[1].empty() ||
            std::abs(temp_res[1].front() - 3.0) > 1e-6) {
            std::cerr << "TEST9 FAIL: exported height series should use "
                         "absolute height"
                      << std::endl;
            return 1;
        }
        if (std::abs(measurements.front().height - 3.0) > 1e-6 ||
            std::abs(measurements.front().signed_height - (-3.0)) > 1e-6) {
            std::cerr << "TEST9 FAIL: measurement should keep both abs and "
                         "signed height"
                      << std::endl;
            return 1;
        }
    }

    // ----- Test 10: Smooth diagonal boundary motion should not be treated as
    // local boundary jumps -----
    {
        std::vector<SliceMeasurement> measurements;
        for (int i = 0; i < n_slices; ++i) {
            const double left_x = 10.0 + 5.0 * i;
            const double right_x = left_x + 8.0;
            measurements.push_back(
                    make_slice(i, left_x, 5.0, right_x, 8.0, 8.0, 3.0));
        }
        auto filtered = filter_slices_by_3d_consistency(
                measurements, trans_mat,
                debug_dir.empty() ? "" : debug_dir + "/test10_diagonal_edge");
        int accepted_count = 0;
        for (const auto& m : filtered)
            if (m.accepted) accepted_count++;
        if (accepted_count < n_slices - 1) {
            std::cerr
                    << "TEST10 FAIL: smooth diagonal boundaries should remain "
                       "accepted, got "
                    << accepted_count << "/" << n_slices << std::endl;
            return 1;
        }
    }

    return 0;
}
#endif

}  // namespace pipeline
}  // namespace hymson3d

#include "GapStepDetection.h"

#include <math.h>

#include <cmath>
#include <fstream>
#include <opencv2/opencv.hpp>

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
constexpr int kMaxSurfaceGap = 2;
constexpr size_t kMinSurfacePoints = 4;

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
    double height = 0.0;
    Eigen::Vector2d left_boundary =
            Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::quiet_NaN());
    Eigen::Vector2d right_boundary =
            Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::quiet_NaN());
    bool accepted = false;
    std::string reject_reason;
};

using SliceLineSegments =
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>;

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
    if (pts.size() < 8) return fit_line_segment(pts);

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
    return fit_line_segment(inliers);
}

double line_slope(const std::pair<Eigen::Vector2d, Eigen::Vector2d>& line) {
    const double dx = line.second.x() - line.first.x();
    if (std::abs(dx) < 1e-12) return 0.0;
    return (line.second.y() - line.first.y()) / dx;
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
    std::vector<Eigen::Vector2d> expanded;
    expanded.reserve(region.size());

    for (const auto& pt : region) {
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
        bool use_lht_width) {
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
        measurement.height = corners[i].second.y() - corners[i].first.y();
        if (measurement.width == -255.0 || measurement.width <= 0.0 ||
            !std::isfinite(measurement.width) ||
            !std::isfinite(measurement.height)) {
            measurement.reject_reason = "invalid_measurement";
            measurements.push_back(measurement);
            continue;
        }

        measurement.accepted = true;
        measurements.push_back(measurement);
    }

    std::vector<double> widths;
    std::vector<double> heights;
    for (const auto& measurement : measurements) {
        if (!measurement.accepted) continue;
        widths.push_back(measurement.width);
        heights.push_back(measurement.height);
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
                std::abs(measurement.height - height_median) > height_limit;
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
        heights.push_back(measurement.height);
        if (temp_res != nullptr) {
            (*temp_res)[0].emplace_back(measurement.width);
            (*temp_res)[1].emplace_back(measurement.height);
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
    if (debug_path.empty()) return;
    std::string path = debug_path;
    if (!path.empty() && path.back() != '/' && path.back() != '\\') path += "/";
    path += "slice_metrics.csv";

    std::ofstream ofs(path);
    if (!ofs.is_open()) return;
    ofs << "slice,width,height,left_x,left_y,right_x,right_y,accepted,"
           "reject_reason\n";
    for (const auto& measurement : measurements) {
        ofs << measurement.index << "," << measurement.width << ","
            << measurement.height << "," << measurement.left_boundary.x() << ","
            << measurement.left_boundary.y() << ","
            << measurement.right_boundary.x() << ","
            << measurement.right_boundary.y() << ","
            << (measurement.accepted ? 1 : 0) << ","
            << measurement.reject_reason << "\n";
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
            std::max(kMinSurfaceSlopeLimit, slope_limit * 1.5);

    auto flush_segment = [&]() {
        if (segment.size() >= kMinSurfacePoints) {
            auto candidate = make_surface_candidate(segment);
            const double slope = std::abs(line_slope(candidate.line));
            const double rms_limit =
                    std::max(0.2, candidate.span * slope_limit * 0.05);
            if (candidate.span > 0.0 && slope <= slope_limit &&
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
    auto right_candidates = collect_surface_candidates(right_candidates_region);
    if (right_candidates.empty()) {
        return {left_region, right_candidates_region};
    }

    auto right_it = std::max_element(
            right_candidates.begin(), right_candidates.end(),
            [&right_search](const SurfaceCandidate& a,
                            const SurfaceCandidate& b) {
                auto score = [&right_search](
                                     const SurfaceCandidate& candidate) {
                    const double edge_distance = std::max(
                            0.0, candidate.x_min - right_search.edge_x);
                    const double valley_distance = std::abs(
                            candidate.x_center - right_search.valley_x);
                    const double valley_penalty =
                            valley_distance < std::max(candidate.span, 1.0)
                                    ? 1000.0
                                    : 0.0;
                    return candidate.span - 5.0 * candidate.rms -
                           3.0 * std::abs(line_slope(candidate.line)) -
                           0.02 * edge_distance - valley_penalty;
                };
                return score(a) < score(b);
            });
    if (right_it == right_candidates.end())
        return {left_region, right_candidates_region};

    return {left_region, expand_surface_around_candidate(
                                 right_candidates_region, *right_it)};
}

std::vector<Eigen::Vector2d> filter_surface_group(
        const std::vector<Eigen::Vector2d>& points,
        bool prefer_stable_subsegment) {
    if (points.size() < 8) return points;

    if (prefer_stable_subsegment) {
        auto candidates = collect_surface_candidates(points);
        if (!candidates.empty()) {
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
        }
    }

    const auto line = fit_line_segment_robust(points);
    double sum_sq_residual = 0.0;
    for (const auto& pt : points) {
        const double residual = pt.y() - line_y_at_x(line, pt.x());
        sum_sq_residual += residual * residual;
    }
    const double rms =
            std::sqrt(sum_sq_residual / static_cast<double>(points.size()));
    const double threshold = std::max(3.0 * rms, 0.03);

    std::vector<Eigen::Vector2d> filtered;
    filtered.reserve(points.size());
    for (const auto& pt : points) {
        const double residual = std::abs(pt.y() - line_y_at_x(line, pt.x()));
        if (residual <= threshold) filtered.push_back(pt);
    }
    return filtered.size() >= kMinSurfacePoints ? filtered : points;
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
    }
    // std::cout << "1" << std::endl;

    // LOG_DEBUG("Slice along Y-axis");
    //  slice along y axis
    slice_along_y(cloud, transformation_matrix);
    // std::cout << "2" << std::endl;

    // bspline interpolation
    // double height_threshold = 0.01;
    lineSegments corners;
    // bspline_interpolation(cloud, height_threshold, corners, debug_mode);
    // std::cout << "3" << std::endl;
    bspline_interpolation_dll(cloud, height_threshold, corners, debug_path,
                              debug_mode);
    // std::cout << "3" << std::endl;

    // calculate the gap step result
    // double gap_step = 0.0, step_width = 0.0;
    std::vector<double> LHT_width;
    bool LHT = false;
    calculate_gap_step_dll_plot(corners, LHT_width, gap_step, step_width,
                                temp_res, LHT);
    if (debug_mode) {
        write_slice_measurements_csv(
                debug_path,
                collect_slice_measurements(corners, LHT_width, LHT));
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
    // debug mode
    if (debug_mode) {
        utility::filesystem::MakeDirectory_dll(debug_path);
    }

    // slice along y axis
    slice_along_y(cloud, transformation_matrix);

    // bspline interpolation
    std::vector<double> LHT_width;
    lineSegments corners;
    // bspline_interpolation(cloud, height_threshold, corners, debug_mode);
    // std::cout << "3" << std::endl;
    bspline_interpolation_dll2(cloud, height_threshold, corners, LHT_width,
                               debug_path, LHT, debug_mode);
    // std::cout << "3" << std::endl;

    // calculate the gap step result
    calculate_gap_step_dll_plot(corners, LHT_width, gap_step, step_width,
                                temp_res, LHT);
    if (debug_mode) {
        write_slice_measurements_csv(
                debug_path,
                collect_slice_measurements(corners, LHT_width, LHT));
    }
    // std::cout << "4" << std::endl;
}

void GapStepDetection::slice_along_y(geometry::PointCloud::Ptr cloud,
                                     Eigen::Vector3d transformation_matrix) {
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
        bool debug_mode) {
    // use common part to fit a curve
    core::PlaneDetection plane_detector;
    std::vector<double> step_height;
    step_height.resize(cloud->y_slices_.size());
    corners.resize(cloud->y_slices_.size());
    LHT_width.resize(cloud->y_slices_.size());
    std::fill(corners.begin(), corners.end(), invalid_corner());
    std::fill(step_height.begin(), step_height.end(), -255.0);
    std::fill(LHT_width.begin(), LHT_width.end(), -255.0);

#pragma omp parallel for
    for (int i = 0; i < cloud->y_slices_.size(); i++) {
        // Cubic B-spline needs at least degree+1 = 4 control points
        if (cloud->y_slices_[i].size() < 4) continue;
        int sampled_pts = adaptive_sample_count(cloud->y_slices_[i].size());
        std::vector<Eigen::Vector2d> resampled_pts =
                plane_detector.resample_a_curve(cloud->y_slices_[i],
                                                sampled_pts, i, false);
        // compute the derivative
        std::vector<Eigen::Vector2d> limit_pts;
        std::vector<std::vector<Eigen::Vector2d>> groups =
                group_by_derivative_dll(resampled_pts);

        // Guard: mark this slice as invalid if either group is empty (cannot
        // cluster)
        if (groups.size() < 2 || groups[0].empty() || groups[1].empty()) {
            LHT_width[i] = std::numeric_limits<double>::quiet_NaN();
            corners[i] = std::make_pair(
                    Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN()),
                    Eigen::Vector2d(std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN()));
            continue;
        }

        std::vector<std::vector<Eigen::Vector2d>> filter_groups =
                statistics_filter(groups, limit_pts);

        double left_height_threshold = height_threshold,
               right_height_threshold = height_threshold;
        lineSegments lines = line_segment(filter_groups);
        if (lines.size() < 2) {
            step_height[i] = -255.0;
            LHT_width[i] = -255.0;
            if (debug_mode) {
                plot_clusters_dll(resampled_pts, filter_groups, lines, {},
                                  limit_pts, debug_path, i);
            }
            continue;
        }
        step_height[i] = std::abs(lines[0].first.y() - lines[1].first.y());
        std::vector<std::vector<Eigen::Vector2d>> intersections;
        std::vector<double> temp_width;
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

        if (debug_mode) {
            if (LHT) {
                plot_clusters_dll_lht(cloud->y_slices_[i], resampled_pts,
                                      filter_groups, lines, height_threshold,
                                      intersections, limit_pts, debug_path, i);
            } else {
                plot_clusters_dll(resampled_pts, filter_groups, lines,
                                  intersections, limit_pts, debug_path, i);
            }
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
        res.push_back(fit_line_segment_robust(pt_groups[i]));
    }
    if (res.size() >= 2 && res[0].second.x() > res[1].first.x())
        std::swap(res[1], res[0]);
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
        res.push_back(fit_line_segment_robust(filter_pt_groups[i]));
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
#endif

}  // namespace pipeline
}  // namespace hymson3d

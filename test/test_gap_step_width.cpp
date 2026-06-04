#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "GapStepDetection.h"

using hymson3d::pipeline::GapStepDetection;

namespace {

using Line = std::pair<Eigen::Vector2d, Eigen::Vector2d>;

Line horizontal_line(double x0, double x1, double y) {
    return {Eigen::Vector2d(x0, y), Eigen::Vector2d(x1, y)};
}

Line sloped_line(double x0, double x1, double intercept, double slope) {
    return {Eigen::Vector2d(x0, intercept + slope * x0),
            Eigen::Vector2d(x1, intercept + slope * x1)};
}

bool expect_near(const std::string& name,
                 double actual,
                 double expected,
                 double tolerance = 1e-6) {
    if (std::abs(actual - expected) <= tolerance) return true;
    std::cerr << name << ": expected " << expected << ", got " << actual
              << std::endl;
    return false;
}

bool expect_true(const std::string& name, bool value) {
    if (value) return true;
    std::cerr << name << ": expected true" << std::endl;
    return false;
}

bool expect_false(const std::string& name, bool value) {
    if (!value) return true;
    std::cerr << name << ": expected false" << std::endl;
    return false;
}

std::vector<Eigen::Vector2d> make_profile(double left_height,
                                          double right_height) {
    std::vector<Eigen::Vector2d> profile;
    for (int x = 0; x <= 30; ++x) profile.emplace_back(x, left_height);
    for (int x = 31; x <= 40; ++x) {
        const double t = static_cast<double>(x - 30) / 10.0;
        profile.emplace_back(x, left_height * (1.0 - t) + 5.0 * t);
    }
    for (int x = 41; x <= 59; ++x) profile.emplace_back(x, 5.0);
    for (int x = 60; x <= 70; ++x) {
        const double t = static_cast<double>(x - 60) / 10.0;
        profile.emplace_back(x, 5.0 * (1.0 - t) + right_height * t);
    }
    for (int x = 71; x <= 100; ++x) profile.emplace_back(x, right_height);
    return profile;
}

bool test_equal_height_u_gap() {
    auto profile = make_profile(10.0, 10.0);
    const auto result = GapStepDetection::test_compute_threshold_width(
            profile, horizontal_line(0, 30, 10.0),
            horizontal_line(70, 100, 10.0), 2.0, 30.0, 70.0);
    return expect_true("equal-height gap valid", result.valid) &&
           expect_near("equal-height width", result.width, 32.0) &&
           expect_near("equal-height left intersection",
                       result.left_intersection.x(), 34.0) &&
           expect_near("equal-height right intersection",
                       result.right_intersection.x(), 66.0);
}

bool test_left_low_and_right_low() {
    auto left_low_profile = make_profile(10.0, 15.0);
    const auto left_low = GapStepDetection::test_compute_threshold_width(
            left_low_profile, horizontal_line(0, 30, 10.0),
            horizontal_line(70, 100, 15.0), 2.0, 30.0, 70.0);
    if (!expect_true("left-low gap valid", left_low.valid) ||
        !expect_near("left-low width", left_low.width, 29.0) ||
        !expect_near("left-low shifted line y", left_low.shifted_line.first.y(),
                     8.0)) {
        return false;
    }

    auto right_low_profile = make_profile(15.0, 10.0);
    const auto right_low = GapStepDetection::test_compute_threshold_width(
            right_low_profile, horizontal_line(0, 30, 15.0),
            horizontal_line(70, 100, 10.0), 2.0, 30.0, 70.0);
    return expect_true("right-low gap valid", right_low.valid) &&
           expect_near("right-low width", right_low.width, 29.0) &&
           expect_near("right-low shifted line y",
                       right_low.shifted_line.first.y(), 8.0);
}

bool test_sloped_reference_uses_horizontal_projection() {
    const Line low_line = sloped_line(0.0, 30.0, 10.0, 0.1);
    const Line high_line = sloped_line(70.0, 100.0, 15.0, 0.1);
    std::vector<Eigen::Vector2d> profile;
    for (int x = 0; x <= 100; ++x) {
        double y = 0.0;
        if (x <= 30)
            y = 10.0 + 0.1 * x;
        else if (x <= 40)
            y = 13.0 - 0.8 * (x - 30);
        else if (x < 60)
            y = 5.0;
        else if (x < 70)
            y = 5.0 + 1.7 * (x - 60);
        else
            y = 15.0 + 0.1 * x;
        profile.emplace_back(x, y);
    }

    const auto result = GapStepDetection::test_compute_threshold_width(
            profile, low_line, high_line, 2.0, 30.0, 70.0);
    const double shifted_slope =
            (result.shifted_line.second.y() - result.shifted_line.first.y()) /
            (result.shifted_line.second.x() - result.shifted_line.first.x());
    return expect_true("sloped gap valid", result.valid) &&
           expect_near("sloped horizontal width", result.width,
                       31.0294117647, 1e-6) &&
           expect_near("shifted line is horizontal", shifted_slope, 0.0) &&
           expect_near("shifted line uses near-gap endpoint",
                       result.shifted_line.first.y(), low_line.second.y() - 2.0);
}

bool test_lower_reference_is_selected_by_near_gap_endpoints() {
    const Line left_line = sloped_line(0.0, 30.0, -5.0, 0.5);
    const Line right_line = sloped_line(70.0, 100.0, -24.0, 0.5);
    std::vector<Eigen::Vector2d> profile;
    for (int x = 0; x <= 30; ++x)
        profile.emplace_back(x, -5.0 + 0.5 * x);
    for (int x = 31; x <= 40; ++x)
        profile.emplace_back(x, 10.0 - 0.5 * (x - 30));
    for (int x = 41; x <= 59; ++x) profile.emplace_back(x, 5.0);
    for (int x = 60; x <= 70; ++x)
        profile.emplace_back(x, 5.0 + 0.6 * (x - 60));
    for (int x = 71; x <= 100; ++x)
        profile.emplace_back(x, -24.0 + 0.5 * x);

    const auto result = GapStepDetection::test_compute_threshold_width(
            profile, left_line, right_line, 2.0, 30.0, 70.0);
    return expect_true("endpoint-selected gap valid", result.valid) &&
           expect_near("left endpoint is lower", result.lower_line_index, 0) &&
           expect_near("endpoint-selected shifted line y",
                       result.shifted_line.first.y(), 8.0) &&
           expect_near("endpoint-selected shifted line is horizontal",
                       result.shifted_line.second.y(), 8.0);
}

bool test_right_low_sloped_reference_uses_near_gap_endpoint() {
    const Line left_line = sloped_line(0.0, 30.0, 15.0, -0.1);
    const Line right_line = sloped_line(70.0, 100.0, 17.0, -0.1);
    std::vector<Eigen::Vector2d> profile;
    for (int x = 0; x <= 100; ++x) {
        double y = 0.0;
        if (x <= 30)
            y = 15.0 - 0.1 * x;
        else if (x <= 40)
            y = 12.0 - 0.7 * (x - 30);
        else if (x < 60)
            y = 5.0;
        else if (x < 70)
            y = 5.0 + 0.5 * (x - 60);
        else
            y = 17.0 - 0.1 * x;
        profile.emplace_back(x, y);
    }

    const auto result = GapStepDetection::test_compute_threshold_width(
            profile, left_line, right_line, 2.0, 30.0, 70.0);
    return expect_true("right-low sloped gap valid", result.valid) &&
           expect_near("right-low line selected", result.lower_line_index, 1) &&
           expect_near("right-low endpoint sets shifted height",
                       result.shifted_line.first.y(), 8.0) &&
           expect_near("right-low horizontal width", result.width,
                       30.2857142857, 1e-6);
}

bool test_missing_or_insufficient_intersections_return_zero() {
    std::vector<Eigen::Vector2d> no_intersection;
    for (int x = 0; x <= 100; ++x) no_intersection.emplace_back(x, 10.0);
    auto none = GapStepDetection::test_compute_threshold_width(
            no_intersection, horizontal_line(0, 30, 10.0),
            horizontal_line(70, 100, 10.0), 2.0, 30.0, 70.0);
    if (!expect_false("no-intersection invalid", none.valid) ||
        !expect_near("no-intersection width", none.width, 0.0)) {
        return false;
    }

    std::vector<Eigen::Vector2d> one_intersection;
    for (int x = 0; x <= 100; ++x)
        one_intersection.emplace_back(x, 10.0 - 0.1 * x);
    auto one = GapStepDetection::test_compute_threshold_width(
            one_intersection, horizontal_line(0, 30, 10.0),
            horizontal_line(70, 100, 10.0), 2.0, 30.0, 70.0);
    return expect_false("one-intersection invalid", one.valid) &&
           expect_near("one-intersection width", one.width, 0.0);
}

bool test_multiple_intersections_choose_outer_pair_around_center() {
    std::vector<Eigen::Vector2d> profile;
    for (int x = 0; x <= 100; ++x) {
        double y = 10.0;
        if (x >= 32 && x <= 38) y = 7.0;
        if (x >= 42 && x <= 68) y = 6.0;
        profile.emplace_back(x, y);
    }
    const auto result = GapStepDetection::test_compute_threshold_width(
            profile, horizontal_line(0, 30, 10.0),
            horizontal_line(70, 100, 10.0), 2.0, 30.0, 70.0);
    return expect_true("multiple-intersection gap valid", result.valid) &&
           expect_near("multiple-intersection outer width", result.width,
                       36.8333333333, 1e-5);
}

bool test_missing_data_gap_does_not_create_intersections() {
    std::vector<Eigen::Vector2d> profile;
    for (int x = 0; x <= 30; ++x) profile.emplace_back(x, 10.0);
    for (int x = 70; x <= 100; ++x) profile.emplace_back(x, 10.0);
    const auto result = GapStepDetection::test_compute_threshold_width(
            profile, horizontal_line(0, 30, 10.0),
            horizontal_line(70, 100, 10.0), 2.0, 30.0, 70.0);
    return expect_false("missing-data gap invalid", result.valid) &&
           expect_near("missing-data gap width", result.width, 0.0);
}

bool test_non_positive_threshold_returns_zero() {
    auto profile = make_profile(10.0, 10.0);
    const auto result = GapStepDetection::test_compute_threshold_width(
            profile, horizontal_line(0, 30, 10.0),
            horizontal_line(70, 100, 10.0), 0.0, 30.0, 70.0);
    return expect_false("zero threshold invalid", result.valid) &&
           expect_near("zero threshold width", result.width, 0.0);
}

}  // namespace

int main() {
    if (!test_equal_height_u_gap()) return 1;
    if (!test_left_low_and_right_low()) return 1;
    if (!test_sloped_reference_uses_horizontal_projection()) return 1;
    if (!test_lower_reference_is_selected_by_near_gap_endpoints()) return 1;
    if (!test_right_low_sloped_reference_uses_near_gap_endpoint()) return 1;
    if (!test_missing_or_insufficient_intersections_return_zero()) return 1;
    if (!test_multiple_intersections_choose_outer_pair_around_center())
        return 1;
    if (!test_missing_data_gap_does_not_create_intersections()) return 1;
    if (!test_non_positive_threshold_returns_zero()) return 1;
    if (GapStepDetection::test_width_height_independence() != 0) return 1;
    return 0;
}

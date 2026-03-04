#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "2D/Geometry2D.h"
#include "Eigen.h"
#include "Logger.h"

namespace hymson3d {
namespace geometry {

class Surface : public Geometry2D {
public:
    Surface() : Geometry2D(Geometry::GeometryType::Surface) {}
    ~Surface() override {}

public:
    Geometry& Clear() override {
        surface_ = cv::Mat();
        coeffs_ = cv::Mat();
        return *this;
    }

    bool IsEmpty() const override { return surface_.empty(); }

    Eigen::Vector2d GetMinBound() const override {
        if (surface_.empty()) {
            return Eigen::Vector2d(0, 0);
        }
        return Eigen::Vector2d(0, 0);
    }

    Eigen::Vector2d GetMaxBound() const override {
        if (surface_.empty()) {
            return Eigen::Vector2d(0, 0);
        }
        return Eigen::Vector2d(surface_.cols, surface_.rows);
    }

    // use mask and two-step fitting
    cv::Mat FitQuadraticSurface(
            const cv::Mat& heightMap,
            cv::Mat& mask, /*cv::Mat::ones(rows, cols, CV_8U)*/
            int iterations = 2,
            int step = 1,
            float outlierThreshold = 1.0f);

    // one-step fitting for fast process
    Eigen::Vector6d FitQuadraticSurface(const cv::Mat& heightMap,
                                        float thresholdVal,
                                        const int sample_step = 1);

    cv::Mat FitFreeFormSurface(const cv::Mat& heightMap, int kernel_size);

public:
    cv::Mat surface_;  // z = ax^2 + by^2 + cxy + dx + ey + f
    cv::Mat coeffs_;   //(a...e, f)
};
}  // namespace geometry
}  // namespace hymson3d
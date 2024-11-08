#pragma once

#include <Eigen/Core>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "2D/Geometry2D.h"
#include "Logger.h"

namespace hymson3d {
namespace geometry {

class Image : public Geometry2D {
public:
    Image() : Geometry2D(Geometry::GeometryType::Image) {}
    ~Image() override {}

public:
    Image &Clear() override;
    virtual bool HasData() const { return !data_.empty(); };
    bool IsEmpty() const override;
    Eigen::Vector2d GetMinBound() const override;
    Eigen::Vector2d GetMaxBound() const override;

    void set_data(const cv::Mat &data);

public:
    /// Width of the image.
    int width_ = 0;
    /// Height of the image.
    int height_ = 0;
    /// Number of channels in the image.
    int num_of_channels_ = 0;
    /// Number of bytes per channel.
    int bytes_per_channel_ = 0;
    /// Image storage buffer.
    cv::Mat data_;
    /**************/
    // CV_8U  0
    // CV_8S  1
    // CV_16U 2
    // CV_16S 3
    // CV_32S 4
    // CV_32F 5
    // CV_64F 6
    // CV_16F 7
    /**************/
    int data_type_;
};

}  // namespace geometry
}  // namespace hymson3d
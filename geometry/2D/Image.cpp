#include "Image.h"

namespace hymson3d {
namespace geometry {

Image &Image::Clear() {
    width_ = 0;
    height_ = 0;
    num_of_channels_ = 0;
    bytes_per_channel_ = 0;
    data_.release();
    return *this;
}

bool Image::IsEmpty() const { return data_.empty(); }

Eigen::Vector2d Image::GetMinBound() const { return Eigen::Vector2d(0.0, 0.0); }

Eigen::Vector2d Image::GetMaxBound() const {
    return Eigen::Vector2d(width_, height_);
}

void Image::set_data(const cv::Mat &data) {
    width_ = data.cols;
    height_ = data.rows;
    num_of_channels_ = data.channels();
    bytes_per_channel_ = data.elemSize();
    data_ = data;
    data_type_ = data.depth();
}

}  // namespace geometry
}  // namespace hymson3d
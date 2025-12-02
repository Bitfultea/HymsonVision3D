#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

#include "common.hpp"
#include "ml.h"
#include "yolo_3d_seg.hpp"

namespace hymson3d {
namespace ml {

class Segmentation_ML_3d {
public:
    // typedef std::shared_ptr<Segmentation_ML_3d> Ptr;
    Segmentation_ML_3d(const std::string& engine_file_path,
                       cv::Size size = cv::Size{960, 960},
                       int topk = 100,
                       int seg_h = 160,
                       int seg_w = 160,
                       int seg_channels = 32);
    ~Segmentation_ML_3d() {};

public:
    void infer(const cv::Mat tiff_image,
               float score_thres = 0.25f,
               float iou_thres = 0.65f,
               bool debug_mode = false);
    std::vector<Object> get_objects();

private:
    cv::Mat preprocess(const cv::Mat tiff_image);
    cv::Mat robust_normalize(const cv::Mat& src,
                             double lower_percent = 0.005,
                             double upper_percent = 0.995);
    cv::Mat preprocess_height_to_bgr(const cv::Mat tiff_image);

private:
    std::vector<Object> m_objs;
    cv::Mat raw_2d_grid_image_;
    std::unique_ptr<YOLO_3D_seg> yolo_3d_seg_;

    int m_topk;
    int m_seg_h;
    int m_seg_w;
    int m_seg_channels;
    cv::Size m_size;
};
}  // namespace ml
}  // namespace hymson3d
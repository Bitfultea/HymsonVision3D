#include "segment_3d.h"

#include "opencv2/opencv.hpp"

namespace fs = ghc::filesystem;

namespace hymson3d {
namespace ml {

Segmentation_ML_3d::Segmentation_ML_3d(const std::string& engine_file_path,
                                       cv::Size size,
                                       int topk,
                                       int seg_h,
                                       int seg_w,
                                       int seg_channels) {
    yolo_3d_seg_ = std::make_unique<YOLO_3D_seg>(engine_file_path);
    m_size = size;
    m_topk = topk;
    m_seg_h = seg_h;
    m_seg_w = seg_w;
    m_seg_channels = seg_channels;
}

void Segmentation_ML_3d::infer(const cv::Mat tiff_image,
                               float score_thres,
                               float iou_thres,
                               bool debug_mode) {
    raw_2d_grid_image_ = tiff_image;
    cv::Mat pre_processed_image = preprocess(tiff_image);

    yolo_3d_seg_->make_pipe(true);

    cv::Mat res;
    // cv::Size size = cv::Size{960, 960};
    // float score_thres = 0.25f;
    // float iou_thres = 0.65f;

    // std::vector<Object> objs;

    yolo_3d_seg_->copy_from_Mat(pre_processed_image, m_size);
    auto start = std::chrono::system_clock::now();
    yolo_3d_seg_->infer();
    yolo_3d_seg_->postprocess(m_objs, score_thres, iou_thres, m_topk,
                              m_seg_channels, m_seg_h, m_seg_w);
    yolo_3d_seg_->draw_objects(pre_processed_image, res, m_objs, CLASS_NAMES,
                               COLORS, MASK_COLORS);
    auto end = std::chrono::system_clock::now();
    auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(
                      end - start)
                      .count() /
              1000.;
    if (debug_mode) {
        cv::imwrite("res.png", res);
    }
}

std::vector<Object> Segmentation_ML_3d::get_objects() { return m_objs; }
cv::Mat Segmentation_ML_3d::robust_normalize(const cv::Mat& src,
                                             double lower_percent,
                                             double upper_percent) {
    cv::Mat flat;
    src.reshape(1, 1).convertTo(flat, CV_32F);
    std::vector<float> pixels = flat;

    std::sort(pixels.begin(), pixels.end());
    float lower = pixels[static_cast<int>(pixels.size() * lower_percent)];
    float upper = pixels[static_cast<int>(pixels.size() * upper_percent)];

    cv::Mat clipped;
    cv::threshold(src, clipped, upper, upper, cv::THRESH_TRUNC);
    cv::max(clipped, lower, clipped);

    cv::Mat norm_img;
    if (upper - lower > 1e-6) {
        clipped.convertTo(norm_img, CV_8U, 255.0 / (upper - lower),
                          -lower * 255.0 / (upper - lower));
    } else {
        norm_img = cv::Mat::zeros(src.size(), CV_8U);
    }
    return norm_img;
}

cv::Mat Segmentation_ML_3d::preprocess_height_to_bgr(const cv::Mat tiff_image) {
    cv::Mat raw_f;
    tiff_image.convertTo(raw_f, CV_32F);

    // Ch1: Height
    cv::Mat ch_height = robust_normalize(raw_f);

    // Ch2: Gradient
    cv::Mat gx, gy, grad;
    cv::Sobel(raw_f, gx, CV_32F, 1, 0, 3);
    cv::Sobel(raw_f, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, grad);
    cv::Mat ch_grad = robust_normalize(grad, 0.0, 0.98);

    // Ch3: Texture (CLAHE)
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat ch_texture;
    clahe->apply(ch_height, ch_texture);

    // Merge
    cv::Mat merged;
    std::vector<cv::Mat> channels = {ch_height, ch_grad, ch_texture};
    cv::merge(channels, merged);  // BGR for OpenCV

    return merged;
}

cv::Mat Segmentation_ML_3d::preprocess(const cv::Mat tiff_image) {
    // TODO: add normals as extra feature channels

    return preprocess_height_to_bgr(tiff_image);
}

}  // namespace ml
}  // namespace hymson3d
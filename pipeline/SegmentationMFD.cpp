#include "SegmentationMFD.h"

namespace hymson3d {
namespace pipeline {

SegmentationMFD::SegmentationMFD() { model_ = nullptr; }

SegmentationMFD::SegmentationMFD(const std::string& model_path,
                                 cv::Size size,
                                 int topk,
                                 int seg_h,
                                 int seg_w,
                                 int seg_channelss) {
    model_ = std::make_unique<ml::Segmentation_ML_3d>(
            model_path, size, topk, seg_h, seg_w, seg_channelss);
}
bool SegmentationMFD::enableModel(bool enable,
                                  const std::string& model_path,
                                  cv::Size size,
                                  int topk,
                                  int seg_h,
                                  int seg_w,
                                  int seg_channelss) { 
    if (enable) {
        // 已经初始化，直接返回
        if (model_) {
            return true;
        }
        try {
            model_ = std::make_unique<ml::Segmentation_ML_3d>(
                    model_path, size, topk, seg_h, seg_w, seg_channelss);
        } catch (...) {
            model_.reset();
            return false;
        }
        return true;
    } else {
        if (model_) {
            model_.reset();
        }
        return true;
    }
}

void SegmentationMFD::run_segmentation(const std::string& tiff_file_path,
                                       float score_thres,
                                       float iou_thres,
                                       bool debug_mode) {
    cv::Mat tiff_image;
    if (!utility::read_tiff(tiff_file_path, tiff_image)) {
        LOG_ERROR("Read tiff file failed!");
        return;
    }
    m_raw_tiff = tiff_image;
    model_->infer(tiff_image, score_thres, iou_thres, debug_mode);
}

bool SegmentationMFD::run_segmentation_dll(cv::Mat& tiff_image,
                                       float score_thres,
                                       float iou_thres,
                                       bool debug_mode) {
    m_raw_tiff = tiff_image;
    if (model_ == nullptr) {
        return false;
    }
    model_->infer(tiff_image, score_thres, iou_thres, debug_mode);
    return true;
}

void SegmentationMFD::analysis_defetcts(std::vector<Defect>& defects) {
    std::vector<Object> objects = model_->get_objects();
    LOG_INFO("Detect {} objects", objects.size());

    defects.clear();
    for (auto& obj : objects) {
        Defect defect;
        // area
        defect.area_bbox = obj.rect.area();
        defect.area_mask_pixel = cv::countNonZero(obj.boxMask);
        // position
        defect.center = (obj.rect.tl() + obj.rect.br()) / 2;
        defect.half_h = obj.rect.height / 2.0f;
        defect.half_w = obj.rect.width / 2.0f;
        // label
        defect.label_id = obj.label;
        std::string label = ml::CLASS_NAMES[obj.label];
        defect.label = label;
        // masked points
        defect.height = get_masked_defect_height(obj.rect);
        //score
        defect.score = obj.prob;
        defects.emplace_back(defect);
    }
}

float SegmentationMFD::get_masked_defect_height(cv::Rect mask) {
    cv::Mat masked_tiff = m_raw_tiff(mask);
    double min_val, max_val;
    cv::minMaxLoc(masked_tiff, &min_val, &max_val, nullptr, nullptr);

    return max_val - min_val;
}

}  // namespace pipeline
}  // namespace hymson3d
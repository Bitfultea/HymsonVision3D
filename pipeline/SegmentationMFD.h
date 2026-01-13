#pragma once
#include "FileTool.h"
#include "PointCloud.h"
#include "yolo_3d/segment_3d.h"
namespace hymson3d {
namespace pipeline {

struct Defect {
    int area_mask_pixel;
    float area_bbox;
    float height;
    cv::Point2i center;
    float half_w;
    float half_h;
    std::string label;
    int label_id;
    float score;
};

class SegmentationMFD {
public:
    SegmentationMFD();  // 无参构造
    SegmentationMFD(const std::string& model_path,
                    cv::Size size = cv::Size{960, 960},
                    int topk = 100,
                    int seg_h = 160,
                    int seg_w = 160,
                    int seg_channels = 32);
    ~SegmentationMFD() {};

public:
    void run_segmentation(const std::string& tiff_file_path,
                          float score_thres = 0.25f,
                          float iou_thres = 0.65f,
                          bool debug_mode = false);
    bool run_segmentation_dll(cv::Mat& tiff_file_path,
                          float score_thres = 0.25f,
                          float iou_thres = 0.65f,
                          bool debug_mode = false);

    void analysis_defetcts(std::vector<Defect>& defects);
    float get_masked_defect_height(cv::Rect mask);
    bool enableModel(bool enable,
                    const std::string& model_path,
                    cv::Size size,
                    int topk,
                    int seg_h,
                    int seg_w,
                    int seg_channelss);

private:
    std::unique_ptr<ml::Segmentation_ML_3d> model_ = nullptr;
    cv::Mat m_raw_tiff;
};
}  // namespace pipeline
}  // namespace hymson3d
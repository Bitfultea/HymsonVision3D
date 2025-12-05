#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "FileTool.h"
#include "Logger.h"
#include "SegmentationMFD.h"

using namespace hymson3d;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <tiff_folder>" << std::endl;
        return -1;
    }

    std::string tiff_path = argv[1];
    //std::string tiff_path = "F:\\TestData\\Yolo3d\\CapTime_Dev7_2025_11_23_00_57_49_541_CurTime_00_57_49_869_0B5CBP2MBL0P8MFBD1012148.tiff";

    LOG_INFO("Starting segmentation...");

    std::string engine_file = "../ml/yolo_3d/tools/hymson3d-seg.engine";
    //std::string engine_file = "F:\\TestData\\Yolo3d\\hymson3d-seg.engine";

    // Model parameters
    cv::Size image_size = cv::Size(960, 960);
    int topk = 100;
    int seg_h = 160;
    int seg_w = 160;
    int seg_channels = 32;

    pipeline::SegmentationMFD segmentation_mfd(engine_file, image_size, topk,
                                               seg_h, seg_w, seg_channels);

    // infer params
    float score_thres = 0.4f;
    float iou_thres = 0.65f;
    bool debug_mode = true;
    segmentation_mfd.run_segmentation(tiff_path, score_thres, iou_thres,
                                      debug_mode);

    // calculate defect figures
    std::vector<pipeline::Defect> defects;
    segmentation_mfd.analysis_defetcts(defects);

    for (auto& defect : defects) {
        LOG_INFO("Defect: {};  area:{}; height:{}; centre:[{},{}]",
                 defect.label, defect.area_bbox, defect.height, defect.center.x,
                 defect.center.y);
    }

    LOG_INFO("Segmentation done.");
}
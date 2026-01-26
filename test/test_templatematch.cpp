#include <iostream>
#include <opencv2/opencv.hpp>

// int main() {
//     // 1. 加载源图像和模板图像
//     // 将 "source_image.jpg" 替换为你的源图像路径
//     // 将 "template_image.jpg" 替换为你的模板图像路径
//     cv::Mat srcImage = cv::imread(
//             "/home/charles/Data/Test/预焊2_1/"
//             "20260105_001629_04QCB52091380JG120006435.jpg",
//             cv::IMREAD_COLOR);
//     cv::Mat templateImage = cv::imread(
//             "/home/charles/Data/Test/预焊2_1/emplate.jpg", cv::IMREAD_COLOR);

//     // 检查图像是否成功加载
//     if (srcImage.empty() || templateImage.empty()) {
//         std::cerr << "错误: 无法加载图像!" << std::endl;
//         return -1;
//     }

//     // 2. 创建用于存放结果的矩阵
//     // 结果矩阵的尺寸: (W - w + 1, H - h + 1)
//     // W, H 是源图像的宽高; w, h 是模板图像的宽高
//     int result_cols = srcImage.cols - templateImage.cols + 1;
//     int result_rows = srcImage.rows - templateImage.rows + 1;
//     cv::Mat resultImage;
//     resultImage.create(result_rows, result_cols, CV_32FC1);

//     auto start = std::chrono::high_resolution_clock::now();
//     // 3. 执行模板匹配
//     // TM_CCOEFF_NORMED 是一种常用的匹配算法，它返回归一化的相关系数
//     // 结果越接近 1，表示匹配度越高
//     cv::matchTemplate(srcImage, templateImage, resultImage,
//                       cv::TM_CCOEFF_NORMED);

//     // 4. 找到最佳匹配位置
//     double minVal, maxVal;
//     cv::Point minLoc, maxLoc;
//     cv::Point matchLoc;

//     cv::minMaxLoc(resultImage, &minVal, &maxVal, &minLoc, &maxLoc,
//     cv::Mat());

//     // 对于 TM_CCOEFF_NORMED 方法，最佳匹配点是最大值所在的位置
//     matchLoc = maxLoc;

//     // 5. 在源图像上绘制矩形框来标记匹配区域
//     cv::rectangle(srcImage, matchLoc,
//                   cv::Point(matchLoc.x + templateImage.cols,
//                             matchLoc.y + templateImage.rows),
//                   cv::Scalar(0, 255, 0), 2, 8, 0);

//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> elapsed = end - start;
//     std::cout << "Running times:" << elapsed.count() << "ms" << std::endl;

//     // 6. 显示结果
//     cv::namedWindow("匹配结果", cv::WINDOW_AUTOSIZE);
//     int display_width = std::min(1200, srcImage.cols);  // 最大宽度1200像素
//     int display_height = std::min(800, srcImage.rows);  // 最大高度800像素
//     cv::resizeWindow("匹配结果", display_width, display_height);
//     // cv::imshow("模板图像", templateImage);
//     cv::imshow("匹配结果", srcImage);

//     // 等待用户按键后退出
//     cv::waitKey(0);

//     return 0;
// }

#include <execution>  // C++17 并行协议
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

struct MatchResult {
    cv::Point pos;
    double score;
};

class FastShapeMatcher {
public:
    // 使用金字塔加速匹配
    MatchResult matchWithPyramid(const cv::Mat& scene,
                                 const cv::Mat& templ,
                                 int levels = 2) {
        std::vector<cv::Mat> scenePyramid, templPyramid;

        // 1. 构建金字塔
        cv::buildPyramid(scene, scenePyramid, levels);
        cv::buildPyramid(templ, templPyramid, levels);

        MatchResult bestResult = {cv::Point(0, 0), -1.0};

        // 2. 在顶层（最小图）进行全图搜索
        cv::Mat result;
        cv::matchTemplate(scenePyramid[levels], templPyramid[levels], result,
                          cv::TM_CCOEFF_NORMED);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

        bestResult.pos = maxLoc;
        bestResult.score = maxVal;

        // 3. 逐层向下细化定位
        for (int l = levels - 1; l >= 0; --l) {
            // 将上一层的坐标映射到当前层
            int searchX = bestResult.pos.x * 2;
            int searchY = bestResult.pos.y * 2;

            // 设定一个小的搜索窗口 (ROI)，避免全图搜索
            int margin = 10;
            int roiX = std::max(0, searchX - margin);
            int roiY = std::max(0, searchY - margin);
            int roiW = std::min(scenePyramid[l].cols - roiX,
                                templPyramid[l].cols + margin * 2);
            int roiH = std::min(scenePyramid[l].rows - roiY,
                                templPyramid[l].rows + margin * 2);

            cv::Rect roi(roiX, roiY, roiW, roiH);
            cv::Mat sceneROI = scenePyramid[l](roi);

            cv::matchTemplate(sceneROI, templPyramid[l], result,
                              cv::TM_CCOEFF_NORMED);
            cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

            // 更新全局坐标
            bestResult.pos = cv::Point(roiX + maxLoc.x, roiY + maxLoc.y);
            bestResult.score = maxVal;
        }

        return bestResult;
    }
};

int main() {
    // 载入图像（建议使用灰度图提升处理速度）
    // cv::Mat scene = cv::imread("scene.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat templ = cv::imread("template.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat scene = cv::imread(
            "/home/charles/Data/Test/预焊2_1/"
            "20260105_001629_04QCB52091380JG120006435.jpg",
            cv::IMREAD_GRAYSCALE);
    cv::Mat templ = cv::imread("/home/charles/Data/Test/预焊2_1/emplate.jpg",
                               cv::IMREAD_GRAYSCALE);

    if (scene.empty() || templ.empty()) return -1;

    FastShapeMatcher matcher;

    // 计时开始
    int64 start = cv::getTickCount();

    // 执行加速匹配
    MatchResult res = matcher.matchWithPyramid(scene, templ, 9);

    double duration = (cv::getTickCount() - start) / cv::getTickFrequency();
    std::cout << "匹配耗时: " << duration << " 秒" << std::endl;
    std::cout << "最高分: " << res.score << std::endl;

    // 绘制结果
    cv::Mat display;
    cv::cvtColor(scene, display, cv::COLOR_GRAY2BGR);
    cv::rectangle(display,
                  cv::Rect(res.pos.x, res.pos.y, templ.cols, templ.rows),
                  cv::Scalar(0, 255, 0), 2);

    cv::imshow("Result", display);
    cv::waitKey(0);

    return 0;
}
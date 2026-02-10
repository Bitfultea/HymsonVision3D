#include "Surface.h"

#include <cmath>

namespace hymson3d {
namespace geometry {

// 拟合二次曲面: z = ax^2 + by^2 + cxy + dx + ey + f
// 返回生成的参考面图像
cv::Mat Surface::FitQuadraticSurface(const cv::Mat& heightMap,
                                     cv::Mat& mask,
                                     int iterations,
                                     int step,
                                     float outlierThreshold) {
    int rows = heightMap.rows;
    int cols = heightMap.cols;

    cv::Mat currentHeightMap = heightMap;
    // cv::Mat mask =
    //         cv::Mat::ones(rows, cols, CV_8U);  // 初始掩码：所有点都参与拟合

    // 存储拟合系数 (a0...a5)
    cv::Mat coeffs;

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<double> A_data;
        std::vector<double> b_data;

        // 1. 构建矩阵 A 和向量 b (Ax = b)
        // 只有在 mask 为 1 (非异常点) 时才采样
        // 为了性能，不需要每个像素都采样，可以每隔几点采一个 (例如 step=4)
        for (int y = 0; y < rows; y += step) {
            for (int x = 0; x < cols; x += step) {
                if (mask.at<uchar>(y, x) == 0) continue;

                float z = currentHeightMap.at<float>(y, x);

                // 忽略无效值 (如果你的深度图背景是0或NaN)
                if (z <= 0.0001f) continue;

                // 构建行: [x^2, y^2, xy, x, y, 1]
                A_data.push_back(x * x);
                A_data.push_back(y * y);
                A_data.push_back(x * y);
                A_data.push_back(x);
                A_data.push_back(y);
                A_data.push_back(1.0);

                b_data.push_back(z);
            }
        }

        // 转换为 OpenCV 矩阵
        int numSamples = b_data.size();
        if (numSamples < 6) break;  // 样本太少无法拟合

        cv::Mat A(numSamples, 6, CV_64F, A_data.data());
        cv::Mat b(numSamples, 1, CV_64F, b_data.data());

        // 2.求解线性方程: A * coeffs = b
        // 使用 SVD 分解求解最小二乘问题
        cv::solve(A, b, coeffs, cv::DECOMP_SVD);

        // 如果是最后一次迭代，不需要更新掩码，直接跳出
        if (iter == iterations - 1) break;

        // 3. 计算残差并更新掩码 (剔除异常点)
        // 使用当前系数计算预测面，找出偏差大的点
        double* c = (double*)coeffs.data;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                float z_actual = currentHeightMap.at<float>(y, x);
                if (z_actual <= 0.0001f) continue;

                double z_pred = c[0] * x * x + c[1] * y * y + c[2] * x * y +
                                c[3] * x + c[4] * y + c[5];

                if (std::abs(z_actual - z_pred) > outlierThreshold) {
                    mask.at<uchar>(y, x) = 0;  // 标记为异常，下次不参与拟合
                }
            }
        }
    }

    // 4. 重建完整的参考面图像
    cv::Mat referenceSurface(rows, cols, CV_32F);
    double* c = (double*)coeffs.data;

    for (int y = 0; y < rows; ++y) {
        float* ptr = referenceSurface.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            // z = ax^2 + by^2 + cxy + dx + ey + f
            ptr[x] = (float)(c[0] * x * x + c[1] * y * y + c[2] * x * y +
                             c[3] * x + c[4] * y + c[5]);
        }
    }

    return referenceSurface;
}

Eigen::Vector6d Surface::FitQuadraticSurface(const cv::Mat& heightMap,
                                             float thresholdVal,
                                             const int sample_step) {
    const int rows = heightMap.rows;
    const int cols = heightMap.cols;

    // sample points every sample_step pixels
    std::vector<Eigen::VectorXd> samples;

    samples.reserve((rows / sample_step) * (cols / sample_step));

    // ---  Normal Equation ---
    // z = a*x^2 + b*y^2 + c*xy + d*x + e*y + f
    Eigen::MatrixXd AtA = Eigen::MatrixXd::Zero(6, 6);
    Eigen::VectorXd Atb = Eigen::VectorXd::Zero(6);

#pragma omp parallel
    {
        Eigen::MatrixXd localAtA = Eigen::MatrixXd::Zero(6, 6);
        Eigen::VectorXd localAtb = Eigen::VectorXd::Zero(6);

#pragma omp for nowait
        for (int y = 0; y < rows; y += sample_step) {
            for (int x = 0; x < cols; x += sample_step) {
                float z = heightMap.at<float>(y, x);
                if (z == 0) continue;

                double x2 = (double)x * x;
                double y2 = (double)y * y;
                double xy = (double)x * y;

                Eigen::Vector6d r;
                r << x2, y2, xy, (double)x, (double)y, 1.0;

                // 利用对称性优化秩 1 更新
                localAtA.selfadjointView<Eigen::Lower>().rankUpdate(r);
                localAtb += r * (double)z;
            }
        }
#pragma omp critical
        {
            AtA += localAtA;
            Atb += localAtb;
        }
    }
    AtA.fill(0);  // 补齐对称矩阵的上三角
    AtA.triangularView<Eigen::Upper>() =
            AtA.triangularView<Eigen::Lower>().transpose();
    // AtA.selfadjointView<Eigen::Upper>() =
    // AtA.selfadjointView<Eigen::Lower>();

    Eigen::Vector6d coeffs = AtA.ldlt().solve(Atb);
    double a = coeffs[0], b = coeffs[1], c = coeffs[2], d = coeffs[3],
           e = coeffs[4], f = coeffs[5];

    return coeffs;
}

}  // namespace geometry
}  // namespace hymson3d
#include <chrono>

#include "3D/Mesh.h"
#include "Cluster.h"
#include "Converter.h"
#include "Curvature.h"
#include "DefectDetection.h"
#include "FileTool.h"
#include "GapStepDetection.h"
#include "Logger.h"
#include "MathTool.h"
#include "Normal.h"
#include "PlaneDetection.h"
#include "fmtfallback.h"

using namespace hymson3d;
int main(int argc, char **argv) {
    // Eigen::VectorXd x(8);
    // Eigen::VectorXd y(8);
    // x << 0, 0.05, 0.20, 0.3, 0.40, 0.60, 0.70, 0.95;
    // y << 0, 0.20, 0.35, 0.4, 0.42, 0.43, 0.44, 0.45;
    // std::vector<Eigen::Vector2d> control_pts;
    // for (int i = 0; i < 8; i++) {
    //     control_pts.emplace_back(Eigen::Vector2d(x[i], y[i]));
    // }

    geometry::PointCloud::Ptr pointcloud =
            std::make_shared<geometry::PointCloud>();
    utility::read_ply(argv[1], pointcloud);
    //     core::converter::tiff_to_pointcloud(argv[1], argv[2], pointcloud,
    //                                         Eigen::Vector3d(0.01, 0.03,
    //                                         0.001), true);
    //     core::converter::tiff_to_pointcloud(
    //             argv[1], pointcloud, Eigen::Vector3d(0.01, 0.03, 0.001),
    //             true);

    // core::PlaneDetection plane_detector;
    // plane_detector.fit_a_curve(control_pts, 100);

    //     float height_threshold = atof(argv[3]);
    //     float radius = atof(argv[4]);
    //     size_t min_points = (size_t)atoi(argv[5]);
    //     float long_normal_degree = 2;
    //     float long_curvature_threshold = 0.3;
    //     float rcorner_normal_degree = 1.0;
    //     float rcorner_curvature_threshold = 0.05;
    //     geometry::KDTreeSearchParamRadius param(0.08);

    bool denoise = false;
    bool debug_mode = true;
    Eigen::Vector3d transformation_matrix = Eigen::Vector3d(0.005, 0.1, 1);
    for (int i = 0; i < pointcloud->points_.size(); i++) {
        pointcloud->points_[i] = {
                pointcloud->points_[i].x() * transformation_matrix.x(),
                pointcloud->points_[i].y() * transformation_matrix.y(),
                pointcloud->points_[i].z() * transformation_matrix.z()};
    }
    //     pipeline::DefectDetection::detect_CSAD(
    //             pointcloud, param, height_threshold, radius,
    //             min_points, transformation_matrix, denoise,
    //             debug_mode);
    pipeline::GapStepDetection::detect_gap_step(
            pointcloud, transformation_matrix, debug_mode);

    //     Eigen::MatrixXd M = Eigen::MatrixXd::Random(10, 10);
    //     // 手动添加异常值（稀疏矩阵部分）
    //     M(2, 3) += 10;
    //     M(5, 7) -= 15;

    //     Eigen::MatrixXd L, S;

    //     // 运行 RPCA
    //     utility::mathtool::RPCA(M, L, S);

    //     // 打印结果

    //     std::cout << "Original Matrix M:\n" << M << "\n";
    //     std::cout << "Low-rank Matrix L:\n" << L << "\n";
    //     std::cout << "Sparse Matrix S:\n" << S << "\n";
    // void GapStepDetection::detect_gap_step_dll_plot(
    //        std::shared_ptr<geometry::PointCloud> cloud,
    //        Eigen::Vector3d transformation_matrix, double& gap_step,
    //        double& step_width, double& height_threshold,
    //        std::vector<std::vector<double>>& temp_res, std::string&
    //        debug_path, bool debug_mode)

    ///---------------------------------------------------
    //     geometry::PointCloud::Ptr pointcloud =
    //             std::make_shared<geometry::PointCloud>();
    //     utility::read_ply("F:/qhchen/EncapsulationCplus/EncapsulationCplus/output_pointcloud.ply",
    //     pointcloud); double step_height = 0; double step_width = 0;
    //     //double height_threshold = 0.01;
    //     Eigen::Vector3d transformation_matrix = Eigen::Vector3d(0.005, 0.1,
    //     1); std::vector<std::vector<double>> temp_res; temp_res.resize(2);
    //     std::string debug_path =
    //     "C:\\Users\\Administrator\\Desktop\\res\\bspline\\"; bool debug_mode
    //     = true;

    //     for (int i = 0; i < 3; i++) {
    //         std::cout << "current i: " << i << std::endl;
    //         //pipeline::GapStepDetection::detect_gap_step_dll_plot(
    //         //        pointcloud, transformation_matrix, step_height,
    //         step_width,
    //         //    height_threshold, temp_res, debug_path,
    //         //    debug_mode);
    //         pipeline::GapStepDetection::detect_gap_step_dll_plot(
    //                 pointcloud, transformation_matrix, step_height,
    //                 step_width, temp_res, debug_path, debug_mode);
    //         pointcloud->points_.clear();
    //         pointcloud->y_slices_.clear();
    //         pointcloud->y_slice_peaks.clear();
    //         //pointcloud->normals_.clear();  // 可选
    //         //pointcloud->colors_.clear();   // 可选
    //         //pointcloud->labels_.clear();   // 可选
    //     utility::read_ply(
    //             "F:/qhchen/EncapsulationCplus/EncapsulationCplus/"
    //             "output_pointcloud.ply",
    //             pointcloud);
    //     step_height = 0;
    //     step_width = 0;
    //         //height_threshold = 0.01;
    //         transformation_matrix = Eigen::Vector3d(0.005, 0.1, 1);
    //         temp_res.clear();
    //         temp_res.resize(2);
    //         //std::vector<std::vector<double>> temp_res;
    //         debug_path =
    //                 "C:\\Users\\Administrator\\Desktop\\res\\bspline\\";
    //         debug_mode = true;
    // }
    //     system("pause");
    return 0;
}
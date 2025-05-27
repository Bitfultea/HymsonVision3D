#include "3D/PointCloud.h"
#include "Normal.h"
#include "fmtfallback.h"

namespace hymson3d {
namespace pipeline {

class DefectDetection {
public:
    // convex and concave
    static void detect_defects(std::shared_ptr<geometry::PointCloud> cloud,
                               geometry::KDTreeSearchParamRadius param,
                               float long_normal_degree,
                               float long_curvature_threshold,
                               float rcorner_normal_degree,
                               float rcorner_curvature_threshold,
                               float height_threshold = 0.0,
                               float radius = 0.08,
                               size_t min_points = 5,
                               size_t min_defects_size = 500,
                               bool debug_mode = true);

    static void detect_pinholes(std::shared_ptr<geometry::PointCloud> cloud,
                                geometry::KDTreeSearchParamRadius param,
                                float height_threshold = 0.0,
                                float radius = 0.08,
                                size_t min_points = 5,
                                Eigen::Vector3d transformation_matrix =
                                        Eigen::Vector3d(0.01, 0.03, 0.001),
                                bool denoise = true,
                                bool debug_mode = true);

    static void detect_pinholes_nva(std::shared_ptr<geometry::PointCloud> cloud,
                                    geometry::KDTreeSearchParamRadius param,
                                    float height_threshold = 0.0,
                                    float radius = 0.08,
                                    size_t min_points = 5,
                                    Eigen::Vector3d transformation_matrix =
                                            Eigen::Vector3d(0.01, 0.03, 0.001),
                                    float ratio_x = 0.4,
                                    float ratio_y = 0.4,
                                    double dist_x = 1e-4,
                                    double dist_y = 1e-4,
                                    bool denoise = true,
                                    bool debug_mode = true);

    static void detect_pinholes_nva_dll(std::shared_ptr<geometry::PointCloud> cloud,
                                geometry::KDTreeSearchParamRadius param,
                                std::vector<geometry::PointCloud::Ptr>& filtered_defects,
                                std::string &debug_path,
                                float height_threshold = 0.0,
                                float radius = 0.08,
                                size_t min_points = 5,
                                Eigen::Vector3d transformation_matrix =
                                        Eigen::Vector3d(0.01, 0.03, 0.001),
                                float ratio_x = 0.4,
                                float ratio_y = 0.4,
                                double dist_x = 1e-4,
                                double dist_y = 1e-4,
                                bool denoise = true,
                                bool debug_mode = true);

    static void detect_CSAD(std::shared_ptr<geometry::PointCloud> cloud,
                            geometry::KDTreeSearchParamRadius param,
                            float height_threshold = 0.0,
                            float radius = 0.08,
                            size_t min_points = 5,
                            Eigen::Vector3d transformation_matrix =
                                    Eigen::Vector3d(0.01, 0.03, 0.001),
                            bool denoise = true,
                            bool debug_mode = true);

private:
    static void process_y_slice(std::vector<Eigen::Vector2d> &y_slice,
                                std::vector<Eigen::Vector3d> &ny_slice,
                                std::vector<double> &y_derivative,
                                std::vector<size_t> &local_idxs);

    // Pipeline of turbine blade defect detection based on local geometric
    // pattern analysis
    static std::shared_ptr<geometry::PointCloud> FPFH_NVA(
            std::shared_ptr<geometry::PointCloud> cloud,
            float ratio_x,
            float ratio_y,
            double dist_x,
            double dist_y,
            bool use_fpfh = false);

    static void height_filter(std::shared_ptr<geometry::PointCloud> cloud,
                              std::shared_ptr<geometry::PointCloud> points,
                              float height_threshold);

    static void part_separation(
            std::shared_ptr<geometry::PointCloud> cloud,
            std::vector<geometry::PointCloud::Ptr> &clusters,
            int num_clusters);

    static void extract_long_edge(
            std::vector<geometry::PointCloud::Ptr> &long_clouds,
            std::vector<geometry::PointCloud::Ptr> &corners_clouds,
            std::vector<geometry::PointCloud::Ptr> &clusters,
            int num_clusters);

    //     static void fpfh_filter(std::shared_ptr<geometry::PointCloud> cloud,
    //                             std::vector<int> &fpfh_marker,
    //                             Eigen::MatrixXd &fpfh_matrix);

    static void slice_along_y(
            std::vector<geometry::PointCloud::Ptr> &long_clouds,
            Eigen::Vector3d transformation_matrix);

    static void slice_along_x(
            std::vector<geometry::PointCloud::Ptr> &long_clouds,
            Eigen::Vector3d transformation_matrix);

    static Eigen::MatrixXd bspline_interpolation(
            geometry::PointCloud::Ptr cloud);

    static void generate_low_rank_matrix(Eigen::MatrixXd &mat);

    static void plot_matrix(Eigen::MatrixXd &mat,
                            std::string name = "matrix.png");
};

}  // namespace pipeline
}  // namespace hymson3d
#include "3D/PointCloud.h"
#include "Normal.h"
#include "fmtfallback.h"

namespace hymson3d {
namespace pipeline {
class GapStepDetection {
    typedef std::vector<std::pair<Eigen::Vector2d, Eigen::Vector2d>>
            lineSegments;

public:
    static void detect_gap_step(std::shared_ptr<geometry::PointCloud> cloud,
                                Eigen::Vector3d transformation_matrix,
                                bool debug_mode);
    static void detect_gap_step_dll(std::shared_ptr<geometry::PointCloud> cloud,
                                    Eigen::Vector3d transformation_matrix,
                                    double& gap_step,
                                    double& step_width,
                                    bool debug_mode);
    static void detect_gap_step_dll_plot(std::shared_ptr<geometry::PointCloud> cloud,
                                Eigen::Vector3d transformation_matrix,
                                double& gap_step,
                                double& step_width,
                                double& height_threshold,
                                std::vector<std::vector<double>>& temp_res,
                                std::string& debug_path,
                                bool debug_mode);
    static void detect_gap_step_dll_plot2(
            std::shared_ptr<geometry::PointCloud> cloud,
            Eigen::Vector3d transformation_matrix,
            double& gap_step,
            double& step_width,
            double& height_threshold,
            std::vector<std::vector<double>>& temp_res,
            std::string& debug_path,
            bool LHT,
            bool debug_mode);

    static void slice_along_y(geometry::PointCloud::Ptr cloud,
                              Eigen::Vector3d transformation_matrix);

    static void bspline_interpolation(geometry::PointCloud::Ptr cloud,
                                      double height_threshold,
                                      lineSegments& corners,
                                      bool debug_mode);
    static void bspline_interpolation_dll(geometry::PointCloud::Ptr cloud,
                                      double height_threshold,
                                      lineSegments& corners,
                                      std::string& debug_path,
                                      bool debug_mode);
    static void bspline_interpolation_dll2(geometry::PointCloud::Ptr cloud,
                                          double height_threshold,
                                          double up_height_threshold,
                                          lineSegments& corners,
                                          std::string& debug_path,
                                          bool debug_mode);
    static void calculate_gap_step(lineSegments& corners,
                                   double& gap_step,
                                   double& step_width);
    static void calculate_gap_step_dll_plot(lineSegments& corners,
                                   double& gap_step,
                                   double& step_width,
                                   std::vector<std::vector<double>>& temp_res);

private:
    static std::vector<std::vector<Eigen::Vector2d>> group_by_derivative(
            std::vector<Eigen::Vector2d>& sampled_pts);
    static std::vector<std::vector<Eigen::Vector2d>> group_by_derivative_dll(
            std::vector<Eigen::Vector2d>& sampled_pts);
    static std::vector<std::vector<Eigen::Vector2d>> statistics_filter(
            std::vector<std::vector<Eigen::Vector2d>>& clusters);
    static std::vector<std::vector<Eigen::Vector2d>>  statistics_filter(
            std::vector<std::vector<Eigen::Vector2d>>& clusters,
            std::vector<Eigen::Vector2d>& limit_pts);
    static void plot_clusters(
            std::vector<Eigen::Vector2d>& resampled_pts,
            std::vector<std::vector<Eigen::Vector2d>>& clusters,
            lineSegments& line_segs,
            std::vector<std::vector<Eigen::Vector2d>> intersections,
            int img_id);
    static void plot_clusters_dll(
            std::vector<Eigen::Vector2d>& resampled_pts,
            std::vector<std::vector<Eigen::Vector2d>>& clusters,
            lineSegments& line_segs,
            std::vector<std::vector<Eigen::Vector2d>> intersections,
            std::vector<Eigen::Vector2d>& limit_pts,
            std::string& debug_path,
            int img_id);
    static void plot_clusters_dll(
            std::vector<Eigen::Vector2d>& resampled_pts,
            std::vector<std::vector<Eigen::Vector2d>>& clusters,
            lineSegments& line_segs,
            std::vector<std::vector<Eigen::Vector2d>> intersections,
            std::string& debug_path,
            int img_id);
    static lineSegments line_segment(
            std::vector<std::vector<Eigen::Vector2d>>& pt_groups);
    static lineSegments line_segment_dll(
            std::vector<std::vector<Eigen::Vector2d>>& pt_groups,
            std::vector<std::vector<Eigen::Vector2d>>& filter_pt_groups,
            double& left_height_threshold,
            double& right_height_threshold);
    static void compute_step_width(
            std::vector<Eigen::Vector2d>& resampled_pts,
            lineSegments& line_segs,
            std::vector<std::vector<Eigen::Vector2d>>& intersections,
            double height_threshold);
    static void compute_step_width_dll(
            std::vector<Eigen::Vector2d>& resampled_pts,
            lineSegments& line_segs,
            std::vector<std::vector<Eigen::Vector2d>>& intersections,
            double& left_height_threshold,
            double& right_height_threshold,
            std::vector<Eigen::Vector2d>& limit_pts,
            double& up_height_threshold);
};

}  // namespace pipeline
}  // namespace hymson3d
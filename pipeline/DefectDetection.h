#include "3D/PointCloud.h"
#include "Normal.h"

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

private:
    static void process_y_slice(std::vector<Eigen::Vector2d> &y_slice,
                                std::vector<Eigen::Vector3d> &ny_slice,
                                std::vector<double> &y_derivative,
                                std::vector<size_t> &local_idxs);
};

}  // namespace pipeline
}  // namespace hymson3d
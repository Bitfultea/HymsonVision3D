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
    static void detect_pinholes();
};

}  // namespace pipeline
}  // namespace hymson3d
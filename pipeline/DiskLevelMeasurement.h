#include "3D/PointCloud.h"
#include "Normal.h"
#include "PlaneDetection.h"
#include "fmtfallback.h"

namespace hymson3d {
namespace pipeline {

struct DiskLevelMeasurementResult {
    double plane_angle;
    double plane_height_gap;
};

class DiskLevelMeasurement {
public:
    static void measure_pindisk_heightlevel(
            std::shared_ptr<geometry::PointCloud> cloud,
            geometry::KDTreeSearchParamRadius param,
            DiskLevelMeasurementResult *result,
            float central_plane_size = 200.0,
            float normal_angle_threshold = 0.0,
            float distance_threshold = 0.0,
            int min_planar_points = 100,
            bool debug_mode = true);

    static void measure_pindisk_heightlevel(
            std::shared_ptr<geometry::PointCloud> cloud,
            DiskLevelMeasurementResult *result,
            float central_plane_size = 200.0,
            bool debug_mode = true);

private:
    static void segment_plane_instances(
            std::shared_ptr<geometry::PointCloud> cloud,
            geometry::KDTreeSearchParamRadius param,
            std::vector<geometry::Plane::Ptr> &planes,
            float normal_angle_threshold = 0.0,
            int min_planar_points = 100,
            bool debug_mode = true);

    // useful when detection of outlier ring is required
    static void merge_plane_instances(
            std::shared_ptr<geometry::PointCloud> cloud,
            std::vector<geometry::Plane::Ptr> &planes,
            float plane_distance_threshold = 1.0);

    // determine the central plane and bottom plane
    static std::pair<geometry::Plane::Ptr, geometry::Plane::Ptr>
    identify_plane_instances(std::shared_ptr<geometry::PointCloud> cloud,
                             std::vector<geometry::Plane::Ptr> &planes,
                             float central_plane_size = 200.0,
                             bool detect_bottom_plane = true,
                             bool debug_mode = true);

    static DiskLevelMeasurementResult calculate_planes_figure(
            std::pair<geometry::Plane::Ptr, geometry::Plane::Ptr> input_planes);

    static geometry::Plane::Ptr get_plane_in_range(
            geometry::PointCloud::Ptr cloud, float central_plane_size);
};

}  // namespace pipeline
}  // namespace hymson3d
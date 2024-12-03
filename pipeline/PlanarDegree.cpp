#include "PlanarDegree.h"

#include "Distance.h"

namespace hymson3d {
namespace pipeline {

double PlanarDegree::compute_planar_degree(geometry::PointCloud& pointcloud) {
    core::PlaneDetection plane_detector;
    geometry::Plane::Ptr plane = plane_detector.fit_a_plane(pointcloud);
    if (!plane) {
        return -1;
    }
    double upper_dist = 0;
    double lower_dist = 0;
    for (int i = 0; i < pointcloud.points_.size(); ++i) {
        double dist = core::feature::point_to_plane_signed_distance(
                pointcloud.points_[i], *plane);
        if (dist < lower_dist) lower_dist = dist;
        if (dist > upper_dist) upper_dist = dist;
    }
    // std::cout << "Plane coeff " << plane->coeff_ << std::endl;
    return upper_dist - lower_dist;
}

}  // namespace pipeline
}  // namespace hymson3d
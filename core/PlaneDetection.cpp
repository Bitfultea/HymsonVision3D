#include "PlaneDetection.h"

namespace hymson3d {
namespace core {

geometry::Plane::Ptr PlaneDetection::fit_a_plane(
        geometry::PointCloud& pointcloud) {
    Eigen::Vector3d centroid = pointcloud.GetCenter();

    // Calculate full 3x3 covariance matrix, excluding symmetries:
    double xx = 0, xy = 0, xz = 0;
    double yy = 0, yz = 0, zz = 0;

    for (auto pt : pointcloud.points_) {
        Eigen::Vector3d r = pt - centroid;
        xx += r.x() * r.x();
        xy += r.x() * r.y();
        xz += r.x() * r.z();
        yy += r.y() * r.y();
        yz += r.y() * r.z();
        zz += r.z() * r.z();
    }

    // TODO::Experiemnt with following
    // from https://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
    size_t n = pointcloud.points_.size();
    xx /= n;
    xy /= n;
    xz /= n;
    yy /= n;
    yz /= n;
    zz /= n;

    Eigen::Vector3d weighted_dir(0, 0, 0);
    // x direction
    double det_x = yy * zz - yz * yz;
    Eigen::Vector3d axis_dir_x(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
    double weight_x = det_x * det_x;
    if (weighted_dir.dot(axis_dir_x) < 0.0) weight_x = -weight_x;
    weighted_dir += axis_dir_x * weight_x;

    // y direction
    double det_y = xx * zz - xz * xz;
    Eigen::Vector3d axis_dir_y(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
    double weight_y = det_y * det_y;
    if (weighted_dir.dot(axis_dir_y) < 0.0) weight_y = -weight_y;
    weighted_dir += axis_dir_y * weight_y;

    // z direction
    double det_z = xx * yy - xy * xy;
    Eigen::Vector3d axis_dir_z(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
    double weight_z = det_z * det_z;
    if (weighted_dir.dot(axis_dir_z) < 0.0) weight_z = -weight_z;
    weighted_dir += axis_dir_z * weight_z;

    double norm = weighted_dir.norm();
    if (norm == 0) {
        LOG_ERROR("Invalid Plane Normal Detected!");
    }
    weighted_dir /= weighted_dir.norm();  // normaliszed
    double d = -weighted_dir.dot(centroid);

    Eigen::Vector4d plane_coeff(weighted_dir.x(), weighted_dir.y(),
                                weighted_dir.z(), d);
    geometry::Plane::Ptr plane = std::make_shared<geometry::Plane>();
    plane->coeff_ = plane_coeff;
    plane->normal_ = weighted_dir;
    return plane;
}
}  // namespace core
}  // namespace hymson3d
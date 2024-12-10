#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>

namespace hymson3d {
namespace core {

/// 相机的简单头文件， 不考虑相机模型的， 只考虑投影矩阵
// TODO: Work with GL Camera
class Camera {
public:
    typedef std::shared_ptr<Camera> Ptr;
    enum class FovType { Vertical, Horizontal };
    enum class Projection { Perspective, Ortho };
    using Transform = Eigen::Transform<float, 3, Eigen::Affine>;
    using ProjectionMatrix = Eigen::Transform<float, 3, Eigen::Projective>;

public:
    Camera() = default;
    Camera(Eigen::Vector3d position,
           Eigen::Vector3d direction,
           Eigen::Vector3d camera_up) {
        camera_position_ = position;
        camera_direction_ = direction;
        camera_up_ = camera_up;
        update_extrinsic();
    }

public:
    Eigen::Vector3d get_camera_position() { return camera_position_; }
    void set_camera_position(Eigen::Vector3d position) {
        camera_position_ = position;
    }
    Eigen::Vector3d get_camera_direction() { return camera_direction_; }
    void set_camera_direction(Eigen::Vector3d direction) {
        camera_direction_ = direction;
    }
    void set_camera_up(Eigen::Vector3d up) { camera_up_ = up; }
    void set_intrinsics(Eigen::Matrix3d intrinsics) { intrinsic_ = intrinsics; }
    Eigen::Matrix3d get_intrinsics() { return intrinsic_; }

    void update_extrinsic() {
        // opposite of camera direction
        Eigen::Vector3d z_cam = -camera_direction_.normalized();
        // camera right
        Eigen::Vector3d x_cam = camera_up_.cross(z_cam).normalized();
        Eigen::Vector3d y_cam = z_cam.cross(x_cam).normalized();  // camera up
        Eigen::Matrix3d Rotation;
        Rotation << x_cam, y_cam, z_cam;
        Eigen::Vector3d translation = -Rotation * camera_position_;
        extrinsic_ = Eigen::Matrix4d::Identity();
        extrinsic_.block<3, 3>(0, 0) = Rotation;
        extrinsic_.block<3, 1>(0, 3) =
                translation;  //[R|t] matrix with homogeneous
    }

    Eigen::Vector3d get_camera_up() { return camera_up_; }
    Eigen::Matrix4d get_extrinsic() { return extrinsic_; }
    void set_extrinsic(Eigen::Matrix4d extrinsic) { extrinsic_ = extrinsic; }

public:
    Eigen::Matrix4d projection_matrix;
    Eigen::Matrix4d view_matrix;
    Eigen::Matrix4d model_matrix;

protected:
    Eigen::Vector3d camera_position_;
    Eigen::Vector3d camera_direction_;
    Eigen::Vector3d camera_up_;
    Eigen::Matrix3d intrinsic_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix4d extrinsic_;  // world to camera
};

}  // namespace core
}  // namespace hymson3d
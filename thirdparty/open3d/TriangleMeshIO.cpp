// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "TriangleMeshIO.h"

#include <unordered_map>

#include "FileSystem.h"
#include "Logger.h"
#include "ProgressBar.h"
#include <rply.h>


namespace open3d {

namespace {
using namespace io;

//static const std::unordered_map<
//        std::string,
//        std::function<bool(const std::string &,
//                           geometry::TriangleMesh &,
//                           const ReadTriangleMeshOptions &)>>
//        file_extension_to_trianglemesh_read_function{
//                {"ply", ReadTriangleMeshFromPLY},
//                {"stl", ReadTriangleMeshUsingASSIMP},
//                {"obj", ReadTriangleMeshUsingASSIMP},
//                {"off", ReadTriangleMeshFromOFF},
//                {"gltf", ReadTriangleMeshUsingASSIMP},
//                {"glb", ReadTriangleMeshUsingASSIMP},
//                {"fbx", ReadTriangleMeshUsingASSIMP},
//        };

static const std::unordered_map<
        std::string,
        std::function<bool(const std::string &,
                           const geometry::TriangleMesh &,
                           const bool,
                           const bool,
                           const bool,
                           const bool,
                           const bool,
                           const bool)>>
        file_extension_to_trianglemesh_write_function{
                {"ply", WriteTriangleMeshToPLY},
              
        };

}  // unnamed namespace

namespace io {

    bool WriteTriangleMeshToPLY(
        const std::string &filename,
        const open3d::geometry::TriangleMesh &mesh,
        bool write_ascii /* = false*/,
        bool compressed /* = false*/,
        bool write_vertex_normals /* = true*/,
        bool write_vertex_colors /* = true*/,
        bool write_triangle_uvs /* = true*/,
        bool print_progress) {
    p_ply ply_file = ply_create(filename.c_str(),
                                write_ascii ? PLY_ASCII : PLY_LITTLE_ENDIAN,
                                NULL, 0, NULL);

    write_vertex_normals = write_vertex_normals && mesh.HasVertexNormals();
    write_vertex_colors = write_vertex_colors && mesh.HasVertexColors();

    ply_add_comment(ply_file, "Created by Open3D");
    ply_add_element(ply_file, "vertex",
                    static_cast<long>(mesh.vertices_.size()));
    ply_add_property(ply_file, "x", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "y", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    ply_add_property(ply_file, "z", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    if (write_vertex_normals) {
        ply_add_property(ply_file, "nx", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "ny", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
        ply_add_property(ply_file, "nz", PLY_DOUBLE, PLY_DOUBLE, PLY_DOUBLE);
    }
    if (write_vertex_colors) {
        ply_add_property(ply_file, "red", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "green", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
        ply_add_property(ply_file, "blue", PLY_UCHAR, PLY_UCHAR, PLY_UCHAR);
    }
    ply_add_element(ply_file, "face",
                    static_cast<long>(mesh.triangles_.size()));
    ply_add_property(ply_file, "vertex_indices", PLY_LIST, PLY_UCHAR, PLY_UINT);
    if (!ply_write_header(ply_file)) {
        // utility::LogWarning("Write PLY failed: unable to write header.");
        ply_close(ply_file);
        return false;
    }

    open3d::utility::ProgressBar progress_bar(
            static_cast<size_t>(mesh.vertices_.size() + mesh.triangles_.size()),
            "Writing PLY: ", print_progress);
    bool printed_color_warning = false;
    for (size_t i = 0; i < mesh.vertices_.size(); i++) {
        const auto &vertex = mesh.vertices_[i];
        ply_write(ply_file, vertex(0));
        ply_write(ply_file, vertex(1));
        ply_write(ply_file, vertex(2));
        if (write_vertex_normals) {
            const auto &normal = mesh.vertex_normals_[i];
            ply_write(ply_file, normal(0));
            ply_write(ply_file, normal(1));
            ply_write(ply_file, normal(2));
        }
        if (write_vertex_colors) {
            const auto &color = mesh.vertex_colors_[i];
            if (!printed_color_warning &&
                (color(0) < 0 || color(0) > 1 || color(1) < 0 || color(1) > 1 ||
                 color(2) < 0 || color(2) > 1)) {
                LOG_WARN("Write Ply clamped color value to valid range");
                printed_color_warning = true;
            }
            auto rgb = open3d::utility::ColorToUint8(color);
            ply_write(ply_file, rgb(0));
            ply_write(ply_file, rgb(1));
            ply_write(ply_file, rgb(2));
        }
        ++progress_bar;
    }
    for (size_t i = 0; i < mesh.triangles_.size(); i++) {
        const auto &triangle = mesh.triangles_[i];
        ply_write(ply_file, 3);
        ply_write(ply_file, triangle(0));
        ply_write(ply_file, triangle(1));
        ply_write(ply_file, triangle(2));
        ++progress_bar;
    }

    ply_close(ply_file);
    return true;
}
//std::shared_ptr<geometry::TriangleMesh> CreateMeshFromFile(
//        const std::string &filename, bool print_progress) {
//    auto mesh = std::make_shared<geometry::TriangleMesh>();
//    ReadTriangleMeshOptions opt;
//    opt.print_progress = print_progress;
//    ReadTriangleMesh(filename, *mesh, opt);
//    return mesh;


//bool ReadTriangleMesh(const std::string &filename,
//                      geometry::TriangleMesh &mesh,
//                      ReadTriangleMeshOptions params /*={}*/) {
//    std::string filename_ext =
//            open3d::utility::filesystem::GetFileExtensionInLowerCase(filename);
//    if (filename_ext.empty()) {
//        utility::LogWarning(
//                "Read geometry::TriangleMesh failed: unknown file "
//                "extension.");
//        return false;
//    }
//    auto map_itr =
//            file_extension_to_trianglemesh_read_function.find(filename_ext);
//    if (map_itr == file_extension_to_trianglemesh_read_function.end()) {
//        utility::LogWarning(
//                "Read geometry::TriangleMesh failed: unknown file "
//                "extension.");
//        return false;
//    }
//
//    if (params.print_progress) {
//        auto progress_text = std::string("Reading ") +
//                             utility::ToUpper(filename_ext) +
//                             " file: " + filename;
//        auto pbar = utility::ProgressBar(100, progress_text, true);
//        params.update_progress = [pbar](double percent) mutable -> bool {
//            pbar.SetCurrentCount(size_t(percent));
//            return true;
//        };
//    }
//
//    bool success = map_itr->second(filename, mesh, params);
//    utility::LogDebug(
//            "Read geometry::TriangleMesh: {:d} triangles and {:d} vertices.",
//            (int)mesh.triangles_.size(), (int)mesh.vertices_.size());
//    if (mesh.HasVertices() && !mesh.HasTriangles()) {
//        utility::LogWarning(
//                "geometry::TriangleMesh appears to be a geometry::PointCloud "
//                "(only contains vertices, but no triangles).");
//    }
//    return success;
//}

bool WriteTriangleMesh(const std::string &filename,
                       const geometry::TriangleMesh &mesh,
                       bool write_ascii /* = false*/,
                       bool compressed /* = false*/,
                       bool write_vertex_normals /* = true*/,
                       bool write_vertex_colors /* = true*/,
                       bool write_triangle_uvs /* = true*/,
                       bool print_progress /* = false*/) {
    std::string filename_ext =
            open3d::utility::filesystem::GetFileExtensionInLowerCase(filename);
    if (filename_ext.empty()) {
        utility::LogWarning(
                "Write geometry::TriangleMesh failed: unknown file "
                "extension.");
        return false;
    }
    auto map_itr =
            file_extension_to_trianglemesh_write_function.find(filename_ext);
    if (map_itr == file_extension_to_trianglemesh_write_function.end()) {
        utility::LogWarning(
                "Write geometry::TriangleMesh failed: unknown file "
                "extension.");
        return false;
    }
    bool success = map_itr->second(filename, mesh, write_ascii, compressed,
                                   write_vertex_normals, write_vertex_colors,
                                   write_triangle_uvs, print_progress);
    utility::LogDebug(
            "Write geometry::TriangleMesh: {:d} triangles and {:d} vertices.",
            (int)mesh.triangles_.size(), (int)mesh.vertices_.size());
    return success;
}

// Reference: https://stackoverflow.com/a/43896965
//bool IsPointInsidePolygon(const Eigen::MatrixX2d &polygon, double x, double y) {
//    bool inside = false;
//    for (int i = 0; i < polygon.rows(); ++i) {
//        // i and j are the indices of the first and second vertices.
//        int j = (i + 1) % polygon.rows();
//
//        // The vertices of the edge that will be checked.
//        double vx0 = polygon(i, 0);
//        double vy0 = polygon(i, 1);
//        double vx1 = polygon(j, 0);
//        double vy1 = polygon(j, 1);
//
//        // Check whether the edge intersects a line from (-inf,y) to (x,y).
//        // First, check if the line crosses the horizontal line at y in either
//        // direction.
//        if (((vy0 <= y) && (vy1 > y)) || ((vy1 <= y) && (vy0 > y))) {
//            // If so, get the point where it crosses that line.
//            double cross = (vx1 - vx0) * (y - vy0) / (vy1 - vy0) + vx0;
//
//            // Finally, check if it crosses to the left of the test point.
//            if (cross < x) inside = !inside;
//        }
//    }
//    return inside;
//}
//
//bool AddTrianglesByEarClipping(geometry::TriangleMesh &mesh,
//                               std::vector<unsigned int> &indices) {
//    int n = int(indices.size());
//    Eigen::Vector3d face_normal = Eigen::Vector3d::Zero();
//    if (n > 3) {
//        for (int i = 0; i < n; i++) {
//            const Eigen::Vector3d &v1 = mesh.vertices_[indices[(i + 1) % n]] -
//                                        mesh.vertices_[indices[i % n]];
//            const Eigen::Vector3d &v2 = mesh.vertices_[indices[(i + 2) % n]] -
//                                        mesh.vertices_[indices[(i + 1) % n]];
//            face_normal += v1.cross(v2);
//        }
//        double l = std::sqrt(face_normal.dot(face_normal));
//        face_normal *= (1.0 / l);
//    }
//
//    bool found_ear = true;
//    while (n > 3) {
//        if (!found_ear) {
//            // If no ear is found after all indices are looped through, the
//            // polygon is not triangulable.
//            return false;
//        }
//
//        found_ear = false;
//        for (int i = 1; i < n - 1; i++) {
//            const Eigen::Vector3d &v1 =
//                    mesh.vertices_[indices[i]] - mesh.vertices_[indices[i - 1]];
//            const Eigen::Vector3d &v2 =
//                    mesh.vertices_[indices[i + 1]] - mesh.vertices_[indices[i]];
//            bool is_convex = (face_normal.dot(v1.cross(v2)) > 0.0);
//            bool is_ear = true;
//            if (is_convex) {
//                // If convex, check if vertex is an ear
//                // (no vertices within triangle v[i-1], v[i], v[i+1])
//                Eigen::MatrixX2d polygon(3, 2);
//                for (int j = 0; j < 3; j++) {
//                    polygon(j, 0) = mesh.vertices_[indices[i + j - 1]](0);
//                    polygon(j, 1) = mesh.vertices_[indices[i + j - 1]](1);
//                }
//
//                for (int j = 0; j < n; j++) {
//                    if (j == i - 1 || j == i || j == i + 1) {
//                        continue;
//                    }
//                    const Eigen::Vector3d &v = mesh.vertices_[indices[j]];
//                    if (IsPointInsidePolygon(polygon, v(0), v(1))) {
//                        is_ear = false;
//                        break;
//                    }
//                }
//
//                if (is_ear) {
//                    found_ear = true;
//                    mesh.triangles_.push_back(Eigen::Vector3i(
//                            indices[i - 1], indices[i], indices[i + 1]));
//                    indices.erase(indices.begin() + i);
//                    n = int(indices.size());
//                    break;
//                }
//            }
//        }
//    }
//    mesh.triangles_.push_back(
//            Eigen::Vector3i(indices[0], indices[1], indices[2]));
//
//    return true;
//}

}  // namespace io
}  // namespace open3d

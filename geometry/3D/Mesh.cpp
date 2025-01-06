#include "Mesh.h"

namespace hymson3d {
namespace geometry {

void HymsonMesh::construct_mesh(std::shared_ptr<geometry::PointCloud> cloud) {
    Point_set point_set;
    for (auto pt : cloud->points_) {
        point_set.insert(Point_3(pt.x(), pt.y(), pt.z()));
    }
    Mesh2d delaunay_mesh(point_set.points().begin(), point_set.points().end());
    // debug
    Mesh output_mesh;
    CGAL::copy_face_graph(delaunay_mesh, output_mesh);
    std::ofstream mesh_ofile("mesh.ply", std::ios_base::binary);
    CGAL::IO::set_binary_mode(mesh_ofile);
    CGAL::IO::write_PLY(mesh_ofile, output_mesh);
    mesh_ofile.close();
}

}  // namespace geometry
}  // namespace hymson3d
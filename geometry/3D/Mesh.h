#include <CGAL/Classification.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Constrained_triangulation_plus_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/WKT.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/locate.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polyline_simplification_2/Squared_distance_cost.h>
#include <CGAL/Polyline_simplification_2/simplify.h>
#include <CGAL/Projection_traits_xy_3.h>
#include <CGAL/Random.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/boost/graph/copy_face_graph.h>
#include <CGAL/boost/graph/graph_traits_Delaunay_triangulation_2.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>
#include <CGAL/compute_average_spacing.h>

#include <boost/graph/adjacency_list.hpp>
#include <fstream>
#include <queue>

#include "3D/PointCloud.h"

namespace hymson3d {
namespace geometry {
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Projection_traits = CGAL::Projection_traits_xy_3<Kernel>;
using Point_2 = Kernel::Point_2;
using Point_3 = Kernel::Point_3;
using Segment_3 = Kernel::Segment_3;

// Triangulated Irregular Network (with info)
using Point_set = CGAL::Point_set_3<Point_3>;
using Vbi = CGAL::Triangulation_vertex_base_with_info_2<Point_set::Index,
                                                        Projection_traits>;
using Fbi = CGAL::Triangulation_face_base_with_info_2<int, Projection_traits>;
using TDS = CGAL::Triangulation_data_structure_2<Vbi, Fbi>;
using TIN_with_info = CGAL::Delaunay_triangulation_2<Projection_traits, TDS>;

using CDT_vertex_base =
        CGAL::Polyline_simplification_2::Vertex_base_2<Projection_traits>;
using CDT_face_base =
        CGAL::Constrained_triangulation_face_base_2<Projection_traits>;
using CDT_TDS =
        CGAL::Triangulation_data_structure_2<CDT_vertex_base, CDT_face_base>;
using CDT =
        CGAL::Constrained_Delaunay_triangulation_2<Projection_traits, CDT_TDS>;
using CTP = CGAL::Constrained_triangulation_plus_2<CDT>;

// Triangulated Irregular Network
// TIN
using Mesh2d = CGAL::Delaunay_triangulation_2<Projection_traits>;
// TIN with Info
using Mesh2d_with_info = CGAL::Delaunay_triangulation_2<Projection_traits, TDS>;
using Mesh = CGAL::Surface_mesh<Point_3>;

class HymsonMesh {
public:
    HymsonMesh() {}
    ~HymsonMesh() {}

public:
    void construct_mesh(std::shared_ptr<geometry::PointCloud> cloud);
};

}  // namespace geometry
}  // namespace hymson3d
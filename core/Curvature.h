#include <Eigen/Sparse>

#include "3D/KDtree.h"
#include "3D/PointCloud.h"

namespace hymson3d {
namespace core {
namespace feature {
void ComputeCurvature_PCL(geometry::PointCloud& cloud,
                          geometry::KDTreeSearchParam& param);

void ComputeCurvature_PCA(geometry::PointCloud& cloud,
                          geometry::KDTreeSearchParam& param);

void ComputeSurfaceVariation(geometry::PointCloud& cloud,
                             geometry::KDTreeSearchParam& param);

// TNV:Triangle Normal Variation from https://arxiv.org/pdf/2305.12653
// TODO:: implement fast curvature computation
void ComputeCurvature_TNV(geometry::PointCloud& cloud,
                          geometry::KDTreeSearchParam& param);

Eigen::Vector3d color_with_curvature(double curvature,
                                     double min_val,
                                     double max_val);

std::pair<double, double> calculate_point_curvature(geometry::PointCloud& cloud,
                                                    Eigen::Vector3d normal,
                                                    Eigen::Vector3d pt,
                                                    std::vector<int> indices);

Eigen::SparseMatrix<double> CotangentLaplacian(const Eigen::MatrixXd& V,
                                               const Eigen::MatrixXi& F);

class TotalCurvaturePointCloud {
public:
    static std::vector<int> Where(int i, const Eigen::MatrixXi& inArray);

    static bool compare_triangles(const Eigen::Vector3i& t1,
                                  const Eigen::Vector3i& t2);

    static Eigen::MatrixXi DelaunayKNN(
            const Eigen::MatrixXd& knn_locations_including_self,
            const Eigen::MatrixXi& idx);

    static double PerTriangleLaplacianTriangleFanCurvature(
            const Eigen::MatrixXi& adjacent_triangles_idx,
            const Eigen::MatrixXd& V,
            const Eigen::MatrixXd& N);

    static void TotalCurvaturePCD(const geometry::PointCloud& cloud,
                                  std::vector<double>& k_S,
                                  int knn);
};

}  // namespace feature
}  // namespace core
}  // namespace hymson3d
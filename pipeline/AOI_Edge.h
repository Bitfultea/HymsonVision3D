#include "3D/PointCloud.h"
#include "Converter.h"
#include "3D/KDtree.h"
#include "PlaneDetection.h"
#include <Eigen/Core>
#include "Curvature.h"
#include "../core/Filter.h"
#include "Cluster.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/region_growing.h>
namespace hymson3d {
namespace pipeline {

class AOI_Detect{
struct FittedLine {
    Eigen::Vector3f point;       // 直线上的点
    Eigen::Vector3f direction;   // 直线方向向量
    pcl::PointIndices::Ptr inliers; // 该直线的内点索引
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
};
public:
    AOI_Detect(){}
    ~AOI_Detect(){}
     static std::shared_ptr<geometry::PointCloud> AOI_Detect::read_tiff_img(cv::Mat tiff_img, Eigen::Vector3d a);
    static    std::shared_ptr<geometry::PointCloud> read_tiff(std::string& tiff_path, Eigen::Vector3d a);
    static std::vector<Eigen::Vector3d> cloudPreDeal(std::shared_ptr<geometry::PointCloud> cloud,float threshold,size_t nb_neighbors, double std_ratio,float ClusterTolerance ,float DistanceThreshold,double distance_between_lines);
   //static std::vector<FittedLine> mergeCollinearLines(const std::vector<FittedLine>& inputLines, const pcl::PointCloud<pcl::PointXYZ>::Ptr& originalCloud,  double distance_between_lines,double angleThreshold = 150.0,  double distanceThreshold = 0.05) ;

    //static std::vector<Eigen::Vector3d> spliteCloud(std::shared_ptr<geometry::PointCloud> cloud);
    
};





}
}

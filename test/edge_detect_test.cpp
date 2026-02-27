#include "AOI_Edge.h"


int main(){
   
   std::string path="C:/Users/Administrator/Desktop/image_1.tiff";
 
   std::shared_ptr<hymson3d::geometry::PointCloud> cloud ;
   Eigen::Vector3d a(0.1, 0.1, 10);
   cloud = hymson3d::pipeline::AOI_Detect::read_tiff(path,a);
   Eigen::Vector3d max = cloud->GetMaxBound();
   Eigen::Vector3d min = cloud->GetMinBound();
   float d = max.z()-min.z();
 
   float threshold = 0.001;
   size_t nb_neighbors = 5;
   double std_ratio=0.3;
   float ClusterTolerance = 0.2;
   float DistanceThreshold = 0.5;
   double distance_between_lines = 5;
   auto contour = hymson3d::pipeline::AOI_Detect::cloudPreDeal(cloud,threshold,nb_neighbors,std_ratio,ClusterTolerance,DistanceThreshold,distance_between_lines);
   std::cout<<"contours.size():"<<contour.size()<<std::endl;
}
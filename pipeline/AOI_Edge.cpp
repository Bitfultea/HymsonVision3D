#include "AOI_Edge.h"

namespace hymson3d {
namespace pipeline {

std::shared_ptr<geometry::PointCloud> AOI_Detect::read_tiff_img(
        cv::Mat tiff_img, Eigen::Vector3d a) {
    geometry::PointCloud::Ptr cloud = std::make_shared<geometry::PointCloud>();
    core::converter::tiff_to_pointcloud_img(tiff_img, cloud, a);
    // utility::write_ply("C:/Users/Administrator/Desktop/aaac.ply",cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pclPoint(
            new pcl::PointCloud<pcl::PointXYZ>());
    pclPoint->width = cloud->points_.size();
    pclPoint->height = 1;
    pclPoint->is_dense = false;
    pclPoint->points.resize(cloud->points_.size());
    for (int i = 0; i < cloud->points_.size(); i++) {
        pclPoint->points[i].x = cloud->points_[i].x();
        pclPoint->points[i].y = cloud->points_[i].y();
        pclPoint->points[i].z = cloud->points_[i].z();
    }

    return cloud;
}

std::shared_ptr<geometry::PointCloud> AOI_Detect::read_tiff(
        std::string& tiff_path, Eigen::Vector3d a) {
    geometry::PointCloud::Ptr cloud = std::make_shared<geometry::PointCloud>();
    core::converter::tiff_to_pointcloud(tiff_path, cloud, a);
    // utility::write_ply("C:/Users/Administrator/Desktop/aaac.ply",cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr pclPoint(
            new pcl::PointCloud<pcl::PointXYZ>());
    pclPoint->width = cloud->points_.size();
    pclPoint->height = 1;
    pclPoint->is_dense = false;
    pclPoint->points.resize(cloud->points_.size());
    for (int i = 0; i < cloud->points_.size(); i++) {
        pclPoint->points[i].x = cloud->points_[i].x();
        pclPoint->points[i].y = cloud->points_[i].y();
        pclPoint->points[i].z = cloud->points_[i].z();
    }

    return cloud;
}

std::vector<Eigen::Vector3d> AOI_Detect::cloudPreDeal(
        std::shared_ptr<geometry::PointCloud> cloud,
        float threshold,
        size_t nb_neighbors,
        double std_ratio,
        float ClusterTolerance,
        float DistanceThreshold,
        double distance_between_lines) {
    geometry::KDTreeSearchParamRadius param(0.3);
    cloud->colors_.resize(cloud->points_.size());

    core::feature::ComputeCurvature_PCL(*cloud, param);
    std::vector<double> cur(cloud->points_.size());
#pragma omp parallel for
    for (int i = 0; i < cloud->points_.size(); i++) {
        cur[i] = cloud->curvatures_[i].total_curvature;
    }
    std::sort(cur.begin(), cur.end());
    if (!cloud->HasColors()) {
        cloud->colors_.reserve(cloud->points_.size());
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr pclPoint(
            new pcl::PointCloud<pcl::PointXYZ>());
    pclPoint->width = cloud->points_.size();
    pclPoint->height = 1;
    pclPoint->is_dense = false;
    pclPoint->points.reserve(cloud->points_.size());
    for (int i = 0; i < cloud->curvatures_.size(); ++i) {
        if (cloud->curvatures_[i].total_curvature > threshold) {
            cloud->colors_[i].x() = 0.8;
            cloud->colors_[i].y() = 0.0;
            cloud->colors_[i].z() = 0.0;
            pcl::PointXYZ pt;
            // 坐标赋值
            pt.x = cloud->points_[i].x();
            pt.y = cloud->points_[i].y();
            pt.z = cloud->points_[i].z();
            pclPoint->push_back(pt);
        }
    }

    // std::shared_ptr<geometry::PointCloud> filter_cloud
    // =std::make_shared<geometry::PointCloud>();
    // filter_cloud->points_.reserve(pclPoint->points.size());

    // for (int i = 0; i < pclPoint->points.size(); ++i) {
    //     Eigen::Vector3d point(pclPoint->points[i].x, pclPoint->points[i].y,
    //                           pclPoint->points[i].z);
    //     filter_cloud->points_.push_back(point);
    // }
    /// 使用pcl的半径滤波
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
    ror.setInputCloud(pclPoint);
    ror.setRadiusSearch(std_ratio);
    ror.setMinNeighborsInRadius(nb_neighbors + 1);  // PCL 包含自身，所以+1

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(
            new pcl::PointCloud<pcl::PointXYZ>());
    ror.filter(*cloud_filtered);

    // 将 int 索引转换为 size_t
    // std::vector<size_t> indices(pcl_indices.begin(), pcl_indices.end());

    // std::tuple<pcl::PointCloud<pcl::PointXYZ>::Ptr,std::vector<int>> outCloud
    // = std::make_tuple(cloud_filtered, indices);

    // core::Filter filter;
    // auto outCloud =
    //         filter.RadiusOutliers(filter_cloud, nb_neighbors, std_ratio);

    // if (!std::get<0>(outCloud)->HasColors()) {
    //     std::get<0>(outCloud)->colors_.reserve(
    //             std::get<0>(outCloud)->points_.size());
    //     for (int i = 0; i < std::get<0>(outCloud)->points_.size(); ++i) {
    //         std::get<0>(outCloud)->colors_[i].x() = 0.8;
    //         std::get<0>(outCloud)->colors_[i].y() = 0.0;
    //         std::get<0>(outCloud)->colors_[i].z() = 0.0;
    //     }
    // }
    std::vector<Eigen::Vector3d> p_points;
    for (int i = 0; i < cloud_filtered->points.size(); ++i) {
        Eigen::Vector3d pt = Eigen::Vector3d(cloud_filtered->points[i].x,
                                             cloud_filtered->points[i].y,
                                             cloud_filtered->points[i].z);
        p_points.push_back(pt);
    }
    auto maxar = std::accumulate(
            p_points.begin(), p_points.end(), p_points[0],
            [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a.array().max(b.array()).matrix();
            });
    auto minar = std::accumulate(
            p_points.begin(), p_points.end(), p_points[0],
            [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
                return a.array().min(b.array()).matrix();
            });

    // auto maxar = std::get<0>(outCloud)->GetMaxBound();
    // auto minar = std::get<0>(outCloud)->GetMinBound();
    auto midz = (maxar.z() + minar.z()) / 2;
    std::shared_ptr<geometry::PointCloud> topCloud =
            std::make_shared<geometry::PointCloud>();
    std::shared_ptr<geometry::PointCloud> topCloud_ =
            std::make_shared<geometry::PointCloud>();
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud(
            new pcl::PointCloud<pcl::PointXYZ>());
    pcloud->width = p_points.size();
    pcloud->height = 1;
    pcloud->is_dense = false;
    pcloud->points.reserve(p_points.size());
    for (int i = 0; i < p_points.size(); ++i) {
        if (p_points[i].z() > midz) {
            topCloud->points_.push_back(Eigen::Vector3d(
                    p_points[i].x(), p_points[i].y(), p_points[i].z()));
            topCloud_->points_.push_back(
                    Eigen::Vector3d(p_points[i].x(), p_points[i].y(), 0));
            pcl::PointXYZ pt;
            // 坐标赋值
            pt.x = p_points[i].x();
            pt.y = p_points[i].y();
            pt.z = p_points[i].z();
            pcloud->push_back(pt);
        }
    }

    // 区域增长聚类
    // 计算法向量
    ///////使用角的检测
    //     pcl::search::Search<pcl::PointXYZ>::Ptr rgtree(new
    //     pcl::search::KdTree<pcl::PointXYZ>);
    //     pcl::PointCloud<pcl::Normal>::Ptr normals(new
    //     pcl::PointCloud<pcl::Normal>); pcl::NormalEstimation<pcl::PointXYZ,
    //     pcl::Normal> ne; ne.setSearchMethod(rgtree);
    //     ne.setInputCloud(pcloud);
    //     ne.setKSearch(5);//
    //     ne.compute(*normals);

    // std::vector<bool> is_corner(pcloud->size(), false);
    //     for (size_t i = 0; i < pcloud->size(); ++i) {
    //         std::vector<int> k_indices;
    //         std::vector<float> k_dists;
    //         rgtree->nearestKSearch(pcloud->points[i], 5, k_indices, k_dists);

    //         // 计算邻域内法线方差/最大夹角
    //         float max_angle = 0.0f;
    //         for (size_t j = 1; j < k_indices.size(); ++j) {
    //             float dot = normals->points[i].getNormalVector3fMap()
    //                        .dot(normals->points[k_indices[j]].getNormalVector3fMap());
    //             float angle = std::acos(std::min(1.0f, std::abs(dot))) *
    //             180.0f / M_PI; max_angle = std::max(max_angle, angle);
    //         }

    //         // L形的拐角处，邻域法线会有90°变化
    //         if (max_angle > 45.0f) {
    //             is_corner[i] = true;
    //         }
    //     }

    //     // 4. 以拐角点为界，做洪水填充分割
    //     // 找拐角点中最中心的作为分割起点
    //     size_t seed1 = 0, seed2 = 0;
    //     float max_dist = 0.0f;
    //     for (size_t i = 0; i < pcloud->size(); ++i) {
    //         for (size_t j = i + 1; j < pcloud->size(); ++j) {
    //             float d = (pcloud->points[i].getVector3fMap() -
    //                       pcloud->points[j].getVector3fMap()).norm();
    //             if (d > max_dist) {
    //                 max_dist = d;
    //                 seed1 = i;
    //                 seed2 = j;
    //             }
    //         }
    //     }

    //     // 从两个最远点开始做区域增长，拐角点作为屏障
    //     std::vector<int> labels(pcloud->size(), -1);
    //     std::queue<std::pair<size_t, int>> seeds;  // (索引, 类别)

    //     seeds.push({seed1, 0});
    //     seeds.push({seed2, 1});
    //     labels[seed1] = 0;
    //     labels[seed2] = 1;

    //     while (!seeds.empty()) {
    //         auto [curr, label] = seeds.front();
    //         seeds.pop();

    //         std::vector<int> k_indices;
    //         std::vector<float> k_dists;
    //         rgtree->nearestKSearch(pcloud->points[curr], 10, k_indices,
    //         k_dists);

    //         for (int nb : k_indices) {
    //             if (labels[nb] != -1) continue;  // 已标记
    //             if (is_corner[nb]) continue;      // 拐角点作为屏障

    //             labels[nb] = label;
    //             seeds.push({nb, label});
    //         }
    //     }

    //     // 5. 收集结果
    //     pcl::PointIndices arm1, arm2;
    //     for (size_t i = 0; i < pcloud->size(); ++i) {
    //         if (labels[i] == 0) arm1.indices.push_back(i);
    //         else if (labels[i] == 1) arm2.indices.push_back(i);
    //         else {
    //             // 未标记的点（拐角附近），按距离分配
    //             float d1 = (pcloud->points[i].getVector3fMap() -
    //                        pcloud->points[seed1].getVector3fMap()).norm();
    //             float d2 = (pcloud->points[i].getVector3fMap() -
    //                        pcloud->points[seed2].getVector3fMap()).norm();
    //             if (d1 < d2) arm1.indices.push_back(i);
    //             else arm2.indices.push_back(i);
    //         }
    //     }

    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_arm1(new
    // pcl::PointCloud<pcl::PointXYZRGB>);
    //     pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_arm2(new
    //     pcl::PointCloud<pcl::PointXYZRGB>);

    //     // Arm1 - 红色
    //     for (int idx : arm1.indices) {
    //         pcl::PointXYZRGB pt;
    //         pt.x = pcloud->points[idx].x;
    //         pt.y = pcloud->points[idx].y;
    //         pt.z = pcloud->points[idx].z;
    //         pt.r = 255;
    //         pt.g = 0;
    //         pt.b = 0;
    //         cloud_arm1->push_back(pt);
    //     }

    //     // Arm2 - 绿色
    //     for (int idx : arm2.indices) {
    //         pcl::PointXYZRGB pt;
    //         pt.x = pcloud->points[idx].x;
    //         pt.y = pcloud->points[idx].y;
    //         pt.z = pcloud->points[idx].z;
    //         pt.r = 0;
    //         pt.g = 255;
    //         pt.b = 0;
    //         cloud_arm2->push_back(pt);
    //     }

    //     // 设置点云属性
    //     cloud_arm1->width = cloud_arm1->size();
    //     cloud_arm1->height = 1;
    //     cloud_arm2->width = cloud_arm2->size();
    //     cloud_arm2->height = 1;

    //  pcl::visualization::PCLVisualizer viewer("L-Shape: Two Arms");
    //     viewer.setBackgroundColor(0, 0, 0);  // 黑色背景

    //     // 添加 Arm1（红色）- RGB点云自带颜色，不需要ColorHandler
    //     viewer.addPointCloud<pcl::PointXYZRGB>(cloud_arm1, "arm1");
    //     viewer.setPointCloudRenderingProperties(
    //         pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "arm1");

    //     // 添加 Arm2（绿色）
    //     viewer.addPointCloud<pcl::PointXYZRGB>(cloud_arm2, "arm2");
    //     viewer.setPointCloudRenderingProperties(
    //         pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "arm2");

    //     // 添加坐标系
    //     viewer.addCoordinateSystem(1.0);

    //    while (!viewer.wasStopped()) {
    //         viewer.spinOnce(100);
    //     }

    ////////////////下面这块算角度点云的，可以取消

    // Eigen::Vector3d center_ = topCloud_->GetCenter();

    // double angleResolution = 1.0;
    // int numAngles = static_cast<int>(360.0 / angleResolution);  // 360个扇区
    // std::vector<std::vector<Eigen::Vector3d>> angle_pipe(numAngles);

    // for (const auto& pt : topCloud->points_) {
    //     double dx = pt.x() - center_.x();
    //     double dy = pt.y() - center_.y();
    //     double angle = std::atan2(dy, dx) * 180.0 / M_PI;  // -180 ~ 180
    //     if (angle < 0) {
    //         angle += 360.0;  // 转换为 0 ~ 360
    //     };
    //     int angleIndex = static_cast<int>(angle / angleResolution) %
    //     numAngles; angle_pipe[angleIndex].push_back(pt);
    // }
    // std::vector<Eigen::Vector3d> result;
    // pcl::PointCloud<pcl::PointXYZ>::Ptr innerContour(
    //         new pcl::PointCloud<pcl::PointXYZ>());
    // innerContour->width = angle_pipe.size();
    // innerContour->height = 1;
    // innerContour->is_dense = false;
    // innerContour->points.reserve(angle_pipe.size());
    // for (int i = 0; i < angle_pipe.size(); ++i) {
    //     std::vector<std::tuple<double, Eigen::Vector3d>> dis;
    //     if (angle_pipe[i].size() != 0) {
    //         for (int j = 0; j < angle_pipe[i].size(); ++j) {
    //             double dx = angle_pipe[i][j].x() - center_.x();
    //             double dy = angle_pipe[i][j].y() - center_.y();
    //             double distance = std::sqrt(dx * dx + dy * dy);
    //             dis.push_back(std::make_tuple(
    //                     distance, Eigen::Vector3d(angle_pipe[i][j].x(),
    //                                               angle_pipe[i][j].y(),
    //                                               angle_pipe[i][j].z())));
    //         }
    //         std::sort(dis.begin(), dis.end(),
    //                   [](const std::tuple<double, Eigen::Vector3d> a,
    //                      const std::tuple<double, Eigen::Vector3d> b) {
    //                       return std::get<0>(a) < std::get<0>(b);
    //                   });
    //         pcl::PointXYZ pt;
    //         pt.x = std::get<1>(dis[0]).x();
    //         pt.y = std::get<1>(dis[0]).y();
    //         pt.z = 0;
    //         Eigen::Vector3d sample(std::get<1>(dis[0]).x() * 10,
    //                                std::get<1>(dis[0]).y() * 10,
    //                                std::get<1>(dis[0]).z() / 100);
    //         result.push_back(sample);
    //         innerContour->push_back(pt);
    //     }
    // }


    /////////////////////////////////////////


 pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);            // 优化直线参数
        seg.setModelType(pcl::SACMODEL_LINE);         // 拟合直线模型
        seg.setMethodType(pcl::SAC_RANSAC);           // RANSAC算法
        seg.setDistanceThreshold(DistanceThreshold);  // 内点距离阈值
        seg.setInputCloud(pcloud);
        seg.segment(*inliers, *coefficients);


    std::vector<FittedLine> fitted_lines;

    FittedLine line;
        line.point = Eigen::Vector3f(coefficients->values[0],
                                     coefficients->values[1],
                                     coefficients->values[2]);
        line.direction = Eigen::Vector3f(coefficients->values[3],
                                         coefficients->values[4],
                                         coefficients->values[5]);
        line.inliers = inliers;

        fitted_lines.push_back(line);



    // // -------------------------- 步骤1：欧式聚类分割点云（分成多个子集）
    // // -------------------------- 1.1
    // // 计算点云法向量（聚类可选，若点云稀疏可跳过）
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
    //         new pcl::search::KdTree<pcl::PointXYZ>);
    // tree->setInputCloud(pcloud);

    // // 1.2 欧式聚类（按空间距离分割点云）
    // std::vector<pcl::PointIndices> cluster_indices;
    // pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    // ec.setClusterTolerance(ClusterTolerance);  // 聚类距离阈值
    // ec.setMinClusterSize(20);                  // 最小聚类点数
    // ec.setMaxClusterSize(pcloud->size());      // 最大聚类点数
    // ec.setSearchMethod(tree);
    // ec.setInputCloud(pcloud);
    // ec.extract(cluster_indices);

    // std::set<int> clustered_index;
    // for (const auto& cluster : cluster_indices) {
    //     for (int idx : cluster.indices) {
    //         clustered_index.insert(idx);
    //     }
    // }
    // // 获取聚类剩余的点云的索引
    // std::vector<int> cluster_remain_index;
    // for (size_t i = 0; i < pcloud->size(); ++i) {
    //     if (clustered_index.find(i) == clustered_index.end()) {
    //         cluster_remain_index.push_back(i);
    //     }
    // }
    // pcl::PointIndices::Ptr remaining_ptr(new pcl::PointIndices);
    // remaining_ptr->indices = cluster_remain_index;
    // pcl::ExtractIndices<pcl::PointXYZ> extract_remaining_ptr;
    // extract_remaining_ptr.setInputCloud(pcloud);
    // extract_remaining_ptr.setIndices(remaining_ptr);
    // extract_remaining_ptr.setNegative(false);  // 提取指定索引
    // pcl::PointCloud<pcl::PointXYZ>::Ptr remain_cloud(
    //         new pcl::PointCloud<pcl::PointXYZ>);
    // extract_remaining_ptr.filter(*remain_cloud);  // 将剩余点云放入remain_cloud

    // std::cout << "remain_cloud->points.size()" << remain_cloud->size()
    //           << std::endl;
    // for (int i = 0; i < cluster_indices.size(); ++i) {
    //     std::cout << "cluster_indices.size()"
    //               << cluster_indices[i].indices.size() << std::endl;
    // }

    // pcl::ExtractIndices<pcl::PointXYZ> extract;
    // extract.setInputCloud(pcloud);
    // extract.setNegative(false);
    // std::vector<std::tuple<pcl::PointCloud<pcl::PointXYZ>::Ptr,
    //                        pcl::ModelCoefficients::Ptr, pcl::PointIndices::Ptr>>
    //         cluster_cloud_gather;
    // // 接下来对每一个聚类点集进行处理

    // for (const auto& indices : cluster_indices) {
    //     // 2.1 提取当前聚类的点云子集
    //     pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(
    //             new pcl::PointCloud<pcl::PointXYZ>);
    //     pcl::PointIndices::Ptr cluster_indices_ptr(
    //             new pcl::PointIndices(indices));
    //     extract.setIndices(cluster_indices_ptr);
    //     // 从innerContour中抽出该聚类集合中的点
    //     extract.filter(*cluster_cloud);

    //     // 2.2 RANSAC拟合单直线
    //     pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    //     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    //     pcl::SACSegmentation<pcl::PointXYZ> seg;
    //     seg.setOptimizeCoefficients(true);            // 优化直线参数
    //     seg.setModelType(pcl::SACMODEL_LINE);         // 拟合直线模型
    //     seg.setMethodType(pcl::SAC_RANSAC);           // RANSAC算法
    //     seg.setDistanceThreshold(DistanceThreshold);  // 内点距离阈值
    //     seg.setInputCloud(cluster_cloud);
    //     seg.segment(*inliers, *coefficients);

    //     // 2.3 验证拟合结果（过滤无效直线）
    //     if (inliers->indices.empty()) {
    //         std::cerr << "聚类子集拟合直线失败，跳过该子集" << std::endl;
    //         continue;
    //     }

    //     // 2.4
    //     // 解析直线参数（PCL直线系数格式：[x0,y0,z0,a,b,c]，(x0,y0,z0)是直线上点，(a,b,c)是方向向量）

    //     cluster_cloud_gather.push_back(
    //             std::make_tuple(cluster_cloud, coefficients, inliers));
    // }
    // std::cout << "cluster_cloud_gather.size()" << cluster_cloud_gather.size()
    //           << std::endl;
    // // 获取每一个聚类生成直线的角度，在范围内就合并
    // for (int i = 0; i < remain_cloud->size(); ++i) {
    //     for (int j = 0; j < cluster_cloud_gather.size(); j++) {
    //         auto cof = std::get<1>(cluster_cloud_gather[j]);
    //         Eigen::Vector3f line_p = Eigen::Vector3f(
    //                 cof->values[0], cof->values[1], cof->values[2]);
    //         Eigen::Vector3f line_dir = Eigen::Vector3f(
    //                 cof->values[3], cof->values[4], cof->values[5]);
    //         const pcl::PointXYZ& point = remain_cloud->points[i];
    //         Eigen::Vector3f p = Eigen::Vector3f(point.x, point.y, point.z);
    //         Eigen::Vector3f P0P = p - line_p;
    //         Eigen::Vector3f cross_product = P0P.cross(line_dir);
    //         double numerator = cross_product.norm();
    //         double denominator = line_dir.norm();
    //         double distance = numerator / denominator;
    //         if (distance < 1) {
    //             std::get<0>(cluster_cloud_gather[j])
    //                     ->push_back(pcl::PointXYZ(point.x, point.y, point.z));
    //         }
    //     }
    // }
    // for (int j = 0; j < cluster_cloud_gather.size(); j++) {
    //     auto cloud = std::get<0>(cluster_cloud_gather[j]);
    //     pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    //     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    //     pcl::SACSegmentation<pcl::PointXYZ> seg;
    //     seg.setOptimizeCoefficients(true);            // 优化直线参数
    //     seg.setModelType(pcl::SACMODEL_LINE);         // 拟合直线模型
    //     seg.setMethodType(pcl::SAC_RANSAC);           // RANSAC算法
    //     seg.setDistanceThreshold(DistanceThreshold);  // 内点距离阈值
    //     seg.setInputCloud(cloud);
    //     seg.segment(*inliers, *coefficients);
    //     FittedLine line;
    //     line.point = Eigen::Vector3f(coefficients->values[0],
    //                                  coefficients->values[1],
    //                                  coefficients->values[2]);
    //     line.direction = Eigen::Vector3f(coefficients->values[3],
    //                                      coefficients->values[4],
    //                                      coefficients->values[5]);
    //     line.inliers = inliers;

    //     fitted_lines.push_back(line);
    // }

    pcl::visualization::PCLVisualizer viewerx("Multiple Lines Fitting");
    viewerx.setBackgroundColor(0, 0, 0);

    // 1. 显示原始点云
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color(
            pcloud, 255, 255, 255);
    viewerx.addPointCloud(pcloud, cloud_color, "original_cloud");
    viewerx.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original_cloud");

    // 2. 显示拟合的直线（不同颜色区分）
    std::vector<Eigen::Vector3i> colors = {{255, 0, 0},
                                           {0, 255, 0},
                                           {0, 0, 255},
                                           {255, 255, 0},
                                           {255, 0, 255}};
    for (int i = 0; i < fitted_lines.size(); ++i) {
        // 计算直线的起止点（延长直线便于可视化）
        Eigen::Vector3f start =
                fitted_lines[i].point - fitted_lines[i].direction * 5.0f;
        Eigen::Vector3f end =
                fitted_lines[i].point + fitted_lines[i].direction * 5.0f;
        std::string line_id = "line_" + std::to_string(i);

        // 添加直线到可视化窗口
        viewerx.addLine<pcl::PointXYZ>(
                pcl::PointXYZ(start.x(), start.y(), start.z()),
                pcl::PointXYZ(end.x(), end.y(), end.z()),
                colors[i % colors.size()](0) / 255.0,
                colors[i % colors.size()](1) / 255.0,
                colors[i % colors.size()](2) / 255.0, line_id);
        viewerx.setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, line_id);
    }

    // 3. 显示坐标系
    viewerx.addCoordinateSystem(1.0);

    // 4. 循环可视化
    while (!viewerx.wasStopped()) {
        viewerx.spinOnce(100);
    }

    std::cout << "fitted_lines.size():" << fitted_lines.size() << std::endl;

    // pcl::visualization::PCLVisualizer viewer2("PCL Viewer");
    // viewer2.setBackgroundColor(0, 0, 0);
    // viewer2.addPointCloud<pcl::PointXYZ>(innerContour, "sample cloud");
    // viewer2.setPointCloudRenderingProperties(
    //         pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample
    //         cloud");
    // viewer2.addCoordinateSystem(1.0);
    // viewer2.initCameraParameters();

    // while (!viewer2.wasStopped()) {
    //     viewer2.spinOnce(100);  // 每100ms刷新一次
    //     std::this_thread::sleep_for(
    //             std::chrono::milliseconds(100));  // 避免CPU占用过高
    // }
    // return result;
}

// std::vector<Eigen::Vector3d> AOI_Detect::spliteCloud(
//         std::shared_ptr<geometry::PointCloud> cloud) {

//         }

}  // namespace pipeline

}  // namespace hymson3d
   
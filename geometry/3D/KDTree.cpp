#include "KDtree.h"

#include "Eigen.h"
#include "Logger.h"

namespace hymson3d {
namespace geometry {

KDTree::KDTree() {}

KDTree::KDTree(const Eigen::MatrixXd &data) { SetMatrixData(data); }

KDTree::KDTree(const Feature &feature) { SetFeature(feature); }

KDTree::~KDTree() {}

bool KDTree::SetMatrixData(const Eigen::MatrixXd &data) {
    return SetRawData(Eigen::Map<const Eigen::MatrixXd>(
            data.data(), data.rows(), data.cols()));
}

bool KDTree::SetFeature(const Feature &feature) {
    return SetMatrixData(feature.data_);
}

template <typename T>
int KDTree::Search(const T &query,
                   const KDTreeSearchParam &param,
                   std::vector<int> &indices,
                   std::vector<double> &distance2) const {
    switch (param.GetSearchType()) {
        case KDTreeSearchParam::SearchType::Knn:
            return SearchKNN(query, ((const KDTreeSearchParamKNN &)param).knn_,
                             indices, distance2);
        case KDTreeSearchParam::SearchType::Radius:
            return SearchRadius(
                    query, ((const KDTreeSearchParamRadius &)param).radius_,
                    indices, distance2);
        case KDTreeSearchParam::SearchType::Hybrid:
            return SearchHybrid(
                    query, ((const KDTreeSearchParamHybrid &)param).radius_,
                    ((const KDTreeSearchParamHybrid &)param).max_nn_, indices,
                    distance2);
        default:
            return -1;
    }
    return -1;
}

template <typename T>
int KDTree::SearchKNN(const T &query,
                      int knn,
                      std::vector<int> &indices,
                      std::vector<double> &distance2) const {
    // This is optimized code for heavily repeated search.
    // Other flann::Index::knnSearch() implementations lose performance due to
    // memory allocation/deallocation.
    if (data_.size() == 0 || query.rows() != data_.rows() || knn < 0) {
        return -1;
    }
    indices.resize(knn);
    distance2.resize(knn);
    std::vector<Eigen::Index> indices_eigen(knn);
    int k = nanoflann_index_->index_->knnSearch(
            query.data(), knn, indices_eigen.data(), distance2.data());
    indices.resize(k);
    distance2.resize(k);
    std::copy_n(indices_eigen.begin(), k, indices.begin());
    return k;
}

template <typename T>
int KDTree::SearchRadius(const T &query,
                         double radius,
                         std::vector<int> &indices,
                         std::vector<double> &distance2) const {
    if (data_.size() == 0 || query.rows() != data_.rows()) {
        return -1;
    }
    std::vector<nanoflann::ResultItem<Eigen::Index, double>> indices_dists;
    int k = nanoflann_index_->index_->radiusSearch(
            query.data(), radius * radius, indices_dists,
            nanoflann::SearchParameters(0.0));
    indices.resize(k);
    distance2.resize(k);
    for (int i = 0; i < k; ++i) {
        indices[i] = indices_dists[i].first;
        distance2[i] = indices_dists[i].second;
    }
    return k;
}

template <typename T>
int KDTree::SearchHybrid(const T &query,
                         double radius,
                         int max_nn,
                         std::vector<int> &indices,
                         std::vector<double> &distance2) const {
    if (data_.size() == 0 || query.rows() != data_.rows() || max_nn < 0) {
        return -1;
    }
    distance2.resize(max_nn);
    std::vector<Eigen::Index> indices_eigen(max_nn);
    int k = nanoflann_index_->index_->knnSearch(
            query.data(), max_nn, indices_eigen.data(), distance2.data());
    k = std::distance(distance2.begin(),
                      std::lower_bound(distance2.begin(), distance2.begin() + k,
                                       radius * radius));
    indices.resize(k);
    distance2.resize(k);
    std::copy_n(indices_eigen.begin(), k, indices.begin());
    return k;
}

bool KDTree::SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data) {
    if (data.size() == 0) {
        LOG_WARN("[KDTree::SetRawData] Failed due to no data.");
        return false;
    }
    data_ = data;
    nanoflann_index_ = std::make_unique<KDTree_t>(data_.rows(), data_, 15);
    nanoflann_index_->index_->buildIndex();
    return true;
}

template int KDTree::Search<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template int KDTree::SearchKNN<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        int knn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template int KDTree::SearchRadius<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        double radius,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template int KDTree::SearchHybrid<Eigen::Vector3d>(
        const Eigen::Vector3d &query,
        double radius,
        int max_nn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template int KDTree::Search<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        const KDTreeSearchParam &param,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template int KDTree::SearchKNN<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        int knn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template int KDTree::SearchRadius<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        double radius,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

template int KDTree::SearchHybrid<Eigen::VectorXd>(
        const Eigen::VectorXd &query,
        double radius,
        int max_nn,
        std::vector<int> &indices,
        std::vector<double> &distance2) const;

}  // namespace geometry
}  // namespace hymson3d
#pragma once
#include <Eigen/Core>
#include <memory>
#include <nanoflann.hpp>
#include <vector>

#include "Feature.h"
#include "Geometry3D.h"
#include "PointCloud.h"

namespace nanoflann {

struct metric_L2;
template <class MatrixType, int DIM, class Distance, bool row_major>
struct KDTreeEigenMatrixAdaptor;

}  // namespace nanoflann

namespace hymson3d {
namespace geometry {

// Base class for KDTree search parameters.
class KDTreeSearchParam {
public:
    enum class SearchType {
        Knn = 0,
        Radius = 1,
        Hybrid = 2,
    };

public:
    virtual ~KDTreeSearchParam() {}

protected:
    KDTreeSearchParam(SearchType type) : search_type_(type) {}

public:
    /// Get the search type (KNN, Radius, Hybrid) for the search parameter.
    SearchType GetSearchType() const { return search_type_; }

private:
    SearchType search_type_;
};

class KDTreeSearchParamKNN : public KDTreeSearchParam {
public:
    /// Specifies the knn neighbors that will searched. Default
    ///  is 30.
    KDTreeSearchParamKNN(int knn = 30)
        : KDTreeSearchParam(SearchType::Knn), knn_(knn) {}

public:
    /// Number of the neighbors that will be searched.
    int knn_;
};

class KDTreeSearchParamRadius : public KDTreeSearchParam {
public:
    KDTreeSearchParamRadius(double radius)
        : KDTreeSearchParam(SearchType::Radius), radius_(radius) {}

public:
    double radius_;
};

class KDTreeSearchParamHybrid : public KDTreeSearchParam {
public:
    KDTreeSearchParamHybrid(double radius, int max_nn)
        : KDTreeSearchParam(SearchType::Hybrid),
          radius_(radius),
          max_nn_(max_nn) {}

public:
    /// Search radius.
    double radius_;
    /// At maximum, max_nn neighbors will be searched.
    int max_nn_;
};

class KDTree {
public:
    KDTree();

    /// \param data Provides set of data points for KDTree construction.
    KDTree(const Eigen::MatrixXd &data);

    /// \param feature Provides a set of features from which the KDTree is
    /// constructed.
    KDTree(const Feature &feature);
    ~KDTree();
    KDTree(const KDTree &) = delete;
    KDTree &operator=(const KDTree &) = delete;

public:
    bool SetData(PointCloud &data) {
        std::cout << "start" << std::endl;
        SetRawData(Eigen::Map<const Eigen::MatrixXd>(
                (const double *)((const PointCloud &)data).points_.data(), 3,
                ((const PointCloud &)data).points_.size()));
        std::cout << "end" << std::endl;
    }

    /// Sets the data for the KDTree from a matrix.
    /// \param data Data points for KDTree Construction.
    bool SetMatrixData(const Eigen::MatrixXd &data);

    /// Sets the data for the KDTree from the feature data.
    /// \param feature Set of features for KDTree construction.
    bool SetFeature(const Feature &feature);

    template <typename T>
    int Search(const T &query,
               const KDTreeSearchParam &param,
               std::vector<int> &indices,
               std::vector<double> &distance2) const;

    template <typename T>
    int SearchKNN(const T &query,
                  int knn,
                  std::vector<int> &indices,
                  std::vector<double> &distance2) const;

    template <typename T>
    int SearchRadius(const T &query,
                     double radius,
                     std::vector<int> &indices,
                     std::vector<double> &distance2) const;

    template <typename T>
    int SearchHybrid(const T &query,
                     double radius,
                     int max_nn,
                     std::vector<int> &indices,
                     std::vector<double> &distance2) const;

private:
    /// \brief Sets the KDTree data from the data provided by the other methods.
    /// Internal method that sets all the members of KDTree by data provided by
    /// features, geometry, etc.
    bool SetRawData(const Eigen::Map<const Eigen::MatrixXd> &data);

protected:
    using KDTree_t = nanoflann::KDTreeEigenMatrixAdaptor<const Eigen::MatrixXd,
                                                         -1,
                                                         nanoflann::metric_L2,
                                                         false>;

    Eigen::MatrixXd data_;
    std::unique_ptr<KDTree_t> nanoflann_index_;
};

}  // namespace geometry
}  // namespace hymson3d

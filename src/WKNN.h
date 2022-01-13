#ifndef WKNND_H
#define WKNND_H

#include "util.h"


template<typename T>
struct WKNN {
public:
    WKNN(const Eigen::Map<Eigen::MatrixXd> data, bool buildtree=true);
    // WKNN(const Eigen::MatrixXd data, bool buildtree=true);

    ~WKNN() { delete_tree(); }

    void build_tree(typename NearestNeighbourSearch<T>::SearchType treetype=NearestNeighbourSearch<T>::KDTREE_LINEAR_HEAP);

    void delete_tree();

    arma::uvec query(Eigen::Map<Eigen::MatrixXd > query, const int k, const double eps=0.0, const double radius=0.0);
    // List query(const Eigen::Map<Eigen::MatrixXd > query, const int k, const double eps=0.0, const double radius=0.0);
    // List query(const Eigen::Matrix< T, Dynamic, Dynamic> query, const int k, const double eps=0.0, const double radius=0.0);

    arma::uvec queryWKNN(const WKNN& query, const int k, const double eps=0.0, const double radius=0.0);

    arma::uvec queryT(const Eigen::Matrix<T, Dynamic, Dynamic>& queryT, const int k, const double eps=0.0, const double radius=0.0);
    // List queryT(const Eigen::Matrix<T, Dynamic, Dynamic>& queryT, const int k, const double eps=0.0, const double radius=0.0);

    Eigen::MatrixXd getPoints();

private:
    Eigen::Matrix<T, Dynamic, Dynamic> data_pts;
    // Eigen::MatrixXd data_pts;
    NearestNeighbourSearch<T>* tree;
};

typedef WKNN<double> WKNND;
typedef WKNN<float> WKNNF;

#endif

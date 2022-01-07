#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include "nabo.h"
#include "WKNN.h"
#include "util.h"

// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace Nabo;
using namespace Eigen;


// Function to convert eigen data to arma
arma::mat cast_arma(Eigen::MatrixXd& eigen_A) {
    arma::mat arma_B = arma::mat(eigen_A.data(), eigen_A.rows(), eigen_A.cols(),
                                 false, false);

    return arma_B;
}

// arma::imat cast_arma_int(Eigen::MatrixXi& eigen_A) {
//     arma::imat arma_B = arma::imat(eigen_A.data(), eigen_A.rows(), eigen_A.cols(),
//                                  true, false);
//     // arma::umat arma_C = conv_to<arma::umat>::from(arma_B);
//
//     return arma_B;
// }

template <typename T>
WKNN<T>::WKNN(const Eigen::Map<Eigen::MatrixXd> data, bool buildtree) : tree(0) {
    data_pts = data.template cast<T>().transpose();
    if(buildtree) build_tree();
}

// template <typename T>
// WKNN<T>::WKNN(const Eigen::MatrixXd data, bool buildtree) : tree(0) {
//   data_pts = data.template cast<T>().transpose();
//   if(buildtree) build_tree();
// }

template <typename T>
void WKNN<T>::build_tree(typename NearestNeighbourSearch<T>::SearchType treetype) {
    if(tree==0) {
        tree = NearestNeighbourSearch<T>::create(data_pts, data_pts.rows(), treetype);
    }
}

template <typename T>
void WKNN<T>::delete_tree() {
    if(tree!=0) {
        delete tree;
        tree=0;
    }
}



template <typename T>
arma::uvec WKNN<T>::query(Eigen::Map<Eigen::MatrixXd> query, const int k, const double eps, const double radius) {
    return queryT(query.template cast<T>().transpose(), k, eps, radius);
}

// template <typename T>
// List WKNN<T>::query(const Eigen::Map<Eigen::MatrixXd> query, const int k, const double eps, const double radius) {
//   return queryT(query.template cast<T>().transpose(), k, eps, radius);
// }

// template <typename T>
// List WKNN<T>::query(const Eigen::Matrix< T, Dynamic, Dynamic> query, const int k, const double eps, const double radius) {
//   return queryT(query.template cast<T>().transpose(), k, eps, radius);
// }


template <typename T>
arma::uvec WKNN<T>::queryWKNN(const WKNN& query, const int k, const double eps, const double radius) {
    return queryT(query.data_pts, k, eps, radius);
}

template <typename T>
arma::uvec WKNN<T>::queryT(const Eigen::Matrix<T, Dynamic, Dynamic>& queryT, const int k, const double eps, const double radius) {
    MatrixXi indices(k, queryT.cols());
    Eigen::Matrix<T, Dynamic, Dynamic> dists2(k, queryT.cols());


    // build tree if required
    build_tree();
    tree->knn(queryT, indices, dists2, k, eps, NearestNeighbourSearch<T>::SORT_RESULTS | NearestNeighbourSearch<T>::ALLOW_SELF_MATCH,
              radius==0.0?std::numeric_limits<T>::infinity():radius);

    // MatrixXd indices_d = indices.cast<double>();
    // std::cout << "indices from eigen: " <<  indices << std::endl;;
    // std::cout << idx_arma << std::endl;

    // Can we use RMatrix here with wrap?
    // RMatrix<double> idx_wrap  = wrap(indices);
    // IntegerMatrix idx_rcpp = as<IntegerMatrix>(idx_wrap);
    // std::cout << "wrapped indices: " <<  idx_rcpp << std::endl;
    // arma::mat idx_arma = as<arma::mat>(idx_wrap);

    vector<int> vec_idx(indices.data(), indices.data() + indices.rows() * indices.cols());
    arma::uvec idx_arma = conv_to<arma::uvec>::from(vec_idx);
    // arma::mat idx_arma = cast_arma(indices);

    return idx_arma;
}



// Explicit template instantiation for linker
template struct WKNN<double>;

// Explicit template instantiation for linker
template struct WKNN<float>;

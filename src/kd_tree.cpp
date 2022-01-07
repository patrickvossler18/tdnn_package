#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include "util.h"
#include "nabo.h"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(BH)]]
using namespace Rcpp;
using namespace Nabo;
using namespace Eigen;

#include "WKNN.h"


// List knn_generic(int st, const Eigen::Map<Eigen::MatrixXd> data, const Eigen::Map<Eigen::MatrixXd> query,
//                  const int k, const double eps, const double radius) {
//
//     // create WKNND object but don't build tree
//     WKNND tree = WKNND(data, false);
//
//     // establish search type
//     Nabo::NNSearchD::SearchType nabo_st;
//     if(st==1L){
//         // incoming st==1 implies auto, so choose according to value of k
//         nabo_st = k < 30 ? NNSearchD::KDTREE_LINEAR_HEAP : NNSearchD::KDTREE_TREE_HEAP;
//     } else {
//         // if we receive 2L from R => BRUTE_FORCE etc
//         nabo_st = static_cast<Nabo::NNSearchD::SearchType>(st-2L);
//     }
//
//     // build tree using appropriate search type
//     tree.build_tree(nabo_st);
//
//     return tree.query(query, k, eps, radius);
// }


WKNND create_tree(arma::mat& data){

    // Eigen::MatrixXd data_eigen = cast_eigen(data);
    Eigen::Map<Eigen::MatrixXd> data_eigen(data.memptr(), data.n_rows, data.n_cols);
    // create WKNND object but don't build tree

    WKNND tree = WKNND(data_eigen, false);

    // since we just want 1-NN so linear heap
    tree.build_tree(NNSearchD::KDTREE_LINEAR_HEAP);

    return(tree);
}



arma::uvec query_tree(WKNND& tree, arma::mat query,
                const int k){
    // we're only going to use exact NN so set eps and radius to 0.0
    double eps = 0.0;
    double radius = 0.0;

    // Eigen::MatrixXd query_eigen = cast_eigen(query);
    Eigen::Map<Eigen::MatrixXd> query_eigen (query.memptr(), query.n_rows, query.n_cols);

    return tree.query(query_eigen, k, eps, radius);
}

// [[Rcpp::export]]
arma::vec get_1nn_reg(arma::mat X, arma::mat X_test, arma::mat Y, int k){
    WKNND kd_tree = create_tree(X);

    // this gives us list with nn.dist and nn.idx which are both arma matrices
    arma::uvec query_result = query_tree(kd_tree, X_test, k);
    // arma::uvec nn_idx_vec = conv_to<arma::uvec>::from(query_result);
    // List query_result = query_tree(kd_tree, X_test, k);
    // std::cout << "made query result" << std::endl;
    // arma::imat arma_mat = as<arma::imat>(query_result);
    // std::cout << "query_result: " << query_result << std::endl;
    // // we want to get the rows of Y from nn.idx
    // // std::cout << "trying to convert query_result to imat" << std::endl;
    // arma::uvec nn_idx_mat = as<arma::uvec>(query_result);
    // // std::cout << "success" << std::endl;
    // arma::uvec nn_idx_vec = conv_to<arma::uvec>::from(nn_idx_mat);
    arma::mat Y_nn_idx = Y.rows(query_result);
    // std::cout << "Y_nn_idx: " << Y_nn_idx << std::endl;
    // take row means
    arma::vec preds = rowMeans_arma(Y_nn_idx);
    // std::cout << "preds: " << preds << std::endl;
    // return(preds);

    // std::cout << "nn_idx_vec: " << nn_idx_vec << std::endl;
    // arma::vec preds_test(X_test.n_rows);
    // preds_test.fill(1);
    // return preds_test;

    return(preds);
}

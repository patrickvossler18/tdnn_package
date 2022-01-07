#ifndef KD_TREE_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define KD_TREE_H

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

WKNND create_tree(arma::mat& data);
arma::uvec query_tree(WKNND& tree, arma::mat query, const int k);
arma::vec get_1nn_reg(arma::mat X, arma::mat X_test, arma::mat Y, int k);


#endif

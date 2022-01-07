#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include "util.h"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace Eigen;

// Function to convert arma data to eigen
Eigen::MatrixXd cast_eigen(arma::mat& data) {
    Eigen::MatrixXd eigen_data = Eigen::Map<Eigen::MatrixXd>(data.memptr(),
                                                             data.n_rows,
                                                             data.n_cols);

    return eigen_data;
}

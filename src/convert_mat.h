#ifndef CONVERT_MAT_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define CONVERT_MAT_H

#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include "util.h"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
using namespace Eigen;

Eigen::MatrixXd cast_eigen(const arma::mat& data);
// arma::mat cast_arma(Eigen::MatrixXd& eigen_A);

#endif

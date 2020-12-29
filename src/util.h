#ifndef UTIL_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define UTIL_H

#include <math.h>
#include <RcppArmadillo.h>
#include <RcppParallel.h>

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace RcppParallel;
using namespace arma;
using namespace std;



double nChoosek(double n, double k);
NumericVector seq_cpp(double lo, double hi);
arma::mat matrix_subset_logical(const arma::mat & x, const arma::vec & y, int mrgn=1);

Rcpp::NumericMatrix matrix_subset_idx_rcpp( Rcpp::NumericMatrix x, Rcpp::IntegerVector y);
arma::uvec seq_int(long int a, long int b);

arma::mat matrix_subset_idx(const arma::mat& x, const arma::uvec& y);

#endif

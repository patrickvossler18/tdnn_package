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
arma::mat matrix_subset_logical(arma::mat x, arma::vec y, int mrgn=1);

#endif

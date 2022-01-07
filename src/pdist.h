#ifndef PDIST_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define PDIST_H

#include <RcppArmadillo.h>

using namespace Rcpp;

arma::mat fastPdist(const arma::mat& Ar, const arma::mat& Br);
#endif

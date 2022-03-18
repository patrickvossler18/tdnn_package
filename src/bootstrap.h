#ifndef BOOTSTRAP_H // To make sure you don't declare the function more than once by including the header multiple times.
#define BOOTSTRAP_H

#include "util.h"

NumericMatrix bootstrap_cpp_mt(const arma::mat &X,
                               const arma::mat &Y,
                               const arma::mat &X_test,
                               const arma::vec s_1,
                               const arma::vec c,
                               const double n_prop,
                               const int B,
                               Nullable<NumericVector> W0_ = R_NilValue);

#endif

#ifndef TUNE_PARAMS_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define TUNE_PARAMS_H

#include "util.h"
#include "dnn_parallel.h"
#include "kd_tree.h"


List tune_params(const arma::mat& X,
                 const arma::mat& Y,
                 const arma::mat& X_test,
                 const arma::mat& param_mat,
                 int B,
                 double n_prop,
                 bool verbose = false,
                 Nullable<NumericVector> W0_ = R_NilValue);

#endif

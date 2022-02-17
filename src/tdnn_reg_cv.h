#ifndef TDNN_REG_CV_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define TDNN_REG_CV_H


#include "util.h"
#include "dnn_parallel.h"
#include "tune_params.h"
#include "bootstrap.h"

List tdnn_reg_cv_cpp(
        const arma::mat& X,
        const arma::mat& Y,
        const arma::mat& X_test,
        const arma::mat& param_mat,
        int B,
        double n_prop,
        bool verbose = false,
        Nullable<NumericVector> W0_ = R_NilValue
);

#endif

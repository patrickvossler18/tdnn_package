#ifndef DNN_PARALLEL_H // To make sure you don't declare the function more than once by including the header multiple times.
#define DNN_PARALLEL_H

#include "util.h"
#include "kd_tree.h"

NumericVector tuning(arma::mat X, arma::vec Y,
                     arma::mat X_test, double c,
                     double n_prop,
                     double C_s_2,
                     Nullable<NumericVector> W0_ = R_NilValue);

arma::vec tdnn(arma::mat X, arma::vec Y, arma::mat X_test,
               double c,
               double n_prop,
               int s_1_val,
               int s_2_val,
               Nullable<NumericVector> W0_ = R_NilValue);

#endif

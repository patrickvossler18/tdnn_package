#ifndef TDNN_ORD_Y_H // To make sure you don't declare the function more than once by including the header multiple times.
#define TDNN_ORD_Y_H

#include "util.h"
#include "bootstrap.h"

arma::vec tuning_ord_Y_st(const arma::mat &ordered_Y, int n, int p, int n_obs,
                          int s_1_max,
                          double c,
                          double n_prop);

arma::mat make_B_NN_estimates_st(
    const arma::mat &X,
    const arma::mat &Y,
    const arma::vec &X_test_i,
    const arma::uvec &top_B,
    const arma::vec c_vec,
    const arma::vec s_1_vec_tmp,
    double n_prop = 0.5, int B_NN = 20,
    double scale_p = 1,
    bool debug = false);

double tdnn_ord_y_st(arma::mat ordered_Y_i,
                     arma::vec s_1,
                     int n,
                     int p,
                     double c,
                     double n_prop);

#endif

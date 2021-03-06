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

NumericMatrix bootstrap_trt_effect_cpp_mt(const arma::mat &X,
                                          const arma::mat &Y,
                                          const arma::vec &W,
                                          const arma::mat &X_test,
                                          const arma::vec &s_choice_trt,
                                          const arma::vec &s_choice_ctl,
                                          const arma::vec &c_trt,
                                          const arma::vec &c_ctl,
                                          const double n_prop,
                                          const int B,
                                          Nullable<NumericVector> W0_ = R_NilValue);

NumericMatrix bootstrap_trt_effect_cpp_thread(const arma::mat &X_ctl,
                                              const arma::mat &Y_ctl,
                                              const arma::mat &X_trt,
                                              const arma::mat &Y_trt,
                                              const arma::mat &X_test,
                                              const arma::vec &s_choice_trt,
                                              const arma::vec &s_choice_ctl,
                                              const arma::vec &c_trt,
                                              const arma::vec &c_ctl,
                                              const double n_prop,
                                              const int B,
                                              int num_threads,
                                              bool verbose = false);

    NumericMatrix bootstrap_cpp_thread(const arma::mat &X,
                                       const arma::mat &Y,
                                       const arma::mat &X_test,
                                       const arma::vec s_1,
                                       const arma::vec c,
                                       const double n_prop,
                                       const int B,
                                       int num_threads,
                                       Nullable<NumericVector> W0_ = R_NilValue,
                                       bool verbose = false);

NumericMatrix bootstrap_dnn_cpp_thread(const arma::mat &X,
                                       const arma::mat &Y,
                                       const arma::mat &X_test,
                                       const arma::vec s_1,
                                       const double n_prop,
                                       const int B,
                                       int num_threads,
                                       Nullable<NumericVector> W0_ = R_NilValue,
                                       bool verbose = false);

#endif

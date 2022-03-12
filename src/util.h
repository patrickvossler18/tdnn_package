#ifndef UTIL_H // To make sure you don't declare the function more than once by including the header multiple times.
#define UTIL_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>
#include <RcppParallel.h>

#include <math.h>
#include <tuple>

// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;
using namespace RcppParallel;
using namespace arma;
using namespace std;

double nChoosek(double n, double k);
NumericVector seq_cpp(double lo, double hi);
arma::mat matrix_subset_logical(const arma::mat &x, const arma::vec &y, int mrgn = 1);

Rcpp::NumericMatrix matrix_subset_idx_rcpp(Rcpp::NumericMatrix x, Rcpp::IntegerVector y);
arma::uvec seq_int(long int a, long int b);

arma::mat matrix_subset_idx(const arma::mat &x, const arma::uvec &y);

arma::mat matrix_row_subset_idx(const arma::mat &x, const arma::uvec &y);

arma::vec vector_subset_idx(const arma::vec &x, const arma::uvec &y);

arma::mat weight_mat_lfac(int n, const arma::vec &ord, const arma::vec &s_vec);

arma::mat weight_mat_lfac_s_2_filter(int n, const arma::vec &ord, const arma::vec &s_vec, double n_prop, bool is_s_2);

double round_modified(const double &x);
arma::vec round_modified_vec(const arma::vec &x);
arma::vec arma_round(const arma::vec &x);

arma::vec rowMeans_arma(const arma::mat &x);

arma::vec colMeans_arma(const arma::mat &x);

arma::vec colSums_arma(const arma::mat &x);

arma::mat rowSums_arma(const arma::mat &x);

arma::vec colVar_arma(const arma::mat &x);

arma::vec rowVar_arma(const arma::mat &x);

arma::vec select_mat_elements(const arma::mat &x, const arma::uvec &row_idx, const arma::uvec &col_idx);

arma::uvec r_like_order(const arma::vec &x, const arma::vec &y);

arma::uvec sample_replace_index(const int &size);

arma::mat calc_dist_mat_cpp(const arma::mat &A, const arma::mat &B);

#endif

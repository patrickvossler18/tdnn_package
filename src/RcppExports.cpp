// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// matrix_subset_logical
arma::mat matrix_subset_logical(arma::mat x, arma::vec y);
RcppExport SEXP _tdnn_matrix_subset_logical(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(matrix_subset_logical(x, y));
    return rcpp_result_gen;
END_RCPP
}
// dnn
NumericVector dnn(NumericMatrix X, NumericVector Y, NumericMatrix X_test, double n, double p, double s_size);
RcppExport SEXP _tdnn_dnn(SEXP XSEXP, SEXP YSEXP, SEXP X_testSEXP, SEXP nSEXP, SEXP pSEXP, SEXP s_sizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Y(YSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< double >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type p(pSEXP);
    Rcpp::traits::input_parameter< double >::type s_size(s_sizeSEXP);
    rcpp_result_gen = Rcpp::wrap(dnn(X, Y, X_test, n, p, s_size));
    return rcpp_result_gen;
END_RCPP
}
// de_dnn
List de_dnn(arma::mat X, NumericVector Y, arma::mat X_test, double s_size, double bc_p, Nullable<NumericVector> W0_);
RcppExport SEXP _tdnn_de_dnn(SEXP XSEXP, SEXP YSEXP, SEXP X_testSEXP, SEXP s_sizeSEXP, SEXP bc_pSEXP, SEXP W0_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Y(YSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< double >::type s_size(s_sizeSEXP);
    Rcpp::traits::input_parameter< double >::type bc_p(bc_pSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type W0_(W0_SEXP);
    rcpp_result_gen = Rcpp::wrap(de_dnn(X, Y, X_test, s_size, bc_p, W0_));
    return rcpp_result_gen;
END_RCPP
}
// best_s
NumericVector best_s(arma::mat estimate_matrix);
RcppExport SEXP _tdnn_best_s(SEXP estimate_matrixSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type estimate_matrix(estimate_matrixSEXP);
    rcpp_result_gen = Rcpp::wrap(best_s(estimate_matrix));
    return rcpp_result_gen;
END_RCPP
}
// tuning
NumericVector tuning(NumericVector s_seq, NumericMatrix X, NumericVector Y, NumericMatrix X_test, double bc_p, Nullable<NumericVector> W0_);
RcppExport SEXP _tdnn_tuning(SEXP s_seqSEXP, SEXP XSEXP, SEXP YSEXP, SEXP X_testSEXP, SEXP bc_pSEXP, SEXP W0_SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type s_seq(s_seqSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Y(YSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X_test(X_testSEXP);
    Rcpp::traits::input_parameter< double >::type bc_p(bc_pSEXP);
    Rcpp::traits::input_parameter< Nullable<NumericVector> >::type W0_(W0_SEXP);
    rcpp_result_gen = Rcpp::wrap(tuning(s_seq, X, Y, X_test, bc_p, W0_));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_tdnn_matrix_subset_logical", (DL_FUNC) &_tdnn_matrix_subset_logical, 2},
    {"_tdnn_dnn", (DL_FUNC) &_tdnn_dnn, 6},
    {"_tdnn_de_dnn", (DL_FUNC) &_tdnn_de_dnn, 6},
    {"_tdnn_best_s", (DL_FUNC) &_tdnn_best_s, 1},
    {"_tdnn_tuning", (DL_FUNC) &_tdnn_tuning, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_tdnn(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
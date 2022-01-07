#include "util.h"

// [[Rcpp::export]]
arma::mat fastPdist(const arma::mat& Ar, const arma::mat& Br) {
    // arma::mat A = arma::mat(Ar.begin(), m, k, false);
    // arma::mat B = arma::mat(Br.begin(), n, k, false);

    // Probably unnecessary to have so many check interrupts, but
    // this matrix scales up quickly
    arma::colvec An =  sum(square(Ar),1);
    Rcpp::checkUserInterrupt();
    arma::colvec Bn =  sum(square(Br),1);
    Rcpp::checkUserInterrupt();
    arma::mat C = -2 * (Ar * Br.t());
    Rcpp::checkUserInterrupt();
    C.each_col() += An;
    Rcpp::checkUserInterrupt();
    C.each_row() += Bn.t();
    Rcpp::checkUserInterrupt();

    // return wrap(sqrt(C));
    // return wrap(C);
    return C;
}

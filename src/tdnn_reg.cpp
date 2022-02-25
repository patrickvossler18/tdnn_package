#include "tdnn_reg.h"

// [[Rcpp::export]]
List tdnn_reg_cpp(
    const arma::mat &X,
    const arma::mat &Y,
    const arma::mat &X_test,
    double c,
    double n_prop,
    int s_1_val,
    int s_2_val,
    int bootstrap_iter,
    bool estimate_variance,
    bool verbose,
    Nullable<NumericVector> W0_)
{
    NumericVector W0;
    if (W0_.isNotNull())
    {
        W0 = W0_;
    }
    else
    {
        W0 = rep(1, X.n_cols);
    }

    int d = sum(W0);
    // int s_2_val = std::ceil(int(round_modified(exp(M * log(X.n_rows) * (double(d) / (double(d) + 8))))));
    // int s_1_val = std::ceil(int(round_modified(s_2_val * pow(c, double(d) / 2))));

    arma::vec deDNN_pred;

    deDNN_pred = tdnn(X, Y, X_test, c, n_prop, s_1_val, s_2_val, W0);

    if (estimate_variance)
    {
        NumericMatrix bstrap_estimates = bootstrap_cpp_mt(X, Y, X_test,
                                                          c, n_prop, s_1_val,
                                                          s_2_val, bootstrap_iter, W0);
        // need to apply variance over columns
        arma::vec variance = rowVar_arma(as<arma::mat>(bstrap_estimates));
        return (List::create(Named("estimates") = deDNN_pred,
                             Named("variance") = variance,
                             Named("s_1") = s_1_val,
                             Named("s_2") = s_2_val,
                             Named("c") = c));
    }
    else
    {
        return (List::create(Named("estimates") = deDNN_pred,
                             Named("s_1") = s_1_val,
                             Named("s_2") = s_2_val,
                             Named("c") = c));
    }
}

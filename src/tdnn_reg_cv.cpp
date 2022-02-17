#include "tdnn_reg_cv.h"

List tdnn_reg_cv_cpp(
        const arma::mat& X,
        const arma::mat& Y,
        const arma::mat& X_test,
        const arma::mat& param_mat,
        int B,
        double n_prop,
        bool verbose,
        Nullable<NumericVector> W0_
){
    NumericVector W0;
    if (W0_.isNotNull()){
        W0 = W0_;
    } else{
        W0 = rep(1,X.n_cols);
    }
    if(verbose){
        Rcout << "starting tuning" << std::endl;
    }

    // first do LOO CV to get tuned c and M
    List tuned_params = tune_params(X, Y, X_test, param_mat, B, n_prop, verbose, W0);

    double c = tuned_params['c'];
    double M = tuned_params['M'];
    // then use the tuned values to get s_1 for the full data
    NumericVector s_sizes = tuning(X, Y, X_test, c, n_prop, M, W0);

    if(verbose){
        Rcout << "past tuning" << std::endl;
    }

    arma::vec deDNN_pred;

    deDNN_pred = tdnn(X, Y, X_test, s_sizes,
                      s_sizes + 1,
                      c, n_prop, M, W0);
    return(List::create( Named("estimates") = deDNN_pred,
                         Named("s") = s_sizes,
                         Named("c") = c,
                         Named("M") = M));

}

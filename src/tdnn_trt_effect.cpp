#include "tdnn_trt_effect.h"

std::tuple<arma::mat, arma::uvec> make_ord_Y(arma::mat Y, arma::vec EuDis_col_i, int B_NN)
{
    arma::vec noise(Y.n_rows);
    double noise_val = arma::randn<double>();
    noise.fill(noise_val);
    arma::vec eu_dist_col = EuDis_col_i;
    // sort each column and get the indices of the top B_NN
    arma::uvec sorted_idx = sort_index(eu_dist_col);
    arma::uvec top_B = sorted_idx.head(B_NN);
    arma::uvec idx_tmp = r_like_order(eu_dist_col, noise);
    arma::mat ordered_Y = conv_to<arma::mat>::from(Y).rows(idx_tmp);
    return std::make_tuple(ordered_Y, top_B);
}

arma::vec min_mse_and_s_1(arma::vec mse_vec, arma::vec s_1_vec)
{
    double min_val = as_scalar(mse_vec.min());
    uword min_idx = index_min(mse_vec);
    double choose_s1 = as_scalar(s_1_vec(min_idx));
    return {min_val, choose_s1};
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppThread)]]
// [[Rcpp::export]]
Rcpp::List tune_treatment_effect_thread(
    arma::mat X,
    arma::vec Y,
    arma::vec W,
    arma::mat X_test,
    Nullable<NumericVector> W0_,
    arma::vec c,
    int B_NN = 20,
    double scale_p = 1,
    double n_prop = 0.5,
    bool estimate_variance = false,
    int bootstrap_iter = 500,
    bool verbose = false,
    int num_threads = 1)
{
    if (W0_.isNotNull())
    {
        NumericVector W0(W0_);
        // Now we need to filter X and X_test to only contain these columns
        X = matrix_subset_logical(X, as<arma::vec>(W0));
        X_test = matrix_subset_logical(X_test, as<arma::vec>(W0));
        // d = sum(W0);
    }

    // Infer n and p from our data after we've filtered for relevant features
    int n = X.n_rows;
    int p = X.n_cols;
    int n_test = X_test.n_rows;

    // Need to filter EuDis for treatment and control groups by W
    arma::uvec trt_idx = find(W == 1);
    arma::uvec ctl_idx = find(W == 0);

    arma::mat X_trt = X.rows(trt_idx);
    arma::mat Y_trt = Y.rows(trt_idx);

    arma::mat X_ctl = X.rows(ctl_idx);
    arma::mat Y_ctl = Y.rows(ctl_idx);

    int n_ctl = X_ctl.n_rows;
    int p_ctl = X_ctl.n_cols;

    int n_trt = X_trt.n_rows;
    int p_trt = X_trt.n_cols;

    if (verbose)
    {
        Rcout << "n: " << n << std::endl;
        Rcout << "p: " << p << std::endl;

        Rcout << "n ctl: " << n_ctl << std::endl;
        Rcout << "p ctl: " << p_ctl << std::endl;

        Rcout << "n trt: " << n_trt << std::endl;
        Rcout << "p trt: " << p_trt << std::endl;
    }

    // calculate EuDist for all test observations
    arma::mat EuDis_trt = calc_dist_mat_cpp(X_trt, X_test);
    arma::mat EuDis_ctl = calc_dist_mat_cpp(X_ctl, X_test);

    NumericVector tuned_estimate_ctl(n_test);
    NumericVector s_1_B_NN_ctl(n_test);
    NumericVector c_B_NN_ctl(n_test);

    NumericVector tuned_estimate_trt(n_test);
    NumericVector s_1_B_NN_trt(n_test);
    NumericVector c_B_NN_trt(n_test);
    // NumericMatrix curve_estimate(n_test, c.n_elem);
    // NumericMatrix s_1_mse_curve(n_test, c.n_elem);

    if (verbose)
    {
        Rcout << "Estimating..." << std::endl;
    }
    RcppThread::ProgressBar bar(X_test.n_rows, 1);
    RcppThread::parallelFor(
        0, X_test.n_rows, [&X_ctl, &Y_ctl, &X_trt, &Y_trt, &X_test, &EuDis_trt, &EuDis_ctl, &c, &tuned_estimate_ctl, &s_1_B_NN_ctl, &c_B_NN_ctl, &tuned_estimate_trt, &s_1_B_NN_trt, &c_B_NN_trt, &B_NN, &n_ctl, &p_ctl, &n_trt, &p_trt, &scale_p, &n_prop, &verbose, &bar](int i)
        {
        arma::mat vary_c_results_ctl(c.n_elem, 3);
        arma::mat vary_c_results_trt(c.n_elem, 3);

        arma::vec X_test_i = X_test.row(i).as_col();
        arma::mat X_test_i_mat = conv_to<arma::mat>::from(X_test.row(i));
        arma::mat ordered_Y_ctl;
        arma::uvec top_B_ctl;
        arma::mat ordered_Y_trt;
        arma::uvec top_B_trt;

        std::tie(ordered_Y_ctl, top_B_ctl) = make_ord_Y(Y_ctl, EuDis_ctl.col(i), B_NN);
        std::tie(ordered_Y_trt, top_B_trt) = make_ord_Y(Y_trt, EuDis_trt.col(i), B_NN);

        for (int j = 0; j < c.n_elem; j++)
        {
            double c_val = c(j);
            arma::vec c_val_vec = {c_val};
            // double estimate_curve = 0;
            double max_s_1_ctl = floor((n_ctl - 5) / c_val) - 1;
            double max_s_1_trt = floor((n_trt - 5) / c_val) - 1;

            arma::vec mse_curve_s_ctl = tuning_ord_Y_st(ordered_Y_ctl, n_ctl, p_ctl, 1, double(max_s_1_ctl), c_val, n_prop);
            arma::vec mse_curve_s_trt = tuning_ord_Y_st(ordered_Y_trt, n_trt, p_trt, 1, double(max_s_1_trt), c_val, n_prop);
            double s_tmp_ctl = arma::as_scalar(mse_curve_s_ctl);
            double s_tmp_trt = arma::as_scalar(mse_curve_s_trt);

            arma::vec s_1_vec_tmp_ctl = seq_cpp_arma(s_tmp_ctl, 2 * s_tmp_ctl);
            arma::vec s_1_vec_tmp_trt = seq_cpp_arma(s_tmp_trt, 2 * s_tmp_trt);
            // Rcout << "s_1_vec_tmp: " << s_1_vec_tmp << std::endl;
            arma::mat B_NN_estimates_ctl = make_B_NN_estimates_st(X_ctl, Y_ctl, X_test_i, top_B_ctl, c_val_vec,
                                                                s_1_vec_tmp_ctl, n_prop, B_NN, scale_p, false);

            arma::mat B_NN_estimates_trt = make_B_NN_estimates_st(X_trt, Y_trt, X_test_i, top_B_trt, c_val_vec,
                                                                s_1_vec_tmp_trt, n_prop, B_NN, scale_p, false);

            arma::mat best_s_1_c_ctl(c_val_vec.n_elem, 3);
            arma::mat best_s_1_c_trt(c_val_vec.n_elem, 3);
            // loop over rows of B_NN_estimates matrix and get entry with smallest value
            for (int k = 0; k < B_NN_estimates_ctl.n_rows; k++)
            {
                arma::vec tuned_mse_ctl = B_NN_estimates_ctl.row(k).as_col();
                arma::vec tuned_mse_trt = B_NN_estimates_trt.row(k).as_col();
                arma::vec min_mse_s_1_vec_ctl = min_mse_and_s_1(tuned_mse_ctl, s_1_vec_tmp_ctl);
                arma::vec min_mse_s_1_vec_trt = min_mse_and_s_1(tuned_mse_trt, s_1_vec_tmp_trt);
                best_s_1_c_ctl.row(k) = {c_val, min_mse_s_1_vec_ctl[1], min_mse_s_1_vec_ctl[0]};
                best_s_1_c_trt.row(k) = {c_val, min_mse_s_1_vec_trt[1], min_mse_s_1_vec_trt[0]};
            }
            // get index of row with minimum tuned mse
            uword best_row_ctl = best_s_1_c_ctl.col(2).index_min();
            uword best_row_trt = best_s_1_c_trt.col(2).index_min();
            arma::rowvec best_row_tmp_ctl = best_s_1_c_ctl.row(best_row_ctl);
            arma::rowvec best_row_tmp_trt = best_s_1_c_trt.row(best_row_trt);
            vary_c_results_ctl.row(j) = best_row_tmp_ctl;
            vary_c_results_trt.row(j) = best_row_tmp_trt;
        }
        // Rcout << "vary_c_results: " << vary_c_results << std::endl;
        // now we have the min values for each c value, need min of mins
        uword best_c_row_ctl = vary_c_results_ctl.col(2).index_min();
        arma::rowvec best_c_row_tmp_ctl = vary_c_results_ctl.row(best_c_row_ctl);
        uword best_c_row_trt = vary_c_results_trt.col(2).index_min();
        arma::rowvec best_c_row_tmp_trt = vary_c_results_trt.row(best_c_row_trt);
        // Rcout << "best_c_row idx :" << best_c_row << std::endl;
        // Rcout << "best_c_row_tmp: " << best_c_row_tmp << std::endl;
        // now get the estimate using these values
        double best_c_ctl = best_c_row_tmp_ctl(0);
        double best_s_1_ctl = best_c_row_tmp_ctl(1);
        arma::vec best_s_1_vec_ctl = {best_s_1_ctl};

        double best_c_trt = best_c_row_tmp_trt(0);
        double best_s_1_trt = best_c_row_tmp_trt(1);
        arma::vec best_s_1_vec_trt = {best_s_1_trt};


        tuned_estimate_ctl[i] = tdnn_ord_y_st(ordered_Y_ctl, best_s_1_vec_ctl, ordered_Y_ctl.n_rows, p_ctl, best_c_ctl, n_prop);
        tuned_estimate_trt[i] = tdnn_ord_y_st(ordered_Y_trt, best_s_1_vec_trt, ordered_Y_trt.n_rows, p_trt, best_c_trt, n_prop);
        s_1_B_NN_ctl[i] = best_s_1_ctl;
        c_B_NN_ctl[i] = best_c_ctl;
        s_1_B_NN_trt[i] = best_s_1_trt;
        c_B_NN_trt[i] = best_c_trt;
        if(verbose){
            bar++;
        } },
        num_threads);

    NumericVector trt_effect_estimate = tuned_estimate_trt - tuned_estimate_ctl;

    if (estimate_variance)
    {
        if (verbose)
        {
            Rcout << "Running bootstrap..." << std::endl;
        }
        NumericMatrix bstrap_estimates = bootstrap_trt_effect_cpp_thread(X_ctl, Y_ctl, X_trt, Y_trt, X_test, s_1_B_NN_trt, s_1_B_NN_ctl, c_B_NN_trt, c_B_NN_ctl, n_prop, bootstrap_iter, num_threads, R_NilValue);
        // need to apply variance over columns
        arma::vec variance = rowVar_arma(as<arma::mat>(bstrap_estimates));
        return Rcpp::List::create(
            Named("treatment_effect") = trt_effect_estimate,
            Named("estimate_trt") = tuned_estimate_trt,
            Named("estimate_ctl") = tuned_estimate_ctl,
            Named("s_1_B_NN_ctl") = s_1_B_NN_ctl,
            Named("c_B_NN_ctl") = c_B_NN_ctl,
            Named("s_1_B_NN_trt") = s_1_B_NN_trt,
            Named("c_B_NN_trt") = c_B_NN_trt,
            Named("variance") = variance);
    }
    else
    {
        return Rcpp::List::create(
            Named("treatment_effect") = trt_effect_estimate,
            Named("estimate_trt") = tuned_estimate_trt,
            Named("estimate_ctl") = tuned_estimate_ctl,
            Named("s_1_B_NN_ctl") = s_1_B_NN_ctl,
            Named("c_B_NN_ctl") = c_B_NN_ctl,
            Named("s_1_B_NN_trt") = s_1_B_NN_trt,
            Named("c_B_NN_trt") = c_B_NN_trt);
    }
}

#include "tune_dnn.h"

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double dnn_ord_y_st(arma::mat ordered_Y_i, arma::vec s_1, int n, int p, double n_prop)
{
    arma::vec ord_arma = seq_cpp_arma(1, n);

    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(n, ord_arma, s_1, n_prop, false);

    arma::rowvec ordered_Y_row = ordered_Y_i.as_row();
    arma::vec U_1_vec(ordered_Y_i.n_elem);

    // the weight matrix is # train obs x # test obs so we want to use
    // the ith column of the weight mat for the ith test observation
    // U_1_vec = reshape(ordered_Y, 1, n) * weight_mat_s_1.col(i);
    U_1_vec = ordered_Y_row * weight_mat_s_1.col(0);

    return arma::as_scalar(U_1_vec);
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
NumericVector dnn(arma::mat X, arma::vec Y, arma::mat X_test,
                  arma::vec s_sizes,
                  double n_prop = 0.5,
                  Nullable<NumericVector> W0_ = R_NilValue)
{
    // Handle case where W0 is not NULL:
    if (W0_.isNotNull())
    {
        NumericVector W0(W0_);
        // Now we need to filter X and X_test to only contain these columns
        X = matrix_subset_logical(X, as<arma::vec>(W0));
        X_test = matrix_subset_logical(X_test, as<arma::vec>(W0));
    }

    // Infer n and p from our data after we've filtered for relevant features
    int n = X.n_rows;
    int p = X.n_cols;

    NumericVector ord = seq_cpp(1, n);
    ord.attr("dim") = Dimension(n, 1);
    arma::vec ord_arma = as<arma::vec>(ord);

    arma::vec s_1 = s_sizes;

    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(n, ord_arma, s_1, n_prop, false);

    NumericVector estimates(X_test.n_rows);

    RcppThread::ProgressBar bar(X_test.n_rows, 1);
    RcppThread::parallelFor(0, X_test.n_rows,
                            [&X, &Y, &X_test, &weight_mat_s_1, &estimates, &n, &p, &bar](int i)
                            {
                                // arma::mat single_vec(int(n),1);
                                // single_vec.fill(1.0);
                                arma::mat all_cols(p, 1);
                                all_cols.fill(1.0);

                                // arma::mat all_rows;
                                arma::mat X_dis;
                                arma::mat EuDis;

                                arma::mat X_test_row = X_test.row(i);
                                // all_rows = single_vec * X_test_row;
                                arma::mat all_rows = arma::repmat(X_test_row, n, 1);

                                X_dis = X - all_rows;

                                EuDis = (pow(X_dis, 2)) * all_cols;
                                // Rcout << "EuDis: "<< EuDis << std::endl;
                                // arma::mat noise(int(n), 1);
                                // double noise_val = R::rnorm(0, 1);
                                // noise.fill(noise_val);
                                // arma::vec noise = arma::randn<vec>(int(n));
                                arma::vec noise(n);
                                double noise_val = arma::randn<double>();
                                noise.fill(noise_val);

                                arma::vec vec_eu_dis = conv_to<arma::vec>::from(EuDis);

                                arma::uvec index = r_like_order(vec_eu_dis, noise);

                                arma::vec ordered_Y;
                                arma::mat ordered_Y_vec = conv_to<arma::mat>::from(Y).rows(index);
                                ordered_Y = ordered_Y_vec;

                                arma::vec U_1_vec(ordered_Y.n_rows);

                                // the weight matrix is # train obs x # test obs so we want to use
                                // the ith column of the weight mat for the ith test observation
                                U_1_vec = reshape(ordered_Y, 1, n) * weight_mat_s_1.col(i);
                                estimates[i] = arma::as_scalar(U_1_vec);
                                bar++;
                            });
    return estimates;
}

// [[Rcpp::export]]
arma::vec dnn_B_NN_estimates(
    const arma::mat &X,
    const arma::mat &Y,
    const arma::vec &X_test_i,
    const arma::uvec &top_B,
    const arma::vec &s_seq,
    double n_prop = 0.5, int B_NN = 20,
    double scale_p = 1,
    bool debug = false)
{
    int n = X.n_rows;
    arma::uvec idx_vec = seq_int(0, n - 1);
    // arma::mat B_NN_estimates(B_NN, 5);
    arma::vec B_NN_estimates(s_seq.n_elem, fill::zeros);
    // now we get the B nearest neighbors validation estimates
    for (int j = 0; j < B_NN; j++)
    {
        arma::uvec val_idx = {top_B(j)};
        // Rcout << "top_B(j): " << top_B(j) << std::endl;
        arma::uvec train_idx = idx_vec.elem(find(idx_vec != int(top_B(j))));

        arma::mat X_train = X.rows(train_idx);
        arma::mat Y_train = Y.rows(train_idx);

        arma::mat X_val = X.rows(val_idx);
        // Rcout << "X_val: " << X_val << std::endl;
        arma::mat Y_val = Y.rows(val_idx);

        arma::mat EuDis_tmp = calc_dist_mat_cpp(X_train, X_val);
        arma::vec noise_tmp(X_train.n_rows);
        double noise_val_tmp = arma::randn<double>();
        noise_tmp.fill(noise_val_tmp);
        arma::uvec idx_tmp_val = r_like_order(EuDis_tmp.as_col(), noise_tmp);
        arma::mat ordered_Y_train = conv_to<arma::mat>::from(Y_train).rows(idx_tmp_val);

        // X_val should always be one observation and we've converted X_test_i
        // to be column format too so we will take the difference of them as columns
        double neighbor_weight = exp(
            -arma::sum(arma::pow((X_val.as_col() - X_test_i), 2)) / scale_p);
        double weighted_Y_val = as_scalar(Y_val * sqrt(neighbor_weight));
        for (int l = 0; l < s_seq.n_elem; l++)
        {
            arma::vec s_1_val = {s_seq(l)};
            double param_estimate = dnn_ord_y_st(ordered_Y_train, s_1_val, X_train.n_rows, X_train.n_cols, n_prop);
            double weighted_estimate = param_estimate * sqrt(neighbor_weight);
            double loss = pow((weighted_estimate - weighted_Y_val), 2);
            B_NN_estimates(l) += loss;
        }
    }
    // now calculate mean loss over the B_NN observations
    B_NN_estimates = B_NN_estimates / B_NN;
    return B_NN_estimates;
}

// [[Rcpp::export]]
Rcpp::List tune_dnn_no_dist_thread(
    arma::mat X,
    arma::vec Y,
    arma::mat X_test,
    arma::vec s_seq,
    Nullable<NumericVector> W0_,
    int B_NN = 20,
    double scale_p = 1,
    double n_prop = 0.5,
    bool estimate_variance = false,
    int bootstrap_iter = 500,
    bool verbose = false,
    bool debug = false,
    int num_threads = 1)
{
    /**
     * @brief Tune dnn for different s_1 values
     * This function calculates Euclidean distance once upfront.
     *
     */
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

    // calculate EuDist for all test observations
    arma::mat EuDis = calc_dist_mat_cpp(X, X_test);
    // Rcout << EuDis << std::endl;
    arma::vec noise(n);
    double noise_val = arma::randn<double>();
    noise.fill(noise_val);

    // loop through test observations and get ordered Y and B_NN for each
    // test observation

    NumericVector tuned_estimate(n_test);
    NumericVector s_1_B_NN(n_test);

    RcppThread::ProgressBar bar(X_test.n_rows, 1);
    RcppThread::parallelFor(
        0, X_test.n_rows, [&X, &Y, &X_test, &EuDis, &s_seq, &noise, &tuned_estimate, &s_1_B_NN, &n, &p, &n_prop, &B_NN, &scale_p, &debug, &bar](int i)
        {
        // get ith test observation
        arma::vec X_test_i = X_test.row(i).as_col();
        arma::mat X_test_i_mat = conv_to<arma::mat>::from(X_test.row(i));
        // get EuDist for ith test observation
        arma::vec eu_dist_col = EuDis.col(i);
        // sort each column and get the indices of the top B_NN
        arma::uvec sorted_idx = sort_index(eu_dist_col);
        arma::uvec top_B = sorted_idx.head(B_NN);
        arma::uvec idx_tmp = r_like_order(eu_dist_col, noise);
        arma::mat ordered_Y = conv_to<arma::mat>::from(Y).rows(idx_tmp);
        // get a 1 x s_seq.n_elem matrix. Find idx min
        arma::vec B_NN_estimates = dnn_B_NN_estimates(X, Y, X_test_i, top_B, s_seq, n_prop, B_NN, scale_p, debug);
        arma::uword min_idx = arma::index_min(B_NN_estimates);
        arma::vec best_s_1 = {s_seq(min_idx)};
        tuned_estimate(i) = dnn_ord_y_st(ordered_Y, best_s_1, n, p, n_prop);
        s_1_B_NN(i) = as_scalar(best_s_1); 
        bar++; },
        num_threads);

    if (estimate_variance)
    {
        if (verbose)
        {
            Rcout << "Running bootstrap...";
        }
        NumericMatrix bstrap_estimates = bootstrap_dnn_cpp_thread(X, Y, X_test, s_1_B_NN, n_prop,
                                                                  bootstrap_iter, num_threads, R_NilValue);
        // need to apply variance over columns
        arma::vec variance = rowVar_arma(as<arma::mat>(bstrap_estimates));
        return Rcpp::List::create(
            Named("estimate_loo") = tuned_estimate,
            Named("s_1_B_NN") = s_1_B_NN,
            Named("variance") = variance);
    }
    else
    {
        return Rcpp::List::create(
            Named("estimate_loo") = tuned_estimate,
            Named("s_1_B_NN") = s_1_B_NN);
    }
}

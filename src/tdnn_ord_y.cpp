#include "tdnn_ord_y.h"

struct TdnnEstimateOrdY : public Worker
{

    // input matrices to read from
    const arma::mat ordered_Y_mat;
    const arma::mat X;
    const arma::vec Y;
    const arma::mat X_test;
    const arma::mat weight_mat_s_1;
    const arma::mat weight_mat_s_2;

    // input constants
    const int n;
    const double c;
    const int p;

    // output vector to write to
    RVector<double> estimates;

    // initialize from Rcpp input and output matrixes (the RMatrix class
    // can be automatically converted to from the Rcpp matrix type)
    TdnnEstimateOrdY(const arma::mat &ordered_Y_mat,
                     const arma::mat &X,
                     const arma::vec &Y,
                     const arma::mat &X_test,
                     NumericVector estimates,
                     const arma::mat &weight_mat_s_1,
                     const arma::mat &weight_mat_s_2,
                     double c, int n, int p)
        : ordered_Y_mat(ordered_Y_mat),
          X(X),
          Y(Y),
          X_test(X_test),
          weight_mat_s_1(weight_mat_s_1),
          weight_mat_s_2(weight_mat_s_2),
          n(n), c(c), p(p),
          estimates(estimates) {}

    // function call operator that work for the specified range (begin/end)
    void operator()(std::size_t begin, std::size_t end)
    {
        for (std::size_t i = begin; i < end; i++)
        {
            arma::vec ordered_Y = ordered_Y_mat.col(i);
            arma::rowvec ordered_Y_row = ordered_Y_mat.col(i).as_row();
            arma::vec U_1_vec(ordered_Y.n_elem);
            // arma::vec U_2_vec(ordered_Y.n_rows);
            arma::vec U_2_vec;
            // double w_1 = c / (c - 1);
            // double w_2 = -1 / (c - 1);
            double w_2 = pow(c, 2 / double(p)) / (pow(c, 2 / double(p)) - 1);
            double w_1 = -1 / (pow(c, 2 / double(p)) - 1);

            // the weight matrix is # train obs x # test obs so we want to use
            // the ith column of the weight mat for the ith test observation
            // U_1_vec = reshape(ordered_Y, 1, n) * weight_mat_s_1.col(i);
            U_1_vec = ordered_Y_row * weight_mat_s_1.col(i);
            U_2_vec = ordered_Y_row * weight_mat_s_2.col(i);

            arma::vec U_vec = w_1 * U_1_vec + w_2 * U_2_vec;
            // Rcout << "U_vec: " << U_vec << std::endl;
            estimates[i] = arma::as_scalar(U_vec);
        }
    }
};

//' @export
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
arma::vec tdnn_ord_y(arma::mat X, arma::vec Y, arma::mat X_test,
                     arma::mat ordered_Y,
                     arma::vec s_1,
                     double c,
                     double n_prop)
{
    // int d = X.n_cols;
    // Infer n and p from our data after we've filtered for relevant features
    int n = X.n_rows;
    int p = X.n_cols;

    // This just creates a sequence 1:n and then reverses it
    NumericVector ord = seq_cpp(1, n);
    ord.attr("dim") = Dimension(n, 1);
    // don't need to reverse if we are using lfactorial
    // ord = n - ord;
    arma::vec ord_arma = as<arma::vec>(ord);

    arma::vec s_2 = arma::ceil(c * s_1);

    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(n, ord_arma, s_1, n_prop, false);
    arma::mat weight_mat_s_2 = weight_mat_lfac_s_2_filter(n, ord_arma, s_2, n_prop, true);

    NumericVector estimates(X_test.n_rows);
    TdnnEstimateOrdY tdnnEstimate(ordered_Y,
                                  X,
                                  Y,
                                  X_test,
                                  estimates,
                                  weight_mat_s_1,
                                  weight_mat_s_2,
                                  c, n, p);

    parallelFor(0, X_test.n_rows, tdnnEstimate);

    return as<arma::vec>(estimates);
}

// [[Rcpp::export]]
arma::vec tuning_ord_Y(const arma::mat &X, const arma::vec &Y,
                       const arma::mat &X_test,
                       const arma::mat &ordered_Y,
                       double c,
                       double n_prop)
{

    int n_obs = X_test.n_rows;
    int n = X.n_rows;
    bool search_for_s = true;
    int s_end = int(sqrt(n));
    arma::mat tuning_mat(n_obs, s_end, fill::zeros);
    arma::vec best_s(n_obs, fill::zeros);
    double s = 0;
    // using zero indexing here to match with C++, note s + 1 -> s+2 in de_dnn call

    while (search_for_s)
    {
        // Rcout << "s: " << s << std::endl;
        // s_val needs to be a vector of the same length as X_test
        arma::vec s_1(n_obs, fill::value(int(s + 2)));
        // double s_fill = s + 2;
        // NumericVector s_val(int(n_obs), s_fill);
        // s_val = s + 2;

        // For a given s, get the de_dnn estimates for each test observation
        // List de_dnn_estimates = de_dnn(X, Y, X_test, s_val, c, W0_);
        // arma::vec de_dnn_estimates = de_dnn(X, Y, X_test, s_val, c, n_prop, W0_);
        arma::vec de_dnn_estimates = tdnn_ord_y(X, Y, X_test,
                                                ordered_Y, s_1,
                                                c, n_prop);
        // This gives me an estimate for each test observation and is a n x 1 matrix
        // arma::vec de_dnn_est_vec = as<arma::vec>(de_dnn_estimates["estimates"]);
        arma::mat candidate_results = de_dnn_estimates;
        candidate_results.reshape(n_obs, 1);

        // Now we add this column to our matrix if the matrix is empty
        if (s == 0 | s == 1)
        {
            tuning_mat.col(s) = candidate_results;
        }
        else if (s >= tuning_mat.n_cols)
        {
            // if s > ncol tuning_mat, then we will choose best s from the existing choices for each row that hasn't found a best s yet and break out of the while loop
            arma::uvec s_vec = seq_int(0, int(s) - 1);
            arma::mat resized_mat = matrix_subset_idx(tuning_mat, s_vec);

            arma::mat out_diff = diff(resized_mat, 1, 1);
            IntegerVector idx = Range(0, (resized_mat.n_cols) - 2);
            arma::mat out_denom = resized_mat.cols(as<uvec>(idx));
            arma::mat diff_ratio = diff(abs(out_diff / out_denom), 1, 1);

            for (R_xlen_t i = 0; i < diff_ratio.n_rows; ++i)
            {
                // Only loop through the columns if we haven't already found a
                // suitable s
                if (best_s(i) == 0)
                {
                    for (R_xlen_t j = 0; j < diff_ratio.n_cols; ++j)
                    {
                        if (diff_ratio(i, j) > -0.01)
                        {
                            best_s(i) = j + 1 + 3;
                            break; // if we've found the column that satisfies our condition, break and move to next row.
                        }
                    }
                }
            }
            search_for_s = false; // since we've gone past the num of columns stop the while loop here
            break;                // break out of our while loop to avoid going past number of columns in tuning_mat
        }
        else
        {

            // instead of resizing the matrix, just select columns 0-s
            arma::uvec s_vec = seq_int(0, int(s) - 1);
            arma::mat resized_mat = matrix_subset_idx(tuning_mat, s_vec);
            // tuning_mat is an n x s matrix and we want to diff each of the rows
            arma::mat out_diff = diff(resized_mat, 1, 1);
            IntegerVector idx = Range(0, (resized_mat.n_cols) - 2);
            arma::mat out_denom = resized_mat.cols(as<uvec>(idx));
            arma::mat diff_ratio = diff(abs(out_diff / out_denom), 1, 1);
            // Now we go through each row and check if any of the columns are
            // greater than -0.01
            for (R_xlen_t i = 0; i < diff_ratio.n_rows; ++i)
            {
                // Only loop through the columns if we haven't already found a
                // suitable s
                if (best_s(i) == 0)
                {
                    for (R_xlen_t j = 0; j < diff_ratio.n_cols; ++j)
                    {
                        if (diff_ratio(i, j) > -0.01)
                        {
                            best_s(i) = j + 1 + 3;
                            break; // if we've found the column that satisfies our condition, break and move to next row.
                        }
                    }
                }
            }

            // Check if we still have observations without an s
            if (all(best_s))
            {
                // then we are done!
                search_for_s = false;
            }
            else
            {
                tuning_mat.col(s) = candidate_results;
            }
        }
        s += 1;
    }

    // return NumericVector(best_s.begin(), best_s.end());
    return best_s;
}

// [[Rcpp::export]]
arma::mat make_ordered_Y_mat_debug(const arma::mat &X,
                                   const arma::mat &Y, const arma::mat &X_test,
                                   int B_NN = 20)
{
    /*
        This is a debugging function for checking that we calculate the full
        matrix of ordered Y values correctly for all test observations.
        In our tuning code we order Y one-at-a-time.
    */
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
    arma::mat ordered_Y_mat(n, n_test);
    for (int i = 0; i < n_test; i++)
    {
        // get ith test observation
        arma::vec X_test_i = X_test.row(i).as_col();
        // get EuDist for ith test observation
        arma::vec eu_dist_col = EuDis.col(i);
        // sort each column and get the indices of the top B_NN
        arma::uvec sorted_idx = sort_index(eu_dist_col);
        arma::uvec top_B = sorted_idx.head(B_NN);
        arma::uvec idx_tmp = r_like_order(eu_dist_col, noise);

        ordered_Y_mat.col(i) = Y.rows(idx_tmp).as_col();
    }
    return ordered_Y_mat;
}

// [[Rcpp::export]]
arma::mat make_B_NN_estimates(
    const arma::mat &X,
    const arma::mat &Y,
    const arma::vec &X_test_i,
    const arma::uvec &top_B,
    double s_tmp, double c,
    double n_prop = 0.5, int B_NN = 20,
    double scale_p = 1,
    bool debug = false)
{
    int n = X.n_rows;
    int p = X.n_cols;
    arma::vec c_vec = {c};
    arma::vec s_1_vec_tmp = as<arma::vec>(seq_cpp(s_tmp, 2 * s_tmp));
    arma::uvec idx_vec = seq_int(0, n - 1);
    // arma::mat B_NN_estimates(B_NN, 5);
    arma::mat B_NN_estimates(c_vec.n_elem, s_1_vec_tmp.n_elem, fill::zeros);
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
        // to be column format, too so we will take the difference of them as columns
        double neighbor_weight = exp(
            -arma::sum(arma::pow((X_val.as_col() - X_test_i), 2)) / scale_p);
        // Rcout << -arma::sum(arma::pow((X_val.as_col() - X_test_i), 2)) << std::endl;
        double weighted_Y_val = as_scalar(Y_val * sqrt(neighbor_weight));
        for (int k = 0; k < c_vec.n_elem; k++)
        {
            double c_val = c_vec(k);
            for (int l = 0; l < s_1_vec_tmp.n_elem; l++)
            {
                arma::vec s_1_val = {s_1_vec_tmp(l)};
                arma::vec param_estimate = tdnn_ord_y(X_train, Y_train, X_val,
                                                      ordered_Y_train,
                                                      s_1_val, c_val,
                                                      n_prop);
                double weighted_estimate = as_scalar(param_estimate * sqrt(neighbor_weight));
                double loss = pow((weighted_estimate - weighted_Y_val), 2);
                if (debug)
                {
                    arma::rowvec debug_vec = {double(j), c_val, s_1_vec_tmp(l), as_scalar(Y_val), neighbor_weight, weighted_estimate, weighted_Y_val, loss};
                    Rcout << "{b, c_val, s_1_vec_tmp(l), as_scalar(Y_val), neighbor_weight, weighted_estimate, weighted_Y_val, loss}" << std::endl;
                    Rcout << debug_vec << std::endl;
                }

                B_NN_estimates(k, l) += loss;
                // B_NN_estimates.row(i) = { loss, weighted_Y_val,weighted_estimate, s_1_vec_tmp(l), c_val};
            }
        }
    }
    // now calculate mean loss over the B_NN observations
    B_NN_estimates = B_NN_estimates / B_NN;
    return B_NN_estimates;
}

// [[Rcpp::export]]
arma::vec choose_s_1_val(arma::vec tuned_mse, arma::vec s_1_vec_tmp)
{
    double min_val = as_scalar(tuned_mse.min());
    arma::uvec near_min_vals = find(tuned_mse <= (1 + 0.01) * min_val);
    arma::vec choose_s1 = {min(s_1_vec_tmp.elem(near_min_vals))};
    return choose_s1;
}

// [[Rcpp::export]]
Rcpp::List tune_de_dnn_no_dist_cpp(
    arma::mat X,
    arma::vec Y,
    arma::mat X_test,
    Nullable<NumericVector> W0_,
    double c = 2,
    int B_NN = 20,
    double scale_p = 1,
    double n_prop = 0.5,
    bool estimate_variance = false,
    int bootstrap_iter = 500,
    bool debug = false)
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

    // calculate EuDist for all test observations
    arma::mat EuDis = calc_dist_mat_cpp(X, X_test);
    // Rcout << EuDis << std::endl;
    arma::vec noise(n);
    double noise_val = arma::randn<double>();
    noise.fill(noise_val);

    // loop through test observations and get ordered Y and B_NN for each
    // test observation

    arma::vec tuned_estimate(n_test);
    arma::vec s_1_B_NN(n_test);
    arma::vec curve_estimate(n_test);
    arma::vec s_1_mse_curve(n_test);
    for (int i = 0; i < n_test; i++)
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
        arma::vec mse_curve_s = tuning_ord_Y(X, Y, X_test_i_mat, ordered_Y, c, n_prop);
        if (debug)
        {
            arma::vec estimate_curve = tdnn_ord_y(X, Y, X_test_i_mat, ordered_Y,
                                                  mse_curve_s, c, n_prop);
            curve_estimate(i) = as_scalar(estimate_curve);
            s_1_mse_curve(i) = as_scalar(mse_curve_s);
        }
        double s_tmp = arma::as_scalar(mse_curve_s);
        arma::vec c_vec = {c};
        arma::vec s_1_vec_tmp = as<arma::vec>(seq_cpp(s_tmp, 2 * s_tmp));
        arma::uvec idx_vec = seq_int(0, n - 1);
        arma::mat B_NN_estimates = make_B_NN_estimates(X, Y, X_test_i, top_B, s_tmp, c, n_prop, B_NN, scale_p, debug);

        // for now we are going to assume we have a single c value so we will get our single tuned MSE vector
        arma::vec tuned_mse = B_NN_estimates.as_col();
        double min_val = as_scalar(tuned_mse.min());
        arma::uvec near_min_vals = find(tuned_mse <= (1 + 0.01) * min_val);
        arma::vec choose_s1 = {min(s_1_vec_tmp.elem(near_min_vals))};
        tuned_estimate(i) = as_scalar(tdnn_ord_y(X, Y, X_test_i_mat,
                                                 ordered_Y, choose_s1, c, n_prop));
        s_1_B_NN(i) = as_scalar(choose_s1);
    }

    if (estimate_variance)
    {
        NumericMatrix bstrap_estimates = bootstrap_cpp_mt(X, Y, X_test, s_1_B_NN, c, n_prop, bootstrap_iter, R_NilValue);
        // need to apply variance over columns
        arma::vec variance = rowVar_arma(as<arma::mat>(bstrap_estimates));
        if (debug)
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("estimate_curve") = curve_estimate,
                Named("s_1_B_NN") = s_1_B_NN,
                Named("s_1_mse_curve") = s_1_mse_curve,
                Named("variance") = variance);
        }
        else
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("s_1_B_NN") = s_1_B_NN,
                Named("variance") = variance);
        }
    }
    else
    {
        if (debug)
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("estimate_curve") = curve_estimate,
                Named("s_1_B_NN") = s_1_B_NN,
                Named("s_1_mse_curve") = s_1_mse_curve);
        }
        else
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("s_1_B_NN") = s_1_B_NN);
        }
    }
}

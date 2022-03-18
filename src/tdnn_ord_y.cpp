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

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
double tdnn_ord_y_st(arma::mat ordered_Y_i,
                     arma::vec s_1,
                     int n,
                     int p,
                     double c,
                     double n_prop)
{

    // This just creates a sequence 1:n and then reverses it
    // NumericVector ord = seq_cpp(1, n);
    // ord.attr("dim") = Dimension(n, 1);
    // don't need to reverse if we are using lfactorial
    // ord = n - ord;
    arma::vec ord_arma = seq_cpp_arma(1, n);
    arma::vec s_2 = arma::ceil(c * s_1);

    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(n, ord_arma, s_1, n_prop, false);
    arma::mat weight_mat_s_2 = weight_mat_lfac_s_2_filter(n, ord_arma, s_2, n_prop, true);

    arma::rowvec ordered_Y_row = ordered_Y_i.as_row();
    arma::vec U_1_vec(ordered_Y_i.n_elem);
    // arma::vec U_2_vec(ordered_Y.n_rows);
    arma::vec U_2_vec;
    // double w_1 = c / (c - 1);
    // double w_2 = -1 / (c - 1);
    double w_2 = pow(c, 2 / double(p)) / (pow(c, 2 / double(p)) - 1);
    double w_1 = -1 / (pow(c, 2 / double(p)) - 1);

    // the weight matrix is # train obs x # test obs so we want to use
    // the ith column of the weight mat for the ith test observation
    // U_1_vec = reshape(ordered_Y, 1, n) * weight_mat_s_1.col(i);
    U_1_vec = ordered_Y_row * weight_mat_s_1.col(0);
    U_2_vec = ordered_Y_row * weight_mat_s_2.col(0);

    arma::vec U_vec = w_1 * U_1_vec + w_2 * U_2_vec;
    // Rcout << "U_vec: " << U_vec << std::endl;
    return arma::as_scalar(U_vec);
}

// [[Rcpp::export]]
Rcpp::List tuning_ord_Y_debug(const arma::mat &X, const arma::vec &Y,
                              const arma::mat &X_test,
                              const arma::mat &ordered_Y,
                              double c,
                              double n_prop)
{

    int n_obs = X_test.n_rows;
    int n = X.n_rows;
    bool search_for_s = true;
    int s_end = int(sqrt(n));
    arma::mat tuning_mat(n_obs, s_end);
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
    // return best_s;
    return Rcpp::List::create(
        Named("best_s") = best_s,
        Named("tuning_mat") = tuning_mat);
}

// [[Rcpp::export]]
arma::mat tuning_ord_Y_tune_c(const arma::mat &X, const arma::vec &Y,
                              const arma::mat &X_test,
                              const arma::mat &ordered_Y,
                              const arma::vec &c,
                              double n_prop)
{

    int n_obs = X_test.n_rows;
    int n = X.n_rows;
    bool search_for_s = true;
    int s_end = int(sqrt(n));
    // tune_c_mat has 3 columns: best_s for each test observation, c value, and corresponding de_dnn estimate
    arma::mat tune_c_mat(n_obs, 3);
    for (int j = 0; j < c.n_elem; j++)
    {
        double c_val = as_scalar(c(j));
        arma::mat tuning_mat(n_obs, s_end, fill::zeros);
        arma::vec best_s(n_obs, fill::zeros);
        arma::vec best_s_estimates(n_obs);
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
                                                    c_val, n_prop);
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
                                best_s_estimates(i) = tuning_mat(i, (j + 1 + 3 - 2));
                                double best_s_tmp = j + 1 + 3;
                                double best_s_est_tmp = as_scalar(tuning_mat(i, (j + 1 + 3 - 2)));
                                Rcout << best_s_tmp << ", " << best_s_est_tmp << std::endl;
                                tune_c_mat.row(i) = {c_val, best_s_tmp, best_s_est_tmp};
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
                                best_s_estimates(i) = tuning_mat(i, (j + 1 + 3 - 2));
                                double best_s_tmp = j + 1 + 3;
                                double best_s_est_tmp = as_scalar(tuning_mat(i, (j + 1 + 3 - 2)));
                                Rcout << best_s_tmp << ", " << best_s_est_tmp << std::endl;
                                tune_c_mat.row(i) = {c_val, best_s_tmp, best_s_est_tmp};
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
        // need to get the estimates for each of the best s values
    }
    // return NumericVector(best_s.begin(), best_s.end());
    return tune_c_mat;
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
        // arma::vec de_dnn_estimates = tdnn_ord_y_st(ordered_Y, s_1, n, p,
        //                                         c, n_prop);
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
arma::vec tuning_ord_Y_st(const arma::mat &ordered_Y, int n, int p, int n_obs,
                          double c,
                          double n_prop)
{

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
        arma::vec de_dnn_estimates = {tdnn_ord_y_st(ordered_Y, s_1, n, p,
                                                    c, n_prop)};
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
            // IntegerVector idx = Range(0, (resized_mat.n_cols) - 2);
            arma::uvec idx = seq_int(0, (resized_mat.n_cols) - 2);
            arma::mat out_denom = resized_mat.cols(idx);
            arma::mat diff_ratio = diff(abs(out_diff / out_denom), 1, 1);

            for (int i = 0; i < diff_ratio.n_rows; ++i)
            {
                // Only loop through the columns if we haven't already found a
                // suitable s
                if (best_s(i) == 0)
                {
                    for (int j = 0; j < diff_ratio.n_cols; ++j)
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
            // IntegerVector idx_R = Range(0, (resized_mat.n_cols) - 2);
            // Rcout << "idx R: " << idx_R << std::endl;
            // arma::mat out_denom = resized_mat.cols(as<uvec>(idx));
            arma::uvec idx = seq_int(0, (resized_mat.n_cols) - 2);
            // Rcout << "idx: " << idx << std::endl;
            arma::mat out_denom = resized_mat.cols(idx);
            arma::mat diff_ratio = diff(abs(out_diff / out_denom), 1, 1);
            // Now we go through each row and check if any of the columns are
            // greater than -0.01
            for (int i = 0; i < diff_ratio.n_rows; ++i)
            {
                // Only loop through the columns if we haven't already found a
                // suitable s
                if (best_s(i) == 0)
                {
                    for (int j = 0; j < diff_ratio.n_cols; ++j)
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
    const arma::vec c_vec,
    double s_tmp,
    double n_prop = 0.5, int B_NN = 20,
    double scale_p = 1,
    bool debug = false)
{
    int n = X.n_rows;
    // arma::vec c_vec = {c};
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
arma::vec choose_s_1_c_val(arma::vec s_1_vec, arma::vec c, arma::mat B_NN_estimates)
{
    arma::mat best_s_1_c(c.n_elem, 3);
    // loop over rows of B_NN_estimates matrix and get entry with smallest value
    for (int k = 0; k < B_NN_estimates.n_rows; k++)
    {
        arma::vec tuned_mse = B_NN_estimates.row(k).as_col();
        double c_val = c(k);
        double min_val = as_scalar(tuned_mse.min());
        arma::uvec near_min_vals = find(tuned_mse <= (1 + 0.01) * min_val);
        arma::vec near_min_s_1 = s_1_vec.elem(near_min_vals);
        int s_1_idx = index_min(near_min_s_1);
        // this gets the smallest s_1 amongst the tuned mse values nearest the minimum mse
        double choose_s1 = as_scalar(near_min_s_1);
        double s_1_mse = tuned_mse(s_1_idx);
        best_s_1_c.row(k) = {c_val, choose_s1, s_1_mse};
    }
    // get index of row with minimum tuned mse
    uword best_row = best_s_1_c.col(2).index_min();
    arma::rowvec best_row_tmp = best_s_1_c.row(best_row);
    double best_c = best_row_tmp(0);
    double best_s_1 = best_row_tmp(1);
    arma::vec best_values = {best_c, best_s_1};
    return best_values;
}

// [[Rcpp::export]]
Rcpp::List tune_de_dnn_no_dist_cpp(
    arma::mat X,
    arma::vec Y,
    arma::mat X_test,
    Nullable<NumericVector> W0_,
    arma::vec c,
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
    double fixed_c = 2;

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
    arma::vec c_B_NN(n_test);
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
        arma::vec mse_curve_s = tuning_ord_Y(X, Y, X_test_i_mat, ordered_Y, fixed_c, n_prop);
        if (debug)
        {
            arma::vec estimate_curve = tdnn_ord_y(X, Y, X_test_i_mat, ordered_Y,
                                                  mse_curve_s, fixed_c, n_prop);
            curve_estimate(i) = as_scalar(estimate_curve);
            s_1_mse_curve(i) = as_scalar(mse_curve_s);
        }
        double s_tmp = arma::as_scalar(mse_curve_s);
        // arma::vec c_vec = {c};
        arma::vec s_1_vec_tmp = as<arma::vec>(seq_cpp(s_tmp, 2 * s_tmp));
        arma::uvec idx_vec = seq_int(0, n - 1);
        arma::mat B_NN_estimates = make_B_NN_estimates(X, Y, X_test_i, top_B, c, s_tmp, n_prop, B_NN, scale_p, debug);
        arma::mat best_s_1_c(c.n_elem, 3);
        // loop over rows of B_NN_estimates matrix and get entry with smallest value
        for (int k = 0; k < B_NN_estimates.n_rows; k++)
        {
            arma::vec tuned_mse = B_NN_estimates.row(k).as_col();
            double c_val = c(k);
            double min_val = as_scalar(tuned_mse.min());
            arma::uvec near_min_vals = find(tuned_mse <= (1 + 0.01) * min_val);
            arma::vec near_min_s_1 = s_1_vec_tmp.elem(near_min_vals);
            int s_1_idx = index_min(near_min_s_1);
            // this gets the smallest s_1 amongst the tuned mse values nearest the minimum mse
            double choose_s1 = as_scalar(min(near_min_s_1));
            double s_1_mse = tuned_mse(s_1_idx);
            best_s_1_c.row(k) = {c_val, choose_s1, s_1_mse};
        }
        // get index of row with minimum tuned mse
        uword best_row = best_s_1_c.col(2).index_min();
        arma::rowvec best_row_tmp = best_s_1_c.row(best_row);
        double best_c = best_row_tmp(0);
        double best_s_1 = best_row_tmp(1);
        arma::vec best_s_1_vec = {best_s_1};

        tuned_estimate(i) = as_scalar(tdnn_ord_y(X, Y, X_test_i_mat,
                                                 ordered_Y, best_s_1_vec, best_c, n_prop));
        s_1_B_NN(i) = best_s_1;
        c_B_NN(i) = best_c;
    }

    if (estimate_variance)
    {
        NumericMatrix bstrap_estimates = bootstrap_cpp_mt(X, Y, X_test, s_1_B_NN, c_B_NN, n_prop, bootstrap_iter, R_NilValue);
        // need to apply variance over columns
        arma::vec variance = rowVar_arma(as<arma::mat>(bstrap_estimates));
        if (debug)
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("estimate_curve") = curve_estimate,
                Named("s_1_B_NN") = s_1_B_NN,
                Named("c_B_NN") = c_B_NN,
                Named("s_1_mse_curve") = s_1_mse_curve,
                Named("variance") = variance);
        }
        else
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("s_1_B_NN") = s_1_B_NN,
                Named("c_B_NN") = c_B_NN,
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

// [[Rcpp::export]]
arma::mat make_B_NN_estimates_st(
    const arma::mat &X,
    const arma::mat &Y,
    const arma::vec &X_test_i,
    const arma::uvec &top_B,
    const arma::vec c_vec,
    double s_tmp,
    double n_prop = 0.5, int B_NN = 20,
    double scale_p = 1,
    bool debug = false)
{
    int n = X.n_rows;
    // arma::vec c_vec = {c};
    // arma::vec s_1_vec_tmp = as<arma::vec>(seq_cpp(s_tmp, 2 * s_tmp));
    arma::vec s_1_vec_tmp = seq_cpp_arma(s_tmp, 2 * s_tmp);
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
                double param_estimate = tdnn_ord_y_st(ordered_Y_train, s_1_val, X_train.n_rows, X_train.n_cols, c_val, n_prop);
                // arma::vec param_estimate = tdnn_ord_y(X_train, Y_train, X_val,
                //                                       ordered_Y_train,
                //                                       s_1_val, c_val,
                //                                       n_prop);
                // double weighted_estimate = as_scalar(param_estimate * sqrt(neighbor_weight));
                double weighted_estimate = param_estimate * sqrt(neighbor_weight);
                double loss = pow((weighted_estimate - weighted_Y_val), 2);

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
Rcpp::List tune_de_dnn_no_dist_vary_c_cpp(
    arma::mat X,
    arma::vec Y,
    arma::mat X_test,
    Nullable<NumericVector> W0_,
    arma::vec c,
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
    double fixed_c = 2;

    // calculate EuDist for all test observations
    arma::mat EuDis = calc_dist_mat_cpp(X, X_test);
    // Rcout << EuDis << std::endl;
    arma::vec noise(n);
    double noise_val = arma::randn<double>();
    noise.fill(noise_val);

    arma::vec tuned_estimate(n_test);
    arma::vec s_1_B_NN(n_test);
    arma::vec c_B_NN(n_test);
    arma::mat curve_estimate(n_test, c.n_elem);
    arma::mat s_1_mse_curve(n_test, c.n_elem);
    // loop through test observations and get ordered Y and B_NN for each
    // test observation
    for (int i = 0; i < n_test; i++)
    {
        arma::mat vary_c_results(c.n_elem, 3);

        arma::vec X_test_i = X_test.row(i).as_col();
        arma::mat X_test_i_mat = conv_to<arma::mat>::from(X_test.row(i));
        // get EuDist for ith test observation
        arma::vec eu_dist_col = EuDis.col(i);
        // sort each column and get the indices of the top B_NN
        arma::uvec sorted_idx = sort_index(eu_dist_col);
        arma::uvec top_B = sorted_idx.head(B_NN);
        arma::uvec idx_tmp = r_like_order(eu_dist_col, noise);
        arma::mat ordered_Y = conv_to<arma::mat>::from(Y).rows(idx_tmp);

        for (int j = 0; j < c.n_elem; j++)
        {
            double c_val = c(j);
            arma::vec c_val_vec = {c_val};
            // get ith test observation
            arma::vec mse_curve_s = tuning_ord_Y(X, Y, X_test_i_mat, ordered_Y, c_val, n_prop);
            if (debug)
            {
                double estimate_curve = tdnn_ord_y_st(ordered_Y, mse_curve_s, n, p, c_val, n_prop);
                // arma::vec estimate_curve = tdnn_ord_y(X, Y, X_test_i_mat, ordered_Y,
                //                                       mse_curve_s, c_val, n_prop);
                // curve_estimate(i, j) = as_scalar(estimate_curve);
                curve_estimate(i, j) = estimate_curve;
                s_1_mse_curve(i, j) = as_scalar(mse_curve_s);
            }
            double s_tmp = arma::as_scalar(mse_curve_s);
            // arma::vec c_vec = {c};
            arma::vec s_1_vec_tmp = as<arma::vec>(seq_cpp(s_tmp, 2 * s_tmp));
            // Rcout << "s_1_vec_tmp: " << s_1_vec_tmp << std::endl;
            arma::mat B_NN_estimates = make_B_NN_estimates(X, Y, X_test_i, top_B, c_val_vec,
                                                           s_tmp, n_prop, B_NN, scale_p, debug);
            // Rcout << "B_NN_estimates: " << B_NN_estimates << std::endl;
            arma::mat best_s_1_c(c_val_vec.n_elem, 3);
            // loop over rows of B_NN_estimates matrix and get entry with smallest value
            for (int k = 0; k < B_NN_estimates.n_rows; k++)
            {
                arma::vec tuned_mse = B_NN_estimates.row(k).as_col();
                // double c_val = c(k);
                double min_val = as_scalar(tuned_mse.min());
                arma::uvec near_min_vals = find(tuned_mse <= (1 + 0.01) * min_val);
                double choose_s1 = as_scalar(min(s_1_vec_tmp.elem(near_min_vals)));
                // Rcout << "choose_s1: " << choose_s1 << std::endl;
                double s_1_mse = as_scalar(tuned_mse.elem(find(s_1_vec_tmp == choose_s1)));
                // Rcout << "s_1_mse: " << s_1_mse << std::endl;
                best_s_1_c.row(k) = {c_val, choose_s1, s_1_mse};
            }
            // get index of row with minimum tuned mse
            uword best_row = best_s_1_c.col(2).index_min();
            arma::rowvec best_row_tmp = best_s_1_c.row(best_row);
            vary_c_results.row(j) = best_row_tmp;
        }
        // Rcout << "vary_c_results: " << vary_c_results << std::endl;
        // now we have the min values for each c value, need min of mins
        uword best_c_row = vary_c_results.col(2).index_min();
        arma::rowvec best_c_row_tmp = vary_c_results.row(best_c_row);
        // Rcout << "best_c_row idx :" << best_c_row << std::endl;
        // Rcout << "best_c_row_tmp: " << best_c_row_tmp << std::endl;
        // now get the estimate using these values
        double best_c = best_c_row_tmp(0);
        double best_s_1 = best_c_row_tmp(1);
        arma::vec best_s_1_vec = {best_s_1};

        // tuned_estimate(i) = as_scalar(tdnn_ord_y(X, Y, X_test_i_mat,
        //                                          ordered_Y, best_s_1_vec, best_c, n_prop));
        tuned_estimate(i) = tdnn_ord_y_st(ordered_Y, best_s_1_vec, n, p, best_c, n_prop);
        s_1_B_NN(i) = best_s_1;
        c_B_NN(i) = best_c;
    }

    if (estimate_variance)
    {
        NumericMatrix bstrap_estimates = bootstrap_cpp_mt(X, Y, X_test, s_1_B_NN,
                                                          c_B_NN, n_prop,
                                                          bootstrap_iter, R_NilValue);
        // need to apply variance over columns
        arma::vec variance = rowVar_arma(as<arma::mat>(bstrap_estimates));
        if (debug)
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("estimate_curve") = curve_estimate,
                Named("s_1_B_NN") = s_1_B_NN,
                Named("c_B_NN") = c_B_NN,
                Named("s_1_mse_curve") = s_1_mse_curve,
                Named("variance") = variance);
        }
        else
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("s_1_B_NN") = s_1_B_NN,
                Named("c_B_NN") = c_B_NN,
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
                Named("c_B_NN") = c_B_NN,
                Named("s_1_mse_curve") = s_1_mse_curve);
        }
        else
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("s_1_B_NN") = s_1_B_NN,
                Named("c_B_NN") = c_B_NN);
        }
    }
}

struct TdnnEstimateTune : public Worker
{
    // outputs
    RVector<double> tuned_estimate;
    RVector<double> s_1_B_NN;
    RVector<double> c_B_NN;
    RMatrix<double> curve_estimate;
    RMatrix<double> s_1_mse_curve;

    // inputs
    const arma::mat X;
    const arma::vec Y;
    const arma::mat X_test;
    const arma::mat EuDis;
    const arma::vec noise;
    const arma::vec c;

    // constants
    int B_NN;
    int n;
    int p;
    double scale_p;
    bool debug;
    double n_prop = 0.5;

    TdnnEstimateTune(
        const arma::mat &X,
        const arma::vec &Y,
        const arma::mat &X_test,
        const arma::mat &EuDis,
        const arma::vec &noise,
        const arma::vec &c,
        NumericVector tuned_estimate,
        NumericVector s_1_B_NN,
        NumericVector c_B_NN,
        NumericMatrix curve_estimate,
        NumericMatrix s_1_mse_curve,
        int B_NN, int n, int p,
        double scale_p,
        bool debug

        ) : X(X), Y(Y), X_test(X_test),
            EuDis(EuDis), noise(noise), c(c),
            tuned_estimate(tuned_estimate),
            s_1_B_NN(s_1_B_NN), c_B_NN(c_B_NN), curve_estimate(curve_estimate), s_1_mse_curve(s_1_mse_curve),
            B_NN(B_NN), n(n), p(p), scale_p(scale_p), debug(debug)
    {
    }

    void operator()(std::size_t begin, std::size_t end)
    {
        for (std::size_t i = begin; i < end; i++)
        {
            arma::mat vary_c_results(c.n_elem, 3);

            arma::vec X_test_i = X_test.row(i).as_col();
            // Rcout << X_test_i << std::endl;
            arma::mat X_test_i_mat = conv_to<arma::mat>::from(X_test.row(i));
            // get EuDist for ith test observation
            arma::vec eu_dist_col = EuDis.col(i);
            // sort each column and get the indices of the top B_NN
            arma::uvec sorted_idx = sort_index(eu_dist_col);
            arma::uvec top_B = sorted_idx.head(B_NN);
            arma::uvec idx_tmp = r_like_order(eu_dist_col, noise);
            arma::mat ordered_Y = conv_to<arma::mat>::from(Y).rows(idx_tmp);

            for (int j = 0; j < c.n_elem; j++)
            {
                double c_val = c(j);
                arma::vec c_val_vec = {c_val};
                double estimate_curve = 0;
                // get ith test observation

                arma::vec mse_curve_s = tuning_ord_Y_st(ordered_Y, n, p, 1, c_val, n_prop);
                mse_curve_s.print();
                cout << c.n_elem << std::endl;
                cout << c_val << std::endl;
                cout << j << std::endl;
                // if (debug)
                // {
                //     estimate_curve = tdnn_ord_y_st(ordered_Y, mse_curve_s, n, p, c_val, n_prop);
                //     // arma::vec estimate_curve = tdnn_ord_y(X, Y, X_test_i_mat, ordered_Y,
                //     //                                       mse_curve_s, c_val, n_prop);
                //     // cout << "i,j: " << i << ", " << j << std::endl;
                //     // cout << "mse_curve_s: " << as_scalar(mse_curve_s) << std::endl;
                // }
                // curve_estimate(i, j) = estimate_curve;
                // s_1_mse_curve(i, j) = as_scalar(mse_curve_s);
                double s_tmp = arma::as_scalar(mse_curve_s);
                // arma::vec c_vec = {c};
                // arma::vec s_1_vec_tmp = as<arma::vec>(seq_cpp(s_tmp, 2 * s_tmp));
                arma::vec s_1_vec_tmp = seq_cpp_arma(s_tmp, 2 * s_tmp);
                // Rcout << "s_1_vec_tmp: " << s_1_vec_tmp << std::endl;
                arma::mat B_NN_estimates = make_B_NN_estimates_st(X, Y, X_test_i, top_B, c_val_vec,
                                                                  s_tmp, n_prop, B_NN, scale_p, debug = false);
                arma::mat best_s_1_c(c_val_vec.n_elem, 3);
                // loop over rows of B_NN_estimates matrix and get entry with smallest value
                for (int k = 0; k < B_NN_estimates.n_rows; k++)
                {
                    arma::vec tuned_mse = B_NN_estimates.row(k).as_col();
                    // double c_val = c(k);
                    double min_val = as_scalar(tuned_mse.min());
                    arma::uvec near_min_vals = find(tuned_mse <= (1 + 0.01) * min_val);
                    double choose_s1 = as_scalar(min(s_1_vec_tmp.elem(near_min_vals)));
                    // Rcout << "choose_s1: " << choose_s1 << std::endl;
                    double s_1_mse = as_scalar(tuned_mse.elem(find(s_1_vec_tmp == choose_s1)));
                    // Rcout << "s_1_mse: " << s_1_mse << std::endl;
                    best_s_1_c.row(k) = {c_val, choose_s1, s_1_mse};
                }
                // get index of row with minimum tuned mse
                uword best_row = best_s_1_c.col(2).index_min();
                arma::rowvec best_row_tmp = best_s_1_c.row(best_row);
                vary_c_results.row(j) = best_row_tmp;
            }
            // Rcout << "vary_c_results: " << vary_c_results << std::endl;
            // now we have the min values for each c value, need min of mins
            uword best_c_row = vary_c_results.col(2).index_min();
            arma::rowvec best_c_row_tmp = vary_c_results.row(best_c_row);
            // Rcout << "best_c_row idx :" << best_c_row << std::endl;
            // Rcout << "best_c_row_tmp: " << best_c_row_tmp << std::endl;
            // now get the estimate using these values
            double best_c = best_c_row_tmp(0);
            double best_s_1 = best_c_row_tmp(1);
            arma::vec best_s_1_vec = {best_s_1};

            // tuned_estimate(i) = as_scalar(tdnn_ord_y(X, Y, X_test_i_mat,
            //                                          ordered_Y, best_s_1_vec, best_c, n_prop));
            tuned_estimate[i] = tdnn_ord_y_st(ordered_Y, best_s_1_vec, n, p, best_c, n_prop);
            s_1_B_NN[i] = best_s_1;
            c_B_NN[i] = best_c;
        }
    }
};

// [[Rcpp::export]]
Rcpp::List tune_de_dnn_no_dist_vary_c_cpp_mt(
    arma::mat X,
    arma::vec Y,
    arma::mat X_test,
    Nullable<NumericVector> W0_,
    arma::vec c,
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
    double fixed_c = 2;

    // calculate EuDist for all test observations
    arma::mat EuDis = calc_dist_mat_cpp(X, X_test);
    // Rcout << EuDis << std::endl;
    arma::vec noise(n);
    double noise_val = arma::randn<double>();
    noise.fill(noise_val);

    NumericVector tuned_estimate(n_test);
    NumericVector s_1_B_NN(n_test);
    NumericVector c_B_NN(n_test);
    NumericMatrix curve_estimate(n_test, c.n_elem);
    NumericMatrix s_1_mse_curve(n_test, c.n_elem);

    TdnnEstimateTune parallel_est_tune(X, Y, X_test, EuDis, noise,
                                       c, tuned_estimate, s_1_B_NN, c_B_NN, curve_estimate,
                                       s_1_mse_curve, B_NN, n, p, scale_p, debug);

    parallelFor(0, X_test.n_rows, parallel_est_tune);

    if (estimate_variance)
    {
        NumericMatrix bstrap_estimates = bootstrap_cpp_mt(X, Y, X_test, s_1_B_NN,
                                                          c_B_NN, n_prop,
                                                          bootstrap_iter, R_NilValue);
        // need to apply variance over columns
        arma::vec variance = rowVar_arma(as<arma::mat>(bstrap_estimates));
        if (debug)
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("estimate_curve") = curve_estimate,
                Named("s_1_B_NN") = s_1_B_NN,
                Named("c_B_NN") = c_B_NN,
                Named("s_1_mse_curve") = s_1_mse_curve,
                Named("variance") = variance);
        }
        else
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("s_1_B_NN") = s_1_B_NN,
                Named("c_B_NN") = c_B_NN,
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
                Named("c_B_NN") = c_B_NN,
                Named("s_1_mse_curve") = s_1_mse_curve);
        }
        else
        {
            return Rcpp::List::create(
                Named("estimate_loo") = tuned_estimate,
                Named("s_1_B_NN") = s_1_B_NN,
                Named("c_B_NN") = c_B_NN);
        }
    }
}

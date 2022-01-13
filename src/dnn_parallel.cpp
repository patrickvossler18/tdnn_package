#include "util.h"
#include "kd_tree.h"

struct TdnnEstimate : public Worker {

    // input matrices to read from
    const arma::mat X;
    const arma::mat X_test;
    const arma::vec Y;
    const arma::mat weight_mat_s_1;
    const arma::mat weight_mat_s_2;
    const arma::mat weight_mat_s_1_plus_1;
    const arma::mat weight_mat_s_2_plus_1;

    // input constants
    const int n;
    const int p;
    const double c;
    const int d;

    // output vector to write to
    RVector<double> estimates;

    // initialize from Rcpp input and output matrixes (the RMatrix class
    // can be automatically converted to from the Rcpp matrix type)
    TdnnEstimate(const arma::mat & X, const arma::vec & Y,
                   const arma::mat & X_test,
                   NumericVector estimates,
                   const arma::mat& weight_mat_s_1,
                   const arma::mat& weight_mat_s_2,
                   const arma::mat& weight_mat_s_1_plus_1,
                   const arma::mat& weight_mat_s_2_plus_1,
                   double c, int n, int p, int d
    )
        : X(X), X_test(X_test),
          Y(Y),
          weight_mat_s_1(weight_mat_s_1),
          weight_mat_s_2(weight_mat_s_2),
          weight_mat_s_1_plus_1(weight_mat_s_1_plus_1),
          weight_mat_s_2_plus_1(weight_mat_s_2_plus_1),
          n(n),
          p(p), c(c), d(d),
          estimates(estimates){}

    // function call operator that work for the specified range (begin/end)
    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            // arma::mat single_vec(int(n),1);
            // single_vec.fill(1.0);
            arma::mat all_cols(p,1);
            all_cols.fill(1.0);

            // arma::mat all_rows;
            arma::mat X_dis;
            arma::mat EuDis;

            arma::mat X_test_row =  X_test.row(i);
            // all_rows = single_vec * X_test_row;
            arma::mat all_rows = arma::repmat(X_test_row,n,1);

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

            // std::vector<double> eu_dis = conv_to<std::vector<double>>::from(vec_eu_dis);
            // std::vector<double> noise_vec = conv_to<std::vector<double>>::from(noise);
            //
            // vector<int> index(int(n), 0);
            // for (int i = 0 ; i != index.size() ; i++) {
            //     index[i] = i;
            // }
            // sort(index.begin(), index.end(),
            //      [&](const int& a, const int& b) {
            //          if (eu_dis[a] != eu_dis[b]){
            //              return eu_dis[a] < eu_dis[b];
            //          }
            //          return noise_vec[a] < noise_vec[b];
            //      }
            // );
            // arma::vec ordered_Y;
            // arma::mat ordered_Y_vec = conv_to<arma::mat>::from(Y).rows(conv_to<arma::uvec>::from(index));

            arma::vec ordered_Y;
            arma::mat ordered_Y_vec = conv_to<arma::mat>::from(Y).rows(index);
            // // Rcout << ordered_Y_vec[order_vec];
            ordered_Y = ordered_Y_vec;
            // // Rcout << ordered_Y;

            arma::vec U_1_vec(ordered_Y.n_rows);
            // arma::vec U_2_vec(ordered_Y.n_rows);
            arma::vec U_2_vec;

            arma::vec U_1_1_vec(ordered_Y.n_rows);
            arma::vec U_2_1_vec;

            double w_1 = c/(c-1);
            double w_2 = -1/(c-1);

            // the weight matrix is # train obs x # test obs so we want to use the ith column of the weight mat for the ith test observation
            U_1_vec = reshape(ordered_Y,1,n) * weight_mat_s_1.col(i);
            U_1_1_vec = reshape(ordered_Y,1,n) * weight_mat_s_1_plus_1.col(i);
            if(arma::accu(weight_mat_s_2.col(i)) == 0){
                // in this case s_2 is too large so we will get the 1-NN to use as the estimate
                // std::cout << "using 1-NN" << std::endl;
                // std::cout << "X_test_row:" << X_test_row << std::endl;
                // std::cout << "X:" << X << std::endl;
                // std::cout << "Y:" << Y << std::endl;
                arma::vec nn_1_result = get_1nn_reg(X, X_test_row, Y, 1);
                U_2_vec = arma::as_scalar(nn_1_result);
                U_2_1_vec = arma::as_scalar(nn_1_result);
                // std::cout << "U_2_vec: " <<  U_2_vec << std::endl;
                // std::cout << "weight_mat_s_2.n_rows: " <<  weight_mat_s_2.n_rows << std::endl;

            } else {
                U_2_vec = reshape(ordered_Y,1,n) * weight_mat_s_2.col(i);
                U_2_1_vec = reshape(ordered_Y,1,n) * weight_mat_s_2_plus_1.col(i);
            }


            arma::vec U_vec = w_1 * U_1_vec + w_2 * U_2_vec;
            arma::vec U_vec_1 = w_1 * U_1_1_vec + w_2 * U_2_1_vec;
            // now take the average of the two estimates and use that as our final estimate
            arma::vec avg_est = (U_vec + U_vec_1)/ 2.0;
            estimates[i]=  arma::as_scalar(avg_est);
        }
    }

};

//' @export
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
arma::vec tdnn( arma::mat X, arma::vec Y, arma::mat X_test,
                  arma::vec s_sizes,
                  arma::vec s_sizes_1,
                  double c,
                  double n_prop,
                  Nullable<NumericVector> W0_ = R_NilValue){
    int d = X.n_cols;
    // Handle case where W0 is not NULL:
    if (W0_.isNotNull()){
        NumericVector W0(W0_);
        // Now we need to filter X and X_test to only contain these columns
        X = matrix_subset_logical(X, as<arma::vec>(W0));
        X_test = matrix_subset_logical(X_test, as<arma::vec>(W0));
        d = sum(W0);
    }

    // Infer n and p from our data after we've filtered for relevant features
    int n = X.n_rows;
    int p = X.n_cols;


    // This just creates a sequence 1:n and then reverses it
    NumericVector ord = seq_cpp(1, n);
    ord.attr("dim") = Dimension(n, 1);
    // don't need to reverse if we are using lfactorial
    // ord = n - ord;
    arma::vec ord_arma = as<arma::vec>(ord);

    arma::vec s_1 = s_sizes;
    arma::vec s_2 = round_modified(s_1 * pow(c, - double(d) / 2.0));

    arma::vec s_1_1 = s_sizes_1;
    arma::vec s_2_1 = round_modified(s_1_1 * pow(c, - double(d) / 2.0));
    // Rcout << "s_1: " << s_1 << std::endl;
    // Rcout << "s_1_1: " << s_1_1 << std::endl;
    //
    // Rcout << "s_2: " << s_2 << std::endl;
    // Rcout << "s_2_1: " << s_2_1 << std::endl;
    // arma::mat weight_mat_s_1 = weight_mat_lfac(int(n), ord_arma, s_1);
    // arma::mat weight_mat_s_2 = weight_mat_lfac(int(n), ord_arma, s_2);

    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(n, ord_arma, s_1, n_prop, false);
    arma::mat weight_mat_s_2 = weight_mat_lfac_s_2_filter(n, ord_arma, s_2, n_prop, true);

    arma::mat weight_mat_s_1_plus_1 = weight_mat_lfac_s_2_filter(n, ord_arma, s_1_1, n_prop, false);
    arma::mat weight_mat_s_2_plus_1 = weight_mat_lfac_s_2_filter(n, ord_arma, s_2_1, n_prop, true);

    //double choose(n, s_size);
    // estimates vector
    NumericVector estimates(X_test.n_rows);
    TdnnEstimate tdnnEstimate(X, Y,
                                  X_test,
                                  estimates,
                                  weight_mat_s_1,
                                  weight_mat_s_2,
                                  weight_mat_s_1_plus_1,
                                  weight_mat_s_2_plus_1,
                                  c, n, p, d);

    parallelFor(0, X_test.n_rows, tdnnEstimate);


    return as<arma::vec>(estimates);
}


struct De_dnnEstimate : public Worker {

    // input matrices to read from
    const arma::mat X;
    const arma::mat X_test;
    const arma::vec Y;
    const arma::mat weight_mat_s_1;
    const arma::mat weight_mat_s_2;
    const arma::vec s_1;
    const arma::vec s_2;

    // input constants
    const int n;
    const int p;
    const double c;
    const int d;

    // output matrix to write to
    RVector<double> estimates;

    // initialize from Rcpp input and output matrixes (the RMatrix class
    // can be automatically converted to from the Rcpp matrix type)
    De_dnnEstimate(const arma::mat & X, const arma::vec & Y,
                 const arma::mat & X_test,
                 NumericVector estimates,
                 const arma::mat& weight_mat_s_1,
                 const arma::mat& weight_mat_s_2,
                 const arma::vec& s_1,
                 const arma::vec& s_2,
                 double c, int n, int p, int d
                 )
        : X(X), X_test(X_test),
          Y(Y),
          weight_mat_s_1(weight_mat_s_1),
          weight_mat_s_2(weight_mat_s_2),
          s_1(s_1),
          s_2(s_2),
          n(n),
          p(p), c(c), d(d),
          estimates(estimates){}

    // function call operator that work for the specified range (begin/end)
    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            // arma::mat single_vec(int(n),1);
            // single_vec.fill(1.0);
            arma::mat all_cols(p,1);
            all_cols.fill(1.0);

            // arma::mat all_rows;
            arma::mat X_dis;
            arma::mat EuDis;

            arma::mat X_test_row =  X_test.row(i);
            // all_rows = single_vec * X_test_row;
            arma::mat all_rows = arma::repmat(X_test_row,n,1);

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
            // std::vector<double> eu_dis = conv_to<std::vector<double>>::from(vec_eu_dis);
            // std::vector<double> noise_vec = conv_to<std::vector<double>>::from(noise);
            //
            // vector<int> index(int(n), 0);
            // for (int i = 0 ; i != index.size() ; i++) {
            //     index[i] = i;
            // }
            // sort(index.begin(), index.end(),
            //      [&](const int& a, const int& b) {
            //          if (eu_dis[a] != eu_dis[b]){
            //              return eu_dis[a] < eu_dis[b];
            //          }
            //          return noise_vec[a] < noise_vec[b];
            //      }
            // );
            // arma::vec ordered_Y;
            // arma::mat ordered_Y_vec = conv_to<arma::mat>::from(Y).rows(conv_to<arma::uvec>::from(index));

            arma::vec ordered_Y;
            arma::mat ordered_Y_vec = conv_to<arma::mat>::from(Y).rows(index);
            ordered_Y = ordered_Y_vec;
            // // Rcout << ordered_Y;

            arma::vec U_1_vec(ordered_Y.n_rows);
            // arma::vec U_2_vec(ordered_Y.n_rows);
            arma::vec U_2_vec;

            double w_1 = c/(c-1);
            double w_2 = -1/(c-1);


            // the weight matrix is # train obs x # test obs so we want to use the ith column of the weight mat for the ith test observation
            U_1_vec = reshape(ordered_Y,1,n) * weight_mat_s_1.col(i);
            if(arma::accu(weight_mat_s_2.col(i)) == 0){
                // in this case s_2 is too large so we will get the 1-NN to use as the estimate
                // std::cout << "using 1-NN" << std::endl;
                arma::vec nn_1_result = get_1nn_reg(X, X_test_row, Y, 1);
                U_2_vec = arma::as_scalar(nn_1_result);
                // std::cout << "U_2_vec: " <<  U_2_vec << std::endl;
                // std::cout << "weight_mat_s_2.n_rows: " <<  weight_mat_s_2.n_rows << std::endl;

            } else {
                U_2_vec = reshape(ordered_Y,1,n) * weight_mat_s_2.col(i); // might need to convert this to mat?
            }

            arma::vec U_vec = w_1 * U_1_vec + w_2 * U_2_vec;
            // estimates.insert(i, sum(U_vec));
            estimates[i]=  arma::as_scalar(U_vec);
        }
    }

};

//' @export
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
arma::vec de_dnn( arma::mat X, arma::vec Y, arma::mat X_test,
             arma::vec s_sizes, double c,
             double n_prop,
             Nullable<NumericVector> W0_ = R_NilValue){
    int d = X.n_cols;
    // Handle case where W0 is not NULL:
    if (W0_.isNotNull()){
        NumericVector W0(W0_);
        // Now we need to filter X and X_test to only contain these columns
        X = matrix_subset_logical(X, as<arma::vec>(W0));
        X_test = matrix_subset_logical(X_test, as<arma::vec>(W0));
        d = sum(W0);
    }

    // Infer n and p from our data after we've filtered for relevant features
    int n = X.n_rows;
    int p = X.n_cols;


    // This just creates a sequence 1:n and then reverses it
    NumericVector ord = seq_cpp(1, n);
    ord.attr("dim") = Dimension(n, 1);
    // don't need to reverse if we are using lfactorial
    // ord = n - ord;
    arma::vec ord_arma = as<arma::vec>(ord);

    arma::vec s_1 = s_sizes;
    arma::vec s_2 = round_modified(s_1 * pow(c, - double(d) / 2.0));
    arma::vec tmp = s_1 * pow(c, - double(d) / 2.0);
    // Rcout << "tmp: " << tmp << std::endl;
    // Rcout << "s_2: " << s_2 << std::endl;
    // arma::mat weight_mat_s_1 = weight_mat_lfac(int(n), ord_arma, s_1);
    // arma::mat weight_mat_s_2 = weight_mat_lfac(int(n), ord_arma, s_2);

    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(n, ord_arma, s_1, n_prop, false);
    arma::mat weight_mat_s_2 = weight_mat_lfac_s_2_filter(n, ord_arma, s_2, n_prop, true);

    //double choose(n, s_size);
    // estimates vector
    NumericVector estimates(X_test.n_rows);

    De_dnnEstimate de_dnnEstimate(X, Y,
                              X_test,
                              estimates,
                              weight_mat_s_1,
                              weight_mat_s_2,
                              s_1,
                              s_2,
                              c, n, p, d);

    parallelFor(0, X_test.n_rows, de_dnnEstimate);

    // List out = List::create( Named("estimates") = estimates);
    // return out;

    return as<arma::vec>(estimates);
}



// [[Rcpp::export]]
NumericVector tuning(arma::mat X, arma::vec Y,
                            arma::mat X_test, double c,
                            double n_prop,
                            Nullable<NumericVector> W0_ = R_NilValue){

    double n_obs = X_test.n_rows;
    bool search_for_s = true;
    arma::mat tuning_mat(n_obs, 100, fill::zeros);
    arma::vec best_s(n_obs, fill::zeros);
    double s = 0;
    // using zero indexing here to match with C++, note s + 1 -> s+2 in de_dnn call

    while (search_for_s){
        // s_val needs to be a vector of the same length as X_test
        double s_fill = s + 2;
        NumericVector s_val(int(n_obs), s_fill);
        // s_val = s + 2;

        // For a given s, get the de_dnn estimates for each test observation
        // List de_dnn_estimates = de_dnn(X, Y, X_test, s_val, c, W0_);
        arma::vec de_dnn_estimates = de_dnn(X, Y, X_test, s_val, c, n_prop, W0_);

        // This gives me an estimate for each test observation and is a n x 1 matrix
        // arma::vec de_dnn_est_vec = as<arma::vec>(de_dnn_estimates["estimates"]);
        arma::mat candidate_results = de_dnn_estimates;
        candidate_results.reshape(n_obs, 1);



        // Now we add this column to our matrix if the matrix is empty
        if (s == 0 | s == 1){
            tuning_mat.col(s) = candidate_results;
        } else if (s >= tuning_mat.n_cols){
            // if s > ncol tuning_mat, then we will choose best s from the existing choices for each row that hasn't found a best s yet and break out of the while loop
            arma::uvec s_vec = seq_int(0, int(s)-1);
            arma::mat resized_mat = matrix_subset_idx(tuning_mat, s_vec);

            arma::mat out_diff = diff(resized_mat, 1, 1);
            IntegerVector idx = Range(0, (resized_mat.n_cols)-2);
            arma::mat out_denom = resized_mat.cols(as<uvec>(idx));
            arma::mat diff_ratio = diff(abs( out_diff / out_denom), 1, 1);

            for(R_xlen_t i = 0; i < diff_ratio.n_rows; ++i) {
                // Only loop through the columns if we haven't already found a
                // suitable s
                if(best_s(i) == 0){
                    for(R_xlen_t j = 0; j < diff_ratio.n_cols; ++j) {
                        if (diff_ratio(i, j) > -0.01){
                            best_s(i) = j + 1 + 3;
                            break; // if we've found the column that satisfies our condition, break and move to next row.
                        }
                    }
                }
            }
            search_for_s = false; // since we've gone past the num of columns stop the while loop here
            break; // break out of our while loop to avoid going past number of columns in tuning_mat
        } else {

            // instead of resizing the matrix, just select columns 0-s
            arma::uvec s_vec = seq_int(0, int(s)-1);
            arma::mat resized_mat = matrix_subset_idx(tuning_mat, s_vec);
            // tuning_mat is an n x s matrix and we want to diff each of the rows
            arma::mat out_diff = diff(resized_mat, 1, 1);
            IntegerVector idx = Range(0, (resized_mat.n_cols)-2);
            arma::mat out_denom = resized_mat.cols(as<uvec>(idx));
            arma::mat diff_ratio = diff(abs( out_diff / out_denom), 1, 1);
            // Now we go through each row and check if any of the columns are
            // greater than -0.01
            for(R_xlen_t i = 0; i < diff_ratio.n_rows; ++i) {
                // Only loop through the columns if we haven't already found a
                // suitable s
                if(best_s(i) == 0){
                    for(R_xlen_t j = 0; j < diff_ratio.n_cols; ++j) {
                        if (diff_ratio(i, j) > -0.01){
                            best_s(i) = j + 1 + 3;
                            break; // if we've found the column that satisfies our condition, break and move to next row.
                        }
                    }
                }
            }

            // Check if we still have observations without an s
            if (all(best_s)){
                // then we are done!
                search_for_s = false;
            } else {
                tuning_mat.col(s) = candidate_results;
            }

        }
        s += 1;
    }

    return NumericVector(best_s.begin(), best_s.end());
    // return best_s;
}


// [[Rcpp::export]]
List tuning_est(arma::mat X, arma::vec Y,
                     arma::mat X_test, double c,
                     double n_prop,
                     Nullable<NumericVector> W0_ = R_NilValue){

    double n_obs = X_test.n_rows;
    bool search_for_s = true;
    arma::mat tuning_mat(n_obs, 100, fill::zeros);
    arma::vec best_s(n_obs, fill::zeros);
    double s = 0;
    // using zero indexing here to match with C++, note s + 1 -> s+2 in de_dnn call

    while (search_for_s){
        // s_val needs to be a vector of the same length as X_test
        double s_fill = s + 2;
        NumericVector s_val(int(n_obs), s_fill);
        // s_val = s + 2;

        // For a given s, get the de_dnn estimates for each test observation
        // List de_dnn_estimates = de_dnn(X, Y, X_test, s_val, c, W0_);
        arma::vec de_dnn_estimates = de_dnn(X, Y, X_test, s_val, c, n_prop, W0_);

        // This gives me an estimate for each test observation and is a n x 1 matrix
        // arma::vec de_dnn_est_vec = as<arma::vec>(de_dnn_estimates["estimates"]);
        arma::mat candidate_results = de_dnn_estimates;
        candidate_results.reshape(n_obs, 1);



        // Now we add this column to our matrix if the matrix is empty
        if (s == 0 | s == 1){
            tuning_mat.col(s) = candidate_results;
        } else if (s >= tuning_mat.n_cols){
            // if s > ncol tuning_mat, then we will choose best s from the existing choices for each row that hasn't found a best s yet and break out of the while loop
            arma::uvec s_vec = seq_int(0, int(s)-1);
            arma::mat resized_mat = matrix_subset_idx(tuning_mat, s_vec);

            arma::mat out_diff = diff(resized_mat, 1, 1);
            IntegerVector idx = Range(0, (resized_mat.n_cols)-2);
            arma::mat out_denom = resized_mat.cols(as<uvec>(idx));
            arma::mat diff_ratio = diff(abs( out_diff / out_denom), 1, 1);

            for(R_xlen_t i = 0; i < diff_ratio.n_rows; ++i) {
                // Only loop through the columns if we haven't already found a
                // suitable s
                if(best_s(i) == 0){
                    for(R_xlen_t j = 0; j < diff_ratio.n_cols; ++j) {
                        if (diff_ratio(i, j) > -0.01){
                            best_s(i) = j + 1 + 3;
                            break; // if we've found the column that satisfies our condition, break and move to next row.
                        }
                    }
                }
            }
            search_for_s = false; // since we've gone past the num of columns stop the while loop here
            break; // break out of our while loop to avoid going past number of columns in tuning_mat
        } else {

            // instead of resizing the matrix, just select columns 0-s
            arma::uvec s_vec = seq_int(0, int(s)-1);
            arma::mat resized_mat = matrix_subset_idx(tuning_mat, s_vec);
            // tuning_mat is an n x s matrix and we want to diff each of the rows
            arma::mat out_diff = diff(resized_mat, 1, 1);
            IntegerVector idx = Range(0, (resized_mat.n_cols)-2);
            arma::mat out_denom = resized_mat.cols(as<uvec>(idx));
            arma::mat diff_ratio = diff(abs( out_diff / out_denom), 1, 1);
            // Now we go through each row and check if any of the columns are
            // greater than -0.01
            for(R_xlen_t i = 0; i < diff_ratio.n_rows; ++i) {
                // Only loop through the columns if we haven't already found a
                // suitable s
                if(best_s(i) == 0){
                    for(R_xlen_t j = 0; j < diff_ratio.n_cols; ++j) {
                        if (diff_ratio(i, j) > -0.01){
                            best_s(i) = j + 1 + 3;
                            break; // if we've found the column that satisfies our condition, break and move to next row.
                        }
                    }
                }
            }

            // Check if we still have observations without an s
            if (all(best_s)){
                // then we are done!
                search_for_s = false;
            } else {
                tuning_mat.col(s) = candidate_results;
            }

        }
        s += 1;
    }
    // we can work backwards from best_s to get the corresponding estimates column in
    // tuning mat by subtracting 2 from each value
    arma::vec best_col = best_s - 2;
    arma::uvec estimate_col_idx = conv_to<arma::uvec>::from(best_col);
    arma::uvec estimate_row_idx = seq_int(0, X_test.n_rows-1);

    // need matrix subset where we get the individual elements for each row
    // so if the first row had s=4 then we need the first element of the column of tuning
    // mat that corresponds to s = 4
    arma::vec tuned_estimates = select_mat_elements(tuning_mat, estimate_row_idx, estimate_col_idx);
    // now we need the s+1 estimates for each of these values
    arma::vec tuned_1_estimates = de_dnn(X, Y, X_test, best_s + 1, c, n_prop, W0_);


    arma::vec final_est = (tuned_estimates + tuned_1_estimates) / 2;
    return(List::create(Named("estimates") = final_est, Named("s") =NumericVector(best_s.begin(),best_s.end())));

    // return NumericVector(best_s.begin(), best_s.end());
    // return best_s;
}

// [[Rcpp::export]]
arma::mat bootstrap_reg_fn(const arma::mat& X, const arma::mat& Y,
                        const arma::mat& X_test,
                        arma::vec s_choice,
                        double c,
                        double n_prop,
                        Nullable<NumericVector> W_0 = R_NilValue,
                        int B =1000,
                        bool verbose = false) {

    NumericVector W0;
    if (W_0.isNotNull()){
        W0 = W_0;
    } else{
        W0 = rep(1,X.n_cols);
    }
    // Pre-allocate storage for statistics
    int n_test = X_test.n_rows;
    arma::mat boot_stat(n_test,B);
    int n = X.n_rows;

    // Perform bootstrap
    for(int i =0; i < B; i++) {
        if(verbose){
            Rcout << "Bootstrap iteration " << i+1 << "/" << B << std::endl;
        }

        // get bootstrap indices
        NumericVector bstrap_idx = floor(runif(n,0, n));
        arma::uvec bstrap_idx_arma = as<arma::uvec>(bstrap_idx);

        arma::mat X_boot = matrix_row_subset_idx(X, bstrap_idx_arma);
        arma::mat Y_boot = matrix_row_subset_idx(Y, bstrap_idx_arma);

        arma::vec reg_est = tdnn(X_boot, Y_boot, X_test, s_choice,
                                 s_choice + 1,
                                 c, n_prop, W0);

        boot_stat.col(i) = reg_est;
    }
    // Rcout << "boot_stat: " << boot_stat << std::endl;
    return boot_stat;
}



// [[Rcpp::export]]
arma::mat bootstrap_cpp(const arma::mat& X, const arma::mat& Y,
                        const arma::vec& W,
                        const arma::mat& X_test,
                        arma::vec s_choice_0,
                        arma::vec s_choice_1,
                        double c,
                        double n_prop,
                        Nullable<NumericVector> W_0 = R_NilValue,
                        int B =1000,
                        bool verbose = false) {

    NumericVector W0;
    if (W_0.isNotNull()){
        W0 = W_0;
    } else{
        W0 = rep(1,X.n_cols);
    }
    // Pre-allocate storage for statistics
    int n_test = X_test.n_rows;
    arma::mat boot_stat(n_test,B);
    // int n = X.n_rows;

    // Perform bootstrap
    for(int i =0; i < B; i++) {
        if(verbose){
            Rcout << "Bootstrap iteration " << i+1 << "/" << B << std::endl;
        }

        // separate data into treatment and control groups and then bootstrap within each group
        arma::uvec trt_idx = arma::find(W == 1);
        arma::uvec ctl_idx = arma::find(W == 0);

        NumericVector trt_bstrap_idx = floor(runif(trt_idx.size(),0, trt_idx.size()));
        arma::uvec trt_bstrap_idx_arma = as<arma::uvec>(trt_bstrap_idx);

        NumericVector ctl_bstrap_idx = floor(runif(ctl_idx.size(),0, ctl_idx.size()));
        arma::uvec ctl_bstrap_idx_arma = as<arma::uvec>(ctl_bstrap_idx);

        arma::mat X_trt_boot = X.rows(trt_bstrap_idx_arma);
        arma::mat Y_trt_boot = Y.rows(trt_bstrap_idx_arma);

        arma::mat X_ctl_boot = X.rows(ctl_bstrap_idx_arma);
        arma::mat Y_ctl_boot = Y.rows(ctl_bstrap_idx_arma);


        // NumericVector bstrap_idx = floor(runif(n,0, n));
        // arma::uvec bstrap_idx_arma = as<arma::uvec>(bstrap_idx);
        // arma::mat X_boot = matrix_row_subset_idx(X, bstrap_idx_arma);
        // arma::mat Y_boot = matrix_row_subset_idx(Y, bstrap_idx_arma);
        // arma::vec W_boot = vector_subset_idx(W, bstrap_idx_arma);
        //
        // arma::uvec trt_idx = arma::find(W_boot == 1);
        // arma::uvec ctl_idx = arma::find(W_boot == 0);
        // Rcout << "trt_idx: " << trt_idx << std::endl;
        // Rcout << "ctl_idx: " << ctl_idx << std::endl;
        //
        // arma::mat X_trt = X.rows(trt_idx);
        // arma::mat Y_trt = Y.rows(trt_idx);
        //
        // arma::mat X_ctl = X.rows(ctl_idx);
        // arma::mat Y_ctl = Y.rows(ctl_idx);
        //
        // Rcout << "X_trt: " << X_trt << std::endl;
        // Rcout << "X_ctl: " << X_ctl << std::endl;


        arma::vec trt_est = tdnn(X_trt_boot, Y_trt_boot, X_test, s_choice_1,
                                 s_choice_1 + 1,
                          c, n_prop, W0);

        arma::vec ctl_est = tdnn(X_ctl_boot, Y_ctl_boot, X_test, s_choice_0,
                                 s_choice_0 + 1,
                                 c, n_prop, W0);
        arma::vec diff = trt_est - ctl_est;
        // Rcout << "diff: " << diff << std::endl;
        boot_stat.col(i) = diff;
        }
    // Rcout << "boot_stat: " << boot_stat << std::endl;
    return boot_stat;
}

// [[Rcpp::export]]
List est_reg_fn_mt_rcpp(const arma::mat& X, const arma::mat& Y,
                        const arma::mat& X_test,
                        double c,
                        double n_prop,
                        bool verbose = false,
                        bool old = false,
                        Nullable<NumericVector> W0_ = R_NilValue){
    // This function is called after we've done data checks in R
    NumericVector W0;
    if (W0_.isNotNull()){
        W0 = W0_;
    } else{
        W0 = rep(1,X.n_cols);
    }
    if(verbose){
        Rcout << "starting tuning" << std::endl;
    }
    NumericVector s_sizes = tuning(X, Y, X_test, c, n_prop, W0);
    if(verbose){
        Rcout << "past tuning" << std::endl;
    }

    arma::vec deDNN_pred;

    if(old){
        arma::vec a_pred = de_dnn(X, Y, X_test, s_sizes,
                                  c, n_prop, W0);
        if(verbose){
            Rcout << "past first pred" << std::endl;
        }
        arma::vec b_pred = de_dnn(X, Y, X_test, s_sizes + 1,
                                  c,n_prop, W0);
        if(verbose){
            Rcout << "past second pred" << std::endl;
        }
        deDNN_pred = (a_pred + b_pred) / 2;
    } else {
        deDNN_pred = tdnn(X, Y, X_test, s_sizes,
                          s_sizes + 1,
                          c, n_prop, W0);
    }


    return(List::create( Named("estimates") = deDNN_pred,
                         Named("s") = s_sizes));
}

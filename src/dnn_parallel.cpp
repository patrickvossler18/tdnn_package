#include "util.h"

struct TdnnEstimate : public Worker {

    // input matrices to read from
    const arma::mat X;
    const arma::mat X_test;
    const NumericVector s_sizes;
    const NumericVector ord;
    const arma::vec Y;

    // input constants
    const double n;
    const double p;
    const double c;
    const int d;

    // output matrix to write to
    RVector<double> estimates;

    // initialize from Rcpp input and output matrixes (the RMatrix class
    // can be automatically converted to from the Rcpp matrix type)
    TdnnEstimate(const arma::mat & X, const arma::vec & Y,
                 const arma::mat & X_test,
                 NumericVector estimates,
                 const NumericVector s_sizes,
                 const NumericVector ord,
                 double c, double n, double p, int d
                 )
        : X(X), X_test(X_test),
          s_sizes(s_sizes), ord(ord), Y(Y), n(n),
          p(p), c(c),estimates(estimates), d(d){}

    // function call operator that work for the specified range (begin/end)
    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            arma::mat single_vec(int(n),1);
            single_vec.fill(1.0);
            arma::mat all_cols(int(p),1);
            all_cols.fill(1.0);

            arma::mat all_rows;
            arma::mat X_dis;
            arma::mat EuDis;

            arma::mat X_test_row =  X_test.row(i);
            all_rows = single_vec * X_test_row;

            X_dis = X - all_rows;

            EuDis = (pow(X_dis, 2)) * all_cols;

            // arma::mat noise(int(n), 1);
            // double noise_val = R::rnorm(0, 1);
            // noise.fill(noise_val);
            // arma::vec noise = arma::randn<vec>(int(n));
            arma::vec noise(n);
            double noise_val = arma::randn<double>();
            noise.fill(noise_val);

            arma::vec vec_eu_dis = conv_to<arma::vec>::from(EuDis);
            std::vector<double> eu_dis = conv_to<std::vector<double>>::from(vec_eu_dis);
            std::vector<double> noise_vec = conv_to<std::vector<double>>::from(noise);

            vector<int> index(int(n), 0);
            for (int i = 0 ; i != index.size() ; i++) {
                index[i] = i;
            }
            sort(index.begin(), index.end(),
                 [&](const int& a, const int& b) {
                     if (eu_dis[a] != eu_dis[b]){
                         return eu_dis[a] < eu_dis[b];
                     }
                     return noise_vec[a] < noise_vec[b];
                 }
            );


            arma::vec ordered_Y;
            arma::mat ordered_Y_vec = conv_to<arma::mat>::from(Y).rows(conv_to<arma::uvec>::from(index));
            // // Rcout << ordered_Y_vec[order_vec];
            ordered_Y = ordered_Y_vec;
            // // Rcout << ordered_Y;

            // TempD = data.frame(EuDis, Y, noise)[order(EuDis, noise), ]
            arma::vec U_1_vec;
            arma::vec U_2_vec;
            rowvec weight_vec;

            double w_1 = c/(c-1);
            double w_2 = -1/(c-1);
            double s_1 = s_sizes(i);
            double s_2 = round(s_1 * pow(c, - double(d) / 2.0));

            // Weight vectors
            arma::vec weight_1(ord.length());
            for (int j = 0 ; j < ord.length() ; j++) {
                weight_1(j) = nChoosek(ord(j), s_1 - 1.0);
            }
            weight_1 /=  nChoosek(n, s_1);
            weight_1.reshape(int(n), 1);

            arma::vec weight_2(ord.length());
            for (int k = 0 ; k < ord.length() ; k++) {
                weight_2(k) = nChoosek(ord(k), (s_2 - 1.0));
            }
            weight_2 /= nChoosek(n, s_2);
            weight_2.reshape(int(n), 1);

            U_1_vec = reshape(ordered_Y,1,int(n)) * conv_to<arma::mat>::from(weight_1);
            U_2_vec = reshape(ordered_Y,1,int(n)) * conv_to<arma::mat>::from(weight_2);

            arma::vec U_vec = w_1 * U_1_vec + w_2 * U_2_vec;
            // estimates.insert(i, sum(U_vec));
            estimates[i]=  sum(U_vec);
        }
    }

};

//' @export
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
List de_dnn( arma::mat X, arma::vec Y, arma::mat X_test,
             NumericVector s_sizes, double c,
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
    double n = X.n_rows;
    double p = X.n_cols;


    // This just creates a sequence 1:n and then reverses it
    NumericVector ord = seq_cpp(1, n);
    ord.attr("dim") = Dimension(n, 1);
    ord = n - ord;

    //double choose(n, s_size);
    // estimates vector
    NumericVector estimates(X_test.n_rows);
    // NumericMatrix estimates(X_test.n_rows, 1); // change to matrix
    // NumericMatrix weight_mat(X_test.n_rows, int(n));

    TdnnEstimate tdnnEstimate(X, Y,
                              X_test,
                              estimates,
                              s_sizes,
                              ord,
                              c, n, p, d);

    parallelFor(0, X_test.n_rows, tdnnEstimate);

    List out = List::create( Named("estimates") = estimates);

    return out;
}



// [[Rcpp::export]]
NumericVector tuning(NumericMatrix X, NumericVector Y,
                            NumericMatrix X_test, double c,
                            Nullable<NumericVector> W0_ = R_NilValue){

    double n_obs = X_test.nrow();
    bool search_for_s = true;
    arma::mat tuning_mat(n_obs, 100, fill::zeros);
    arma::vec best_s(n_obs, fill::zeros);
    double s = 0;
    // using zero indexing here to match with C++, note s + 1 -> s+2 in de_dnn call

    while (search_for_s){
        NumericVector s_val;
        s_val = s + 2;
        // For a given s, get the de_dnn estimates for each test observation
        List de_dnn_estimates = de_dnn(as<arma::mat>(X),
                                                as<arma::vec>(Y),
                                                as<arma::mat>(X_test),
                                                s_val, c, W0_);

        // This gives me an estimate for each test observation and is a n x 1 matrix
        arma::vec de_dnn_est_vec = as<arma::vec>(de_dnn_estimates["estimates"]);
        arma::mat candidate_results = de_dnn_est_vec;
        candidate_results.reshape(n_obs, 1);



        // Now we add this column to our matrix if the matrix is empty
        if (s == 0 | s == 1){
            tuning_mat.col(s) = candidate_results;
        } else if (s > tuning_mat.n_cols){
            Rcout << "s s > tuning_mat.n_cols" << std::endl;
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
NumericMatrix bootstrap_cpp(NumericMatrix X,
                            NumericMatrix X_test,
                            NumericMatrix Y,
                            IntegerVector W,
                            NumericVector W0,
                            NumericVector s_choice_0,
                            NumericVector s_choice_1,
                            double c=0.33,
                            int B =1000) {
    // Preallocate storage for statistics
    int n_test = X_test.nrow();
    NumericMatrix boot_stat(B,n_test);
    int n = X.nrow();

    // Perform bootstrap
    for(int i =0; i < B; i++) {
        NumericVector bstrap_idx = floor(runif(n,0, n));
        NumericMatrix X_boot = matrix_subset_idx_rcpp(X, as<IntegerVector>(bstrap_idx));
        IntegerVector W_boot = W[as<IntegerVector>(bstrap_idx)];
        NumericMatrix Y_boot = matrix_subset_idx_rcpp(Y,as<IntegerVector>(bstrap_idx));

        // // Sample initial data
        LogicalVector trt_idx = W_boot == 1;
        LogicalVector ctl_idx = W_boot == 0;

        arma::mat X_trt = matrix_subset_logical(as<arma::mat>(X_boot),as<arma::vec>(as<IntegerVector>(trt_idx)), 2);
        arma::mat Y_trt = matrix_subset_logical(as<arma::mat>(Y_boot),as<arma::vec>(as<IntegerVector>(trt_idx)), 2);

        arma::mat X_ctl = matrix_subset_logical(as<arma::mat>(X_boot),as<arma::vec>(as<IntegerVector>(ctl_idx)), 2);
        arma::mat Y_ctl = matrix_subset_logical(as<arma::mat>(Y_boot),as<arma::vec>(as<IntegerVector>(ctl_idx)), 2);

        NumericVector trt_est_a = de_dnn(X_trt,
                              conv_to<arma::vec>::from(Y_trt),
                              as<arma::mat>(X_test),
                              s_choice_1, c, W0)["estimates"];
        NumericVector s_choice_1_p1 = s_choice_1 + 1.0;
        NumericVector trt_est_b = de_dnn(X_trt,
                              conv_to<arma::vec>::from(Y_trt),
                              as<arma::mat>(X_test),
                              s_choice_1_p1, c, W0)["estimates"];
        NumericVector trt_est = (trt_est_a + trt_est_b) / 2.0;

        NumericVector ctl_est_a = de_dnn(X_ctl,
                                         conv_to<arma::vec>::from(Y_ctl),
                                         as<arma::mat>(X_test),
                                         s_choice_0, c, W0)["estimates"];
        NumericVector s_choice_0_p1 = s_choice_0 + 1.0;
        NumericVector ctl_est_b = de_dnn(X_ctl,
                                         conv_to<arma::vec>::from(Y_ctl),
                                         as<arma::mat>(X_test),
                                         s_choice_0_p1, c, W0)["estimates"];
        NumericVector ctl_est = (ctl_est_a + ctl_est_b) / 2.0;

        boot_stat(i, _) = trt_est - ctl_est;
        }
    return boot_stat;
}

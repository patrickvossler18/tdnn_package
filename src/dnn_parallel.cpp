// #include <RcppParallel.h>
// #include <math.h>// #include <RcppParallel.h>
// #include <math.h>
#include "util.h"

// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::depends(RcppArmadillo)]]
// using namespace Rcpp;
// using namespace RcppParallel;
// using namespace arma;
// using namespace std;

// double nChoosek(double n, double k)
// {
//     if (k == 0) return 1.0;
//
//     /*
//      Extra computation saving for large R,
//      using property:
//      N choose R = N choose (N-R)
//      */
//     // if (k > n / 2.0) return nChoosek(n, n - k);
//
//     double res = 1.0;
//
//     for (double i = 1.0; i <= k; ++i)
//     {
//         res *= n - i + 1.0;
//         res /= i;
//     }
//
//     return res;
// }
//
// NumericVector seq_cpp(double lo, double hi) {
//     double n = hi - lo + 1;
//
//     // Create a new integer vector, sequence, of size n
//     NumericVector sequence(n);
//
//     for(int i = 0; i < n; i++) {
//         // Set the ith element of sequence to lo plus i
//         sequence[i] = lo + i;
//     }
//
//     // Return
//     return sequence;
// }


// arma::mat matrix_subset_logical(arma::mat x,
//                                 arma::vec y) {
//     // Assumes that y is 0/1 coded.
//     // find() retrieves the integer index when y is equivalent 1.
//     uvec subset_vec = find(y == 1) ;
//     return x.cols(find(y == 1) );
// }


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
    TdnnEstimate(const arma::mat X, const arma::vec Y,
                 const arma::mat X_test,
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

            // arma::mat X_test_row =  as<arma::mat>(NumericMatrix(1, X_test.ncol(),X_test(i,_).begin()));
            arma::mat X_test_row =  X_test.row(i);
            // Rcout << "i: " << i << " row: "<< X_test_row << "\n";
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
            // arma::vec weight_1(ord.length());
            // for (int j = 0 ; j < ord.length() ; j++) {
            //     weight_1(j) = nChoosek(ord(j), s_sizes(i) - 1.0);
            // }
            // weight_1 /=  nChoosek(n, s_sizes(i));
            // weight_1.reshape(int(n), 1);
            // arma::vec weight_2(ord.length());
            // for (int k = 0 ; k < ord.length() ; k++) {
            //     weight_2(k) = nChoosek(ord(k), ((s_sizes(i) * bc_p) - 1.0));
            // }
            // weight_2 /= nChoosek(n, (s_sizes(i) * bc_p));
            // weight_2.reshape(int(n), 1);
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


            // U_1_vec = reshape(ordered_Y,1,int(n)) * weight_1;
            // U_2_vec = reshape(ordered_Y,1,int(n)) * weight_2;

            U_1_vec = reshape(ordered_Y,1,int(n)) * conv_to<arma::mat>::from(weight_1);
            U_2_vec = reshape(ordered_Y,1,int(n)) * conv_to<arma::mat>::from(weight_2);


            // NumericVector weight_1 = choose(ord, s_sizes(i) - 1.0) / nChoosek(n, s_sizes(i));
            // weight_1.attr("dim") = Dimension(n,1);
            //
            // NumericVector weight_2 = choose(ord, (s_sizes(i) * bc_p) - 1.0) / nChoosek(n, (s_sizes(i) * bc_p));
            // weight_2.attr("dim") = Dimension(n,1);
            //
            // U_1_vec = reshape(ordered_Y,1,int(n)) * as<arma::mat>(weight_1);
            // U_2_vec = reshape(ordered_Y,1,int(n)) * as<arma::mat>(weight_2);


            // weight_vec = (reshape(ordered_Y, int(n), 1) % as<arma::mat>(weight_1)).t();

            // arma::mat A_mat(arma::vec{1, 1, 1, pow((1 / bc_p),(2 /   std::min(p, 3.0)) ) });
            // A_mat.reshape(2,2);
            // arma::mat A_mat_inv = A_mat.i();
            //
            //
            // arma::mat B_mat(arma::vec{1, 0});
            // B_mat.reshape(2, 1);
            //
            // arma::mat Coefs = A_mat_inv * B_mat;
            //
            // arma::vec U_vec = Coefs(0,0) * U_1_vec + Coefs(1,0) * U_2_vec;
            arma::vec U_vec = w_1 * U_1_vec + w_2 * U_2_vec;
            // estimates.insert(i, sum(U_vec));
            estimates[i]=  sum(U_vec);
            // estimates.row(i) =  NumericVector(sum(U_vec));
            // weight_mat.row(i) = NumericVector(weight_vec.begin(), weight_vec.end());
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
    // make tuning matrix large and then resize it?
    arma::mat tuning_mat(n_obs, 50, fill::zeros);
    arma::vec best_s(n_obs, fill::zeros);
    double s = 0;
    // using zero indexing here to match with C++, note s + 1 -> s+2 in de_dnn call

    while (search_for_s){
        NumericVector s_val;
        s_val = s + 2;
        // Rcout << "s_val: " << s_val << std::endl;
        // For a given s, get the de_dnn estimates for each test observation
        List de_dnn_estimates = de_dnn(as<arma::mat>(X),
                                                as<arma::vec>(Y),
                                                as<arma::mat>(X_test),
                                                s_val, c, W0_);

        // This gives me an estimate for each test observation and is a n x 1 matrix
        arma::vec de_dnn_est_vec = as<arma::vec>(de_dnn_estimates["estimates"]);
        arma::mat candidate_results = de_dnn_est_vec;
        // arma::mat candidate_results(n_obs, 1 );
        candidate_results.reshape(n_obs, 1);
        // candidate_results.col(0) = de_dnn_est_vec;
        // Rcout << "got candidate estimates: " << candidate_results << std::endl;
        // Might need to reshape this matrix?

        // Now we add this column to our matrix if the matrix is empty
        if (s == 0 | s == 1){
            tuning_mat.col(s) = candidate_results;
        } else {
            // Before we do anything with the matrix we need to resize it to avoid
            // dividing by zero
            arma::mat resized_mat = tuning_mat;
            //
            //             IntegerVector non_zero_idx = Range(0, (s-1));
            //             Rcout << "non_zero_idx: " << non_zero_idx << std::endl;
            //             arma::mat resized_mat = tuning_mat.cols(as<uvec>(non_zero_idx));
            //
            resized_mat.resize(n_obs, s);
            // Rcout << "resized mat: " << resized_mat << std::endl;
            // tuning_mat is an n x s matrix and we want to diff each of the rows
            // arma::mat mat_row = estimate_matrix.row(i);
            arma::mat out_diff = diff(resized_mat, 1, 1);
            // if (s == 1){
            //     IntegerVector idx = Range(0, (resized_mat.n_cols)-2);
            // } else {
            //     IntegerVector idx = Range(0, (resized_mat.n_cols)-2);
            // }
            IntegerVector idx = Range(0, (resized_mat.n_cols)-2);
            // Rcout << "idx: " << idx << std::endl;
            arma::mat out_denom = resized_mat.cols(as<uvec>(idx));
            // Rcout << "out_denom: " << out_denom << std::endl;
            arma::mat diff_ratio = diff(abs( out_diff / out_denom), 1, 1);
            // Rcout << "diff_ratio: " << diff_ratio << std::endl;
            // Now we go through each row and check if any of the columns are
            // greater than -0.01
            for(R_xlen_t i = 0; i < diff_ratio.n_rows; ++i) {
                // Only loop through the columns if we haven't already found a
                // suitable s
                if(best_s(i) == 0){
                    for(R_xlen_t j = 0; j < diff_ratio.n_cols; ++j) {
                        if (diff_ratio(i, j) > -0.01){
                            best_s(i) = j + 1 + 3; // is this the correct indexing soln?
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
        // Rcout << "tuning_mat : " << tuning_mat << std::endl;
        s += 1;
    }
    return NumericVector(best_s.begin(), best_s.end());
    // return best_s;
}

#include <RcppArmadillo.h>
#include <math.h>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;
using namespace std;



// [[Rcpp::export]]
arma::mat matrix_subset_logical(arma::mat x,
                          arma::vec y) {
    // Assumes that y is 0/1 coded.
    // find() retrieves the integer index when y is equivalent 1.
    uvec subset_vec = find(y == 1) ;
    return x.cols(find(y == 1) );
}


double nChoosek(double n, double k)
{
    if (k == 0) return 1.0;

    /*
     Extra computation saving for large R,
     using property:
     N choose R = N choose (N-R)
     */
    if (k > n / 2.0) return nChoosek(n, n - k);

    double res = 1.0;

    for (double i = 1.0; i <= k; ++i)
    {
        res *= n - i + 1.0;
        res /= i;
    }

    return res;
}


NumericVector seq_cpp(double lo, double hi) {
    double n = hi - lo + 1;

    // Create a new integer vector, sequence, of size n
    NumericVector sequence(n);

    for(int i = 0; i < n; i++) {
        // Set the ith element of sequence to lo plus i
        sequence[i] = lo + i;
    }

    // Return
    return sequence;
}



NumericVector dnn( NumericMatrix X, NumericVector Y, NumericMatrix X_test,
                   double n, double p, double s_size){
    NumericVector ord = seq_cpp(1, n);
    ord.attr("dim") = Dimension(n, 1);
    ord = n - ord;
    //double choose(n, s_size);
    // Rcout << nChoosek(n, s_size);
    NumericVector weight = choose(ord, s_size - 1.0) / nChoosek(n, s_size);
    weight.attr("dim") = Dimension(n,1);
    NumericVector estimates;
    for (int i = 0; i < X_test.nrow(); i++){
        arma::mat single_vec(int(n),1);
        single_vec.fill(1.0);
        arma::mat all_cols(int(p),1);
        all_cols.fill(1.0);

        arma::mat all_rows;
        arma::mat X_dis;
        arma::mat EuDis;


        arma::mat X_test_row =  as<arma::mat>(NumericMatrix(1, X_test.ncol(),X_test(i,_).begin()));
        // Not an issue with X_test
        // Rcout << "i: " << i << " row: "<< X_test_row << "\n";
        all_rows = single_vec * X_test_row;

        X_dis = as<arma::mat>(X) - all_rows;

        EuDis = (pow(X_dis, 2)) * all_cols;

        arma::mat noise(int(n), 1);
        double noise_val = R::rnorm(0, 1);
        noise.fill(noise_val);
        // not an issue with the noise values
        // Rcout << "i: " << i << " noise: " << noise_val << "\n";

        Function f("order");
        IntegerVector order_vec = f(EuDis, noise);
        order_vec = order_vec - 1;
        arma::mat ordered_Y(1, int(n));
        // Rcout << as<NumericVector>(Y)[order_vec];
        // arma::vec ordered_Y;
        // ordered_Y = as<arma::mat>(Y[order_vec]);
        arma::mat ordered_Y_vec = as<arma::mat>(clone(Y)).rows(as<uvec>(order_vec));
        // Rcout << ordered_Y_vec[order_vec];
        ordered_Y = ordered_Y_vec;
        // Rcout << ordered_Y;

        // TempD = data.frame(EuDis, Y, noise)[order(EuDis, noise), ]
        arma::vec U_vec;

        U_vec = reshape(ordered_Y,1,int(n)) * as<arma::mat>(weight);
        // Rcout << "Shape of Y: " << size(ordered_Y) << "\n";
        // Rcout << "Shape of weights: " << size(as<arma::mat>(weight)) << "\n";
        // Rcout << "Shape of U_vec: " << size(U_vec) << "\n";
        // Rcout << "i: " << i << " estimate: " << sum(U_vec) << "\n";
        estimates.insert(i,sum(U_vec));
        // U = sum(TempD$Y * weight)
        //     return(U)

    }
    // Rcout << "Estimates: " << estimates;
    return estimates;
}

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
List de_dnn( arma::mat X, NumericVector Y, arma::mat X_test,
                      double s_size, double bc_p,
                      Nullable<NumericVector> W0_ = R_NilValue){
    // Handle case where W0 is not NULL:
    if (W0_.isNotNull()){
        NumericVector W0(W0_);
        // Now we need to filter X and X_test to only contain these columns
        X = matrix_subset_logical(X, as<arma::vec>(W0));
        X_test = matrix_subset_logical(X_test, as<arma::vec>(W0));
    }

    // Infer n and p from our data after we've filtered for relevant features
    double n = X.n_rows;
    double p = X.n_cols;


    // This just creates a sequence 1:n and then reverses it
    NumericVector ord = seq_cpp(1, n);
    ord.attr("dim") = Dimension(n, 1);
    ord = n - ord;

    //double choose(n, s_size);
    // Rcout << nChoosek(n, s_size);

    // Weight vectors
    NumericVector weight_1 = choose(ord, s_size - 1.0) / nChoosek(n, s_size);
    weight_1.attr("dim") = Dimension(n,1);

    NumericVector weight_2 = choose(ord, (s_size * bc_p) - 1.0) / nChoosek(n, (s_size * bc_p));
    weight_2.attr("dim") = Dimension(n,1);

    // estimates vector
    NumericVector estimates;
    // NumericMatrix estimates(X_test.n_rows, 1); // change to matrix
    NumericMatrix weight_mat(X_test.n_rows, int(n));

    // Go through each X_test observation
    for (int i = 0; i < X_test.n_rows; i++){
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

        arma::mat noise(int(n), 1);
        double noise_val = R::rnorm(0, 1);
        noise.fill(noise_val);

        // TO-DO: Rewrite the order vector using std::sort and a custom comparison function
        Function f("order");
        IntegerVector order_vec = f(EuDis, noise);
        order_vec = order_vec - 1;
        arma::mat ordered_Y(1, int(n));
        // Rcout << as<NumericVector>(Y)[order_vec];
        // arma::vec ordered_Y;
        // ordered_Y = as<arma::mat>(Y[order_vec]);
        arma::mat ordered_Y_vec = as<arma::mat>(clone(Y)).rows(as<uvec>(order_vec));
        // Rcout << ordered_Y_vec[order_vec];
        ordered_Y = ordered_Y_vec;
        // Rcout << ordered_Y;

        // TempD = data.frame(EuDis, Y, noise)[order(EuDis, noise), ]
        arma::vec U_1_vec;
        arma::vec U_2_vec;
        rowvec weight_vec;

        U_1_vec = reshape(ordered_Y,1,int(n)) * as<arma::mat>(weight_1);
        U_2_vec = reshape(ordered_Y,1,int(n)) * as<arma::mat>(weight_2);

        weight_vec = (reshape(ordered_Y, int(n), 1) % as<arma::mat>(weight_1)).t();

        arma::mat A_mat(arma::vec{1, 1, 1, pow((1 / bc_p),(2 /   std::min(p, 3.0)) ) });
        A_mat.reshape(2,2);
        arma::mat A_mat_inv = A_mat.i();


        arma::mat B_mat(arma::vec{1, 0});
        B_mat.reshape(2, 1);

        arma::mat Coefs = A_mat_inv * B_mat;

        arma::vec U_vec = Coefs(0,0) * U_1_vec + Coefs(1,0) * U_2_vec;

        // Rcout << "Shape of Y: " << size(ordered_Y) << "\n";
        // Rcout << "Shape of weights: " << size(as<arma::mat>(weight)) << "\n";
        // Rcout << "Shape of U_vec: " << size(U_vec) << "\n";
        // Rcout << "i: " << i << " estimate: " << sum(U_vec) << "\n";
        estimates.insert(i, sum(U_vec));
        // estimates.row(i) =  NumericVector(sum(U_vec));
        weight_mat.row(i) = NumericVector(weight_vec.begin(), weight_vec.end());



    }
    // List out = List::create( Named("estimates") = estimates, Named("weights") = transpose(weight_mat));
    List out = List::create( Named("estimates") = estimates,
                             Named("weights") = weight_mat);
    // Rcout << "Estimates: " << estimates;
    // return estimates;
    return out;

}

// [[Rcpp::export]]
NumericVector best_s(arma::mat estimate_matrix){

    NumericVector s_values(estimate_matrix.n_rows);
    // NumericVector s_values;

    // Loop through each of the rows of the estimate matrix
    for(R_xlen_t i = 0; i < estimate_matrix.n_rows; ++i) {
        arma::mat mat_row = estimate_matrix.row(i);
        arma::mat out_diff = diff(mat_row, 1, 1);

        IntegerVector idx = Range(0, (estimate_matrix.n_cols)-2);
        arma::mat out_denom = mat_row.cols(as<uvec>(idx));

        arma::mat diff_ratio = diff(abs( out_diff / out_denom), 1, 1);
        // Rcout << "diff_ratio: " << diff_ratio << "\n";
        double first_s = 0.0;
        // TO-DO: How do we handle the case where no s satisfies our condition?
        for(R_xlen_t i = 0; i < diff_ratio.size(); ++i) {
            if(diff_ratio(i) > -0.01){
                first_s = i + 1; // handle indexing difference between R and C++
                break;
            }
        }

        first_s += 3;
        s_values[i] = first_s;
        // return  first_s;
    }
    return s_values;
}


// [[Rcpp::export]]
NumericVector tuning(NumericVector s_seq,  NumericMatrix X, NumericVector Y,
              NumericMatrix X_test, double bc_p,
              Nullable<NumericVector> W0_ = R_NilValue){

    R_xlen_t n_vals = s_seq.length();
    double n_obs = X_test.nrow();
    arma::mat out(n_obs, n_vals, fill::zeros);

    // loop through and get the dnn estimates for each s value in sequence
    for(uword i = 0; i < n_vals; ++i) {
        List de_dnn_estimates = de_dnn(as<arma::mat>(X), Y, as<arma::mat>(X_test),
                                       s_seq[i] + 1, bc_p, W0_);
        // Change de_dnn to output a matrix even if only 1 x 1
        // arma::vec de_dnn_preds = as<colvec>(de_dnn_estimates["estimates"]);
        arma::vec de_dnn_preds(n_obs);
        de_dnn_preds = as<arma::vec>(de_dnn_estimates["estimates"]);
        // de_dnn_preds.attr("dim") = Dimension(1,n_obs);
        // Rcout << de_dnn_preds << std::endl;
        // out.column(i) = NumericVector(de_dnn_preds.begin(), de_dnn_preds.end());
        out.col(i) =  de_dnn_preds;
        // Rcout << "issue not with out matrix"<< "\n";
        // out[i] = List::create( Named("estimates") = de_dnn_estimates["estimates"],
        //                        Named("s") = s_seq[i]);

    }
    // Rcout <<
    // which(diff(abs(diff(tuning) / tuning[1:t - 1])) > -0.01)[1] + 3
    NumericVector s_choice = best_s(out);


    return s_choice;


}


// [[Rcpp::export]]
NumericVector tuning_greedy(NumericMatrix X, NumericVector Y,
                     NumericMatrix X_test, double bc_p,
                     Nullable<NumericVector> W0_ = R_NilValue){

    double n_obs = X_test.nrow();
    bool search_for_s = true;
    // make tuning matrix large and then resize it?
    arma::mat tuning_mat(n_obs, 50, fill::zeros);
    arma::vec best_s(n_obs, fill::zeros);
    double s = 0;
    // using zero indexing here to match with C++, note s + 1 -> s+2 in de_dnn call

    while (search_for_s){
        // Rcout << "s: " << s << std::endl;
        // For a given s, get the de_dnn estimates for each test observation
        List de_dnn_estimates = de_dnn(as<arma::mat>(X), Y, as<arma::mat>(X_test),
                                       s + 2, bc_p, W0_);

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

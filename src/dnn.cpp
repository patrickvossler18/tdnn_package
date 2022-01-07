#include "util.h"
#include "pdist.h"


arma::mat order_y_cols(const arma::mat& Y, const arma::mat& eu_dis, const arma::vec& noise){
    int ncol = eu_dis.n_cols;
    int nrow = Y.n_rows;
    std::vector<double> noise_vec = conv_to<std::vector<double>>::from(noise);
    arma::mat Y_sort(nrow, ncol);
    for(int j=0; j <ncol; j++){
        arma::colvec eu_dis_col = eu_dis.col(j);
        std::vector<double> eu_dis_tmp = conv_to<std::vector<double>>::from(eu_dis_col);

        vector<int> index(nrow, 0);
        for (int i = 0 ; i != index.size() ; i++) {
            index[i] = i;
        }
        sort(index.begin(), index.end(),
             [&](const int& a, const int& b) {
                 if (eu_dis_tmp[a] != eu_dis_tmp[b]){
                     return eu_dis_tmp[a] < eu_dis_tmp[b];
                 }
                 return noise_vec[a] < noise_vec[b];
             }
        );
        Y_sort.col(j) = Y.rows(conv_to<arma::uvec>::from(index));
    }
    return Y_sort;
}


arma::mat make_weight_mat(int n, arma::vec ord, arma::vec s_vec){
    arma::mat out(n, s_vec.size());

    for(int i=0; i <s_vec.size(); i ++){
        arma::vec weight_vec(n);
        Rcout << "make_weight_mat: s_vec[i]: " << s_vec[i] << std::endl;
        for (int j = 0 ; j < n ; j++) {
            weight_vec(j) = nChoosek(ord(j), s_vec[i] - 1.0);
        }
        weight_vec /=  nChoosek(n, s_vec[i]);
        weight_vec.reshape(n, 1);
        out.col(i) = weight_vec;
    }
    return out;
}



// [[Rcpp::export]]
arma::mat make_pdist_mat(const arma::mat& X, const arma::mat& X_test,
                         Nullable<NumericVector> W0_ = R_NilValue){
    // Handle case where W0 is not NULL:
    int d = X.n_cols;
    arma::mat X_subset;
    arma::mat X_test_subset;
    if (W0_.isNotNull()){
        NumericVector W0(W0_);
        // Now we need to filter X and X_test to only contain these columns
        X_subset = matrix_subset_logical(X, as<arma::vec>(W0));
        X_test_subset = matrix_subset_logical(X_test, as<arma::vec>(W0));
        d = sum(W0);
    } else{
        X_subset = X;
        X_test_subset = X_test;
    }
    arma::mat eu_dis = fastPdist(X_subset, X_test_subset);
    return(eu_dis);
}

// [[Rcpp::export]]
arma::mat make_ordered_Y_mat( const arma::mat &X, const arma::mat &Y,
                              const arma::mat &X_test, bool debug = false){

    arma::mat ordered_Y_mat(X.n_rows, X_test.n_rows);
    int p = X.n_cols;
    int n = X.n_rows;

    for(int i =0; i < X_test.n_rows; i++ ){
        arma::mat all_cols(int(p),1);
        all_cols.fill(1.0);

        // arma::mat all_rows;
        arma::mat X_dis;
        arma::mat EuDis;

        arma::mat X_test_row =  X_test.row(i);
        // all_rows = single_vec * X_test_row;
        arma::mat all_rows = arma::repmat(X_test_row,int(n),1);

        X_dis = X - all_rows;

        EuDis = (pow(X_dis, 2)) * all_cols;
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
        ordered_Y = ordered_Y_vec;
        ordered_Y_mat.col(i) = ordered_Y;
    }
    // ordered_Y_mat is a matrix with X.n_rows rows and X_test.n_rows columns
    return(ordered_Y_mat);
}


// [[Rcpp::export]]
arma::mat de_dnn_st_tuning( const arma::mat &X, const arma::mat &Y, const arma::mat &X_test,
                              const arma::vec& s_sizes, double c, double n_prop,
                              Nullable<NumericVector> W0_ = R_NilValue){
    // Handle case where W0 is not NULL:
    int d = X.n_cols;
    arma::mat X_subset;
    arma::mat X_test_subset;
    if (W0_.isNotNull()){
        NumericVector W0(W0_);
        // Now we need to filter X and X_test to only contain these columns
        X_subset = matrix_subset_logical(X, as<arma::vec>(W0));
        X_test_subset = matrix_subset_logical(X_test, as<arma::vec>(W0));
        d = sum(W0);
    } else{
        X_subset = X;
        X_test_subset = X_test;
    }

    arma::vec estimates(X_test_subset.n_rows);
    int n = X.n_rows;

    NumericVector ord = seq_cpp(1, n);
    ord.attr("dim") = Dimension(n, 1);
    // if we're using lfactorial, then we don't need ord = n - ord
    // ord = n - ord;
    arma::vec ord_arma = as<arma::vec>(ord);

    double w_1 = c/(c-1);
    double w_2 = -1/(c-1);


    arma::vec s_1 = s_sizes;
    arma::vec s_2 = round_modified(s_1 * pow(c, - double(d) / 2.0));
    // Rcout << "making weight mat s_1 ";
    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(int(n), ord_arma, s_1, n_prop, false);

    // Rcout << "making weight mat s_1 ";
    arma::mat weight_mat_s_2 = weight_mat_lfac_s_2_filter(int(n), ord_arma, s_2, n_prop, true);
    // Rcout << "done..." << std::endl;
    // arma::mat weight_mat_s_1 = weight_mat_lfac(int(n), ord_arma, s_1);
    // arma::mat weight_mat_s_2 = weight_mat_lfac(int(n), ord_arma, s_2);

    // this is an n by n_test matrix
    arma::mat ordered_Y_mat = make_ordered_Y_mat(X_subset,Y,X_test_subset);

    arma::mat U_mat(X_test.n_rows, s_sizes.size());

    // matrices are n_test by s_sizes.length
    // each column is a different s value
    arma::mat U_1_mat = ordered_Y_mat.t() * weight_mat_s_1;

    // need to find where we need to replace values with 1-NN estimate
    // easiest to just get col sums to find large s_2 vals
    arma::mat U_2_mat = ordered_Y_mat.t() * weight_mat_s_2;
    arma::vec U_2_vec = colSums_arma(U_2_mat);
    arma::uvec large_s_2 = find(U_2_vec == 0); // this should give column idx where s_2 was too large
    for(int i = 0; i < large_s_2.size(); i++){
        int col_location = large_s_2(i);
        // get the values for each column in one go
        arma::vec U_2_NN = get_1nn_reg(X_subset, X_test_subset, Y, 1);
        U_2_mat.col(col_location) = U_2_NN;
    }

    U_mat = w_1 * U_1_mat + w_2 * U_2_mat;

    return(U_mat);
}

// [[Rcpp::export]]
arma::vec de_dnn_st_mat_mult( const arma::mat &X, const arma::mat &Y, const arma::mat &X_test,
                          const arma::vec& s_sizes, double c, double n_prop,
                          Nullable<NumericVector> W0_ = R_NilValue, bool debug = false){

    // Handle case where W0 is not NULL:
    int d = X.n_cols;
    arma::mat X_subset;
    arma::mat X_test_subset;
    if (W0_.isNotNull()){
        NumericVector W0(W0_);
        // Now we need to filter X and X_test to only contain these columns
        X_subset = matrix_subset_logical(X, as<arma::vec>(W0));
        X_test_subset = matrix_subset_logical(X_test, as<arma::vec>(W0));
        d = sum(W0);
    } else{
        X_subset = X;
        X_test_subset = X_test;
    }


    arma::vec estimates(X_test_subset.n_rows);
    int n = X_subset.n_rows;

    NumericVector ord = seq_cpp(1, n);
    ord.attr("dim") = Dimension(n, 1);
    // if we're using lfactorial, then we don't need ord = n - ord
    // ord = n - ord;
    arma::vec ord_arma = as<arma::vec>(ord);

    double w_1 = c/(c-1);
    double w_2 = -1/(c-1);


    arma::vec s_1 = s_sizes;
    arma::vec s_2 = round_modified(s_1 * pow(c, - double(d) / 2.0));
    // Rcout << "making weight mat s_1 ";
    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(int(n), ord_arma, s_1, n_prop, false);

    // Rcout << "making weight mat s_1 ";
    arma::mat weight_mat_s_2 = weight_mat_lfac_s_2_filter(int(n), ord_arma, s_2, n_prop, true);
    // Rcout << "done..." << std::endl;
    // arma::mat weight_mat_s_1 = weight_mat_lfac(int(n), ord_arma, s_1);
    // arma::mat weight_mat_s_2 = weight_mat_lfac(int(n), ord_arma, s_2);

    // this is an n by n_test matrix
    arma::mat ordered_Y_mat = make_ordered_Y_mat(X_subset,Y,X_test_subset);

    arma::mat U_1_mat = ordered_Y_mat % weight_mat_s_1;
    arma::mat U_2_mat = ordered_Y_mat % weight_mat_s_2;

    arma::vec U_1_vec = colSums_arma(U_1_mat);
    arma::vec U_2_vec = colSums_arma(U_2_mat);

    //Since weight_mat_lfac_s_2_filter returns zero for
    // large s_2 values, we can find which test observations need to use 1-NN by finding
    // column sums equal to 0
    arma::uvec large_s_2 = find(U_2_vec == 0);
    arma::mat X_test_rows = matrix_row_subset_idx(X_test_subset, large_s_2);
    arma::vec U_2_NN = get_1nn_reg(X, X_test_rows, Y, 1);
    for(int j = 0; j < U_2_NN.size(); j++){
        double U_2_val = arma::as_scalar(U_2_NN(j));
        U_2_vec(j) = U_2_val;
    }

    arma::vec U_vec = w_1 * U_1_vec + w_2 * U_2_vec;

    return(U_vec);
}


// [[Rcpp::export]]
arma::vec de_dnn_st_loop( const arma::mat& X, const arma::mat &Y, const arma::mat &X_test,
                     const arma::vec& s_sizes, double c, double n_prop,
                     Nullable<NumericVector> W0_ = R_NilValue, bool debug = false){

    // Handle case where W0 is not NULL:
    int d = X.n_cols;
    arma::mat X_subset;
    arma::mat X_test_subset;
    if (W0_.isNotNull()){
        NumericVector W0(W0_);
        // Now we need to filter X and X_test to only contain these columns
        X_subset = matrix_subset_logical(X, as<arma::vec>(W0));
        X_test_subset = matrix_subset_logical(X_test, as<arma::vec>(W0));
        d = sum(W0);
    } else{
        X_subset = X;
        X_test_subset = X_test;
    }

    arma::vec estimates(X_test_subset.n_rows);
    int p = X_subset.n_cols;
    int n = X_subset.n_rows;

    NumericVector ord = seq_cpp(1, n);
    ord.attr("dim") = Dimension(n, 1);
    // if we're using lfactorial, then we don't need ord = n - ord
    // ord = n - ord;
    arma::vec ord_arma = as<arma::vec>(ord);


    arma::vec s_1 = s_sizes;
    arma::vec s_2 = round_modified(s_1 * pow(c, - double(d) / 2.0));
    // Rcout << "making weight mat s_1 ";
    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(int(n), ord_arma, s_1, n_prop, false);

    // Rcout << "making weight mat s_1 ";
    arma::mat weight_mat_s_2 = weight_mat_lfac_s_2_filter(int(n), ord_arma, s_2, n_prop, true);
    // Rcout << "done..." << std::endl;
    // arma::mat weight_mat_s_1 = weight_mat_lfac(int(n), ord_arma, s_1);
    // arma::mat weight_mat_s_2 = weight_mat_lfac(int(n), ord_arma, s_2);


    for(int i =0; i < X_test_subset.n_rows; i++ ){
        arma::mat all_cols(int(p),1);
        all_cols.fill(1.0);

        // arma::mat all_rows;
        arma::mat X_dis;
        arma::mat EuDis;

        arma::mat X_test_row =  X_test_subset.row(i);
        // all_rows = single_vec * X_test_row;
        arma::mat all_rows = arma::repmat(X_test_row,int(n),1);

        X_dis = X_subset - all_rows;

        EuDis = (pow(X_dis, 2)) * all_cols;

        // Rcout << "EuDis: "<< EuDis << std::endl;
        arma::vec noise(n);
        double noise_val = arma::randn<double>();
        noise.fill(noise_val);



        arma::vec vec_eu_dis = conv_to<arma::vec>::from(EuDis);
        arma::uvec index = r_like_order(vec_eu_dis, noise);

        arma::vec ordered_Y;
        arma::mat ordered_Y_vec = conv_to<arma::mat>::from(Y).rows(index);
        // // Rcout << ordered_Y_vec[order_vec];
        ordered_Y = ordered_Y_vec;
        // Rcout << "ordered_Y: " << ordered_Y << std::endl;


        arma::vec U_1_vec(ordered_Y.n_rows);
        arma::vec U_2_vec(ordered_Y.n_rows);
        rowvec weight_vec(ordered_Y.n_rows);

        double w_1 = c/(c-1);
        double w_2 = -1/(c-1);
        // Rcout << "w_1: " << w_1 << std::endl;
        // Rcout << "w_2: " << w_2 << std::endl;

        // Rcout << "weight_mat_s_1: " << weight_mat_s_1 << std::endl;
        U_1_vec = reshape(ordered_Y,1,int(n)) * weight_mat_s_1.col(i);
        // Rcout << "U_1: " << U_1_vec << std::endl;
        if(arma::accu(weight_mat_s_2.col(i)) == 0){
            // in this case s_2 is too large so we will get the 1-NN to use as the estimate
            // Rcout << "big s_2, using 1-NN: " << s_2(i) << std::endl;
            U_2_vec = get_1nn_reg(X, X_test_row, Y, 1);
            // Rcout << "U_2_vec: " << U_2_vec << std::endl;
        } else {
            U_2_vec = reshape(ordered_Y,1,int(n)) * weight_mat_s_2.col(i); // might need to convert this to mat?
            // Rcout << "U_2_vec: " << U_2_vec << std::endl;
        }

        arma::vec U_vec = w_1 * U_1_vec + w_2 * U_2_vec;
        // Rcout << "U_vec: " << U_vec << std::endl;
        // estimates.insert(i, sum(U_vec));
        estimates(i)=  sum(U_vec);
    }
    return(estimates);
}

// [[Rcpp::export]]
arma::vec de_dnn_st( const arma::mat& eu_dist_mat,
                     const arma::mat &Y,
                     const arma::mat & X_test,
                      const arma::vec& s_sizes, double c, int d, int n,
                      bool debug = false){


    // This just creates a sequence 1:n and then reverses it
    NumericVector ord = seq_cpp(1, double(n));
    ord.attr("dim") = Dimension(n, 1);
    ord = n - ord;
    arma::vec ord_arma = as<arma::vec>(ord);

    arma::vec noise(n);
    double noise_val = arma::randn<double>();
    noise.fill(noise_val);

    arma::mat Y_sort = order_y_cols(Y, eu_dist_mat, noise);

    double w_1 = c/(c-1);
    double w_2 = -1/(c-1);
    // changing s_sizes to arma::vec might break here
    if(debug){
        Rcout << "de_dnn_st: making s_1 and s_2" << std::endl;
    }

    arma::vec s_1 = s_sizes;
    arma::vec s_2 = round_modified(s_1 * pow(c, - double(d) / 2.0));
    if(debug){
        Rcout << "de_dnn_st: making weight matrices" << std::endl;
    }

    arma::mat weight_1_mat = weight_mat_lfac(n, ord_arma, s_1);
    arma::mat weight_2_mat = weight_mat_lfac(n, ord_arma, s_2);
    // arma::mat weight_1_mat = make_weight_mat(n, ord_arma, s_1);
    // arma::mat weight_2_mat = make_weight_mat(n, ord_arma, s_2);


    arma::vec U1(Y_sort.n_cols);
    arma::vec U2(Y_sort.n_cols);
    arma::vec U(Y_sort.n_cols);

    // this is likely a problem area
    if(debug){
        Rcout << "de_dnn_st: summing up columns" << std::endl;
    }

    U1 = colSums_arma(Y_sort % weight_1_mat);
    U2 = colSums_arma(Y_sort % weight_2_mat);

    U = w_1 * U1 + w_2 * U2;
    if(debug){
        Rcout<<"eu_dist_mat: " << eu_dist_mat<< std::endl;
        Rcout<<"s_1: " << s_1 <<std::endl;
        Rcout<<"s_2: " << s_2 <<std::endl;
        Rcout<<"weight_1_mat: " << weight_1_mat <<std::endl;
        Rcout<<"weight_2_mat: " << weight_2_mat <<std::endl;
        Rcout << "U1:"<< U1<< std::endl;
        Rcout << "U2:"<< U2<< std::endl;
        Rcout << "U:"<< U<< std::endl;
    }

    // List out = List::create( Named("estimates") = U);
    // return(out);
    return(U);

}

// [[Rcpp::export]]
NumericVector best_s(const arma::mat& estimate_matrix){

    NumericVector s_values(estimate_matrix.n_rows);
    // NumericVector s_values;

    // Loop through each of the rows of the estimate matrix
    for(R_xlen_t i = 0; i < estimate_matrix.n_rows; ++i) {
        arma::mat mat_row = estimate_matrix.row(i);
        arma::mat out_diff = diff(mat_row, 1, 1);

        IntegerVector idx = Range(0, (estimate_matrix.n_cols)-2);
        // Rcout << idx << std::endl;
        arma::mat out_denom = mat_row.cols(as<uvec>(idx));
        // Rcout << out_denom << std::endl;
        arma::mat diff_ratio = diff(abs( out_diff / out_denom), 1, 1);
        // Rcout << "diff_ratio: " << diff_ratio << "\n";
        double first_s = 0.0;
        // TO-DO: How do we handle the case where no s satisfies our condition?
        for(R_xlen_t i = 0; i < diff_ratio.size(); ++i) {
            // Rcout << "diff ratio i=" << i << ": " << diff_ratio(i) << std::endl;
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


// tuning_es(const arma::mat &eu_dis_mat, const arma::mat &Y,
//           double c, int d, int n,
//           Nullable<NumericVector> W0_ = R_NilValue)

// [[Rcpp::export]]
arma::vec tuning_st(const NumericVector& s_seq, const arma::mat &eu_dist_mat,
                    const arma::mat &X, const arma::mat &Y,
                    const arma::mat & X_test, double c,
                    int d, int n, int n_test_obs,
              NumericVector W0_, bool debug = false, bool verbose = false){
    // Nullable<NumericVector> W0 = R_NilValue;
    // if (W0_.isNotNull()){
    //     W0 = W0_;
    // }

    // R_xlen_t n_vals = s_seq.length();
    int n_vals = s_seq.length();
    int n_obs = n_test_obs;
    arma::mat out(n_obs, n_vals, fill::zeros);

    // loop through and get the dnn estimates for each s value in sequence
    for(int i = 0; i < n_vals; ++i) {
        Rcpp::checkUserInterrupt();
        if (verbose){
            Rcout << "tuning sequence i: " << s_seq[i] <<std::endl;
        }
        double s_fill = s_seq[i] + 1.0;
        arma::vec s_val(n_obs);
        s_val.fill(s_fill);
        // Rcout << s_val << std::endl;
        // s_val = s_seq[i] + 1.0;

        // Rcout << "X: " << X << std::endl;
        // Rcout << "X_test: " << X_test << std::endl;
        // Rcout << "Y: " << Y << std::endl;
        // Rcout << "s_val: " << s_val << std::endl;
        // Rcout << "bc_p: " << bc_p << std::endl;
        // Rcout << "W0: " << W0_ << std::endl;
        if (verbose){
            Rcout << "tuning: estimating de_dnn" << std::endl;
        }
        arma::vec de_dnn_estimates = de_dnn_st(eu_dist_mat, Y, X_test,
                                             s_val, c, d, n, debug);

        out.col(i) =  de_dnn_estimates;
        // Rcout << "issue not with out matrix"<< "\n";
        // out[i] = List::create( Named("estimates") = de_dnn_estimates["estimates"],
        //                        Named("s") = s_seq[i]);

    }
    // Rcout << out << std::endl;
    // which(diff(abs(diff(tuning) / tuning[1:t - 1])) > -0.01)[1] + 3
    NumericVector s_choice = best_s(out);


    return as<arma::vec>(s_choice);
    // return out;


}



// [[Rcpp::export]]
arma::vec tuning_es(const arma::mat &eu_dist_mat, const arma::mat &X, const arma::mat &Y,
                    const arma::mat & X_test,
                        double c, int d, int n, int n_test_obs,
                     Nullable<NumericVector> W0_ = R_NilValue, bool debug = false, bool verbose = false){
    int n_obs = n_test_obs;
    bool search_for_s = true;
    arma::mat tuning_mat(n_obs, 100, fill::zeros);
    // Dynamically size this matrix?
    // arma::mat tuning_mat;
    arma::vec best_s(n_obs, fill::zeros);
    double s = 0;
    // using zero indexing here to match with C++, note s + 1 -> s+2 in de_dnn call

    while (search_for_s){
        Rcpp::checkUserInterrupt();
        // Rcout << "s: " << s <<std::endl;
        if(verbose){
            Rcout << "s: " << s <<std::endl;
        }

        // s_val needs to be a vector of the same length as X_test
        double s_fill = s + 2;
        // NumericVector s_val(n_obs, s_fill);
        arma::vec s_val(n_obs);
        s_val.fill(s_fill);
        // s_val = s + 2;

        // For a given s, get the de_dnn estimates for each test observation
        // List de_dnn_estimates = de_dnn_st(X, Y, X_test,
        //                                s_val, c, W0_);
        // // This gives me an estimate for each test observation and is a n x 1 matrix
        // arma::vec de_dnn_est_vec = as<arma::vec>(de_dnn_estimates["estimates"]);
        arma::vec de_dnn_est_vec = de_dnn_st(eu_dist_mat, Y,X_test,
                                          s_val, c, d, n, debug);
        arma::mat candidate_results = de_dnn_est_vec;
        candidate_results.reshape(n_obs, 1);
        if (verbose){
            Rcout << "candidate_results: " << candidate_results <<std::endl;
        }

        // Now we add this column to our matrix if the matrix is empty
        if (s == 0 | s == 1){
            tuning_mat.col(s) = candidate_results;
        } else if (s >= tuning_mat.n_cols){
            if(verbose){
                Rcout << "Reaching condition that s >= tuning_mat.ncols" << std::endl;
                Rcout << "  s: " << s <<std::endl;
            }
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
                        Rcout << "diff_ratio(i, j): " << diff_ratio(i, j)  << std::endl;
                        if (diff_ratio(i, j) > -0.01){
                            best_s(i) = j + 1 + 3;
                            break; // if we've found the column that satisfies our condition, break and move to next row.
                        }
                    }
                }
            }
            if(verbose){
                Rcout << "Should be breaking out of the while loop since s > tuning_mat.n_cols" << std::endl;
                Rcout << "  s: " << s <<std::endl;
            }
            search_for_s = false; // since we've gone past the num of columns stop the while loop here
            break; // break out of our while loop to avoid going past number of columns in tuning_mat
        } else {

            // instead of resizing the matrix, just select columns 0-s
            arma::uvec s_vec = seq_int(0, int(s)-1);
            arma::mat resized_mat = matrix_subset_idx(tuning_mat, s_vec);
            // tuning_mat is an n x s matrix and we want to diff each of the rows
            arma::mat out_diff = diff(resized_mat, 1, 1);
            if (verbose){
                Rcout << "out_diff: " << out_diff <<std::endl;
            }
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
    // return NumericVector(best_s.begin(), best_s.end());
    return best_s;
}


// [[Rcpp::export]]
List est_reg_fn_rcpp(const arma::mat& X, const arma::mat& Y, const arma::mat& X_test,
                double c,
                Nullable<NumericVector> W0_ = R_NilValue,
                String tuning_method = "early stopping",
                bool verbose = false){
    // This function is called after we've done data checks in R

    // Handle case where W0 is not NULL:
    int d = X.n_cols;
    NumericVector W0;
    if (W0_.isNotNull()){
        W0 = W0_;
        d = sum(W0);
    } else{
        W0 = rep(1,d);
    }
    int n = X.n_rows;

    // Because the pdist matrix can be quite large, we will calculate it once at the
    // beginning and then pass it to tuning and de_dnn_st
    if(verbose){
        Rcout << "calculating pair-wise distance matrix. This might take a while..." << std::endl;
    }
    arma::mat eu_dist_mat = make_pdist_mat(X,X_test, W0);
    if(verbose){
        Rcout << "done" << std::endl;
    }
    arma::vec s_sizes(X_test.n_rows);
    if(verbose){
        Rcout << "Tuning s..." << std::endl;
    }
    if(tuning_method == "early stopping"){
        s_sizes = tuning_es(eu_dist_mat,X, Y, X_test,
                            c, d,n, X_test.n_rows,
                            W0, false, true);
    } else if(tuning_method == "sequence"){
        NumericVector s_seq = seq_cpp(1.0,50.0);
        if(verbose){
            Rcout << "s_seq..." << s_seq << std::endl;
        }
        s_sizes = tuning_st(s_seq, eu_dist_mat,X, Y, X_test,
                            c, d,n, X_test.n_rows,
                            W0,false, true);
    }
    if(verbose){
        Rcout << "estimating effect..." << std::endl;
    }

    arma::vec a_pred = de_dnn_st(eu_dist_mat, Y, X_test, s_sizes,
                                 c, d, n, false);

    arma::vec b_pred = de_dnn_st(eu_dist_mat, Y, X_test, s_sizes + 1,
                                 c, d, n, false);

    arma::vec deDNN_pred = (a_pred + b_pred) / 2;

    return(List::create( Named("estimates") = deDNN_pred,
                         Named("s") = s_sizes));

}

// [[Rcpp::export]]
arma::vec tuning_st_loop(const NumericVector& s_seq, const arma::mat& X,
                         const arma::mat& X_test,
                         const arma::mat &Y, double c,
                         double n_prop,
                         Nullable<NumericVector> W0_ = R_NilValue,
                         bool debug = false,
                         bool verbose = false){

    int d = X.n_cols;
    NumericVector W0;
    if (W0_.isNotNull()){
        W0 = W0_;
        d = sum(W0);
    } else{
        W0 = rep(1,d);
    }

    int n_vals = s_seq.length();
    int n_obs = X_test.n_rows;
    arma::mat out(n_obs, n_vals, fill::zeros);

    // loop through and get the dnn estimates for each s value in sequence
    for(int i = 0; i < n_vals; ++i) {
        Rcpp::checkUserInterrupt();
        if (verbose){
            Rcout << "tuning sequence i: " << s_seq[i] <<std::endl;
        }
        double s_fill = s_seq[i] + 1.0;
        arma::vec s_val(n_obs);
        s_val.fill(s_fill);

        if (verbose){
            Rcout << "tuning: estimating de_dnn" << std::endl;
        }
        arma::vec de_dnn_estimates = de_dnn_st_loop(X, Y, X_test,
                                                    s_val, c, n_prop, W0, debug);

        out.col(i) =  de_dnn_estimates;
        // Rcout << "issue not with out matrix"<< "\n";
        // out[i] = List::create( Named("estimates") = de_dnn_estimates["estimates"],
        //                        Named("s") = s_seq[i]);

    }
    NumericVector s_choice = best_s(out);


    return as<arma::vec>(s_choice);
    // return out;


}

// [[Rcpp::export]]
arma::vec tuning_st_mat_mult(const NumericVector& s_seq, const arma::mat& X,
                         const arma::mat& X_test,
                         const arma::mat &Y, double c,
                         double n_prop,
                         Nullable<NumericVector> W0_ = R_NilValue,
                         bool debug = false,
                         bool verbose = false){

    int d = X.n_cols;
    NumericVector W0;
    if (W0_.isNotNull()){
        W0 = W0_;
        d = sum(W0);
    } else{
        W0 = rep(1,d);
    }

    // int n_vals = s_seq.length();
    // int n_obs = X_test.n_rows;
    // arma::mat out(n_obs, n_vals, fill::zeros);

    arma::mat out = de_dnn_st_tuning(X,Y,X_test, s_seq + 1, c, n_prop,W0);

    NumericVector s_choice = best_s(out);


    return as<arma::vec>(s_choice);
    // return out;


}

// [[Rcpp::export]]
arma::vec tuning_es_loop(const arma::mat &X, const arma::mat &Y,
                    const arma::mat & X_test,
                    double c, int d, double n_prop,
                    Nullable<NumericVector> W0_ = R_NilValue, bool debug = false, bool verbose = false){

    NumericVector W0;
    if (W0_.isNotNull()){
        W0 = W0_;
        d = sum(W0);
    } else{
        W0 = rep(1,d);
    }

    int n_obs = X_test.n_rows;
    bool search_for_s = true;
    arma::mat tuning_mat(n_obs, 100, fill::zeros);
    // Dynamically size this matrix?
    // arma::mat tuning_mat;
    arma::vec best_s(n_obs, fill::zeros);
    double s = 0;
    // using zero indexing here to match with C++, note s + 1 -> s+2 in de_dnn call

    while (search_for_s){
        Rcpp::checkUserInterrupt();
        // Rcout << "s: " << s <<std::endl;
        if(verbose){
            Rcout << "s: " << s <<std::endl;
        }

        // s_val needs to be a vector of the same length as X_test
        double s_fill = s + 2;
        // NumericVector s_val(n_obs, s_fill);
        arma::vec s_val(n_obs);
        s_val.fill(s_fill);
        // s_val = s + 2;

        // For a given s, get the de_dnn estimates for each test observation
        // List de_dnn_estimates = de_dnn_st(X, Y, X_test,
        //                                s_val, c, W0_);
        // // This gives me an estimate for each test observation and is a n x 1 matrix
        // arma::vec de_dnn_est_vec = as<arma::vec>(de_dnn_estimates["estimates"]);

        arma::vec de_dnn_est_vec = de_dnn_st_loop( X, Y,X_test,
                                             s_val, c, n_prop, W0, debug);
        arma::mat candidate_results = de_dnn_est_vec;
        candidate_results.reshape(n_obs, 1);
        if (verbose){
            Rcout << "candidate_results: " << candidate_results <<std::endl;
        }

        // Now we add this column to our matrix if the matrix is empty
        if (s == 0 | s == 1){
            tuning_mat.col(s) = candidate_results;
        } else if (s >= tuning_mat.n_cols){
            if(verbose){
                Rcout << "Reaching condition that s >= tuning_mat.ncols" << std::endl;
                Rcout << "  s: " << s <<std::endl;
            }
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
                        // Rcout << "diff_ratio(i, j): " << diff_ratio(i, j)  << std::endl;
                        if (diff_ratio(i, j) > -0.01){
                            best_s(i) = j + 1 + 3;
                            break; // if we've found the column that satisfies our condition, break and move to next row.
                        }
                    }
                }
            }
            if(verbose){
                Rcout << "Should be breaking out of the while loop since s > tuning_mat.n_cols" << std::endl;
                Rcout << "  s: " << s <<std::endl;
            }
            search_for_s = false; // since we've gone past the num of columns stop the while loop here
            break; // break out of our while loop to avoid going past number of columns in tuning_mat
        } else {

            // instead of resizing the matrix, just select columns 0-s
            arma::uvec s_vec = seq_int(0, int(s)-1);
            arma::mat resized_mat = matrix_subset_idx(tuning_mat, s_vec);
            // tuning_mat is an n x s matrix and we want to diff each of the rows
            arma::mat out_diff = diff(resized_mat, 1, 1);
            if (verbose){
                Rcout << "out_diff: " << out_diff <<std::endl;
            }
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
    // return NumericVector(best_s.begin(), best_s.end());
    return best_s;
}



// [[Rcpp::export]]
arma::vec est_reg_fn_st_loop(const arma::mat& X,
                        const arma::mat& Y,
                        const arma::mat& X_test,
                     double c,
                     double n_prop,
                     String tuning_method = "early stopping",
                     Nullable<NumericVector> W0_ = R_NilValue,
                     bool verbose = false){
    // This function is called after we've done data checks in R
    // Handle case where W0 is not NULL:
    int d = X.n_cols;
    NumericVector W0;
    if (W0_.isNotNull()){
        W0 = W0_;
        d = sum(W0);
    } else{
        W0 = rep(1,d);
    }
    arma::vec s_sizes;
    if(tuning_method == "early stopping"){
        s_sizes = tuning_es_loop(X, Y, X_test,
                            c, d, n_prop, W0, false, verbose);
    } else if(tuning_method == "sequence"){
        NumericVector s_seq = seq_cpp(1.0,50.0);
        if(verbose){
            Rcout << "s_seq..." << s_seq << std::endl;
        }
        s_sizes = tuning_st_loop(s_seq, X,X_test, Y,
                                           c, n_prop, W0,false, verbose);
    }



    if(verbose){
        Rcout << "estimating effect..." << std::endl;
    }

    arma::vec a_pred = de_dnn_st_loop(X, Y, X_test, s_sizes, c, n_prop, W0);

    arma::vec b_pred = de_dnn_st_loop(X, Y, X_test, s_sizes + 1, c, n_prop, W0);

    arma::vec deDNN_pred = (a_pred + b_pred) / 2;

    return(deDNN_pred);

}

// double est_effect_st(const arma::mat& X, const arma::mat& Y,
//                      const arma::mat& X_test, const arma::vec& W,
//                      const arma::vec& s_choice_0, const arma::vec& s_choice_1,
//                      const NumericVector& W0,
//                      const double c){
//     // get bootstrap data
//     arma::uvec bstrap_idx = floor(randu<uvec>(X.n_rows));
//     arma::mat X_boot = matrix_subset_idx(X, bstrap_idx);
//     arma::mat Y_boot = matrix_subset_idx(Y, bstrap_idx);
//     arma::vec W_boot = vector_subset_idx(W, bstrap_idx);
//
//     // filter by treatment group
//     arma::uvec trt_idx = find(W_boot == 1);
//     arma::uvec ctl_idx = find(W_boot == 0);
//
//     arma::mat X_trt = X_boot.rows(trt_idx);
//     arma::mat Y_trt = Y_boot.rows(trt_idx);
//
//     arma::mat X_ctl = X_boot.rows(ctl_idx);
//     arma::mat Y_ctl = Y_boot.rows(ctl_idx);
//
//     // calc reg fn for each treatment group
//     arma::vec trt_est_a = est_reg_fn_st(X_trt, Y_trt, X_test, c, s_choice_1, W0);
//     arma::vec trt_est_b = est_reg_fn_st(X_trt, Y_trt, X_test, c, s_choice_1 + 1.0, W0);
//
//     arma::vec ctl_est_a = est_reg_fn_st(X_ctl, Y_ctl, X_test, c, s_choice_0, W0);
//     arma::vec ctl_est_b = est_reg_fn_st(X_ctl, Y_ctl, X_test, c, s_choice_0 + 1.0, W0);
//
//     // calculate estimates for treatment and control groups
//     arma::vec trt_est = (trt_est_a + trt_est_b) / 2.0;
//     arma::vec ctl_est = (ctl_est_a + ctl_est_b) / 2.0;
//
//     // effect should be a single number
//     arma::vec diff = trt_est - ctl_est;
//     return(arma::as_scalar(diff));
// }

// de_dnn_st_loop( const arma::mat& X, const arma::mat& Y, const arma::mat& X_test,
//                 const arma::vec& s_sizes, double c, double d,
//                 Nullable<NumericVector> W0_ = R_NilValue, bool debug = false)





// struct BootstrapEstimate: public Worker {
//     // input matrices to read from
//     const arma::mat X;
//     const arma::mat Y;
//     const arma::mat X_test;
//     const arma::mat W;
//     const arma::vec s_choice_0;
//     const arma::vec s_choice_1;
//     const NumericVector W0;
//     const double c;
//
//     // output matrix to write to
//     arma::vec boot_stats;
//
//
//     // for each iteration, we need to pass everything to de_dnn_st
//     BootstrapEstimate(arma::vec boot_stats,
//                       const arma::mat& X,
//                       const arma::mat& Y,
//                       const arma::mat& X_test,
//                       const arma::mat& W,
//                       const arma::vec& s_choice_0,
//                       const arma::vec& s_choice_1,
//                       const NumericVector& W0,
//                       const double c):
//         X(X), Y(Y), X_test(X_test), W(W), s_choice_0(s_choice_0),
//         s_choice_1(s_choice_1), W0(W0), c(c){}
//
//     void operator()(std::size_t begin, std::size_t end) {
//         for (std::size_t i = begin; i < end; i++) {
//             // std::cout << i << endl;
//             double effect = est_effect_st(X, Y,
//                           X_test, W,
//                           s_choice_0, s_choice_1,
//                           W0,
//                           c);
//             boot_stats(i) = effect;
//         }
//     }
// };

// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::export]]
// arma::vec bootstrap_cpp_mt(const arma::mat& X,
//                           const arma::mat& Y,
//                           const arma::mat& X_test,
//                           const arma::vec& W,
//                           const arma::vec& s_choice_0,
//                           const arma::vec& s_choice_1,
//                           const NumericVector& W0,
//                           const double c,
//                           const int B){
//     // initialize results vector
//     arma::vec boot_stats;
//
//     // Initialize the struct
//     BootstrapEstimate bstrap_est(boot_stats,
//                                  X,
//                                  Y,
//                                  X_test,
//                                  W,
//                                  s_choice_0,
//                                  s_choice_1,
//                                  W0,
//                                  c);
//
//     parallelFor(0, B, bstrap_est);
//
//     return(boot_stats);
// };
//
//

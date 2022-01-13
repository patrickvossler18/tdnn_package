#include "util.h"
#include "kd_tree.h"

void SampleReplace(arma::uvec &index, int nOrig, int size) {
    int ii;
    for (ii = 0; ii < size; ii++) {
        arma::vec rand_val = arma::randu<vec>(1);
        index(ii) = nOrig * int(as_scalar(rand_val));
    }
}


// [[Rcpp::export]]
arma::uvec sample_replace_index(const int &size){
    arma::uvec out(size);
    int ii;
    for (ii = 0; ii < size; ii++) {
        // arma::vec rand_val = arma::randu<vec>(1);
        // out(ii) = size * as_scalar(rand_val);
        out(ii) = size * arma::randu<double>();
    }
    return out;
}

// [[Rcpp::export]]
arma::vec de_dnn_st_boot( const arma::mat& X, const arma::mat &Y, const arma::mat &X_test,
                          const arma::vec& s_sizes, const arma::vec& ord,
                          double c, double n_prop){
    // Assuming X and X_test have already been properly subsetted
    int d = X.n_cols;

    arma::vec estimates(X_test.n_rows);
    int p = X.n_cols;
    int n = X.n_rows;

    arma::vec s_1 = s_sizes;
    // arma::vec s_2 = round_modified(s_1 * pow(c, - double(d) / 2.0));
    arma::vec s_2 = arma_round(s_1 * pow(c, - double(d) / 2.0));
    arma::vec tmp = s_1 * pow(c, - double(d) / 2.0);
    // Rcout << "tmp: " << tmp << std::endl;
    // Rcout << "s_2: " << s_2 << std::endl;
    // Rcout << "making weight mat s_1 ";
    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(n, ord, s_1, n_prop, false);

    // Rcout << "making weight mat s_1 ";
    arma::mat weight_mat_s_2 = weight_mat_lfac_s_2_filter(n, ord, s_2, n_prop, true);


    for(int i =0; i < X_test.n_rows; i++ ){
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

        arma::vec noise(n);
        double noise_val = arma::randn<double>();
        noise.fill(noise_val);


        arma::vec vec_eu_dis = conv_to<arma::vec>::from(EuDis);
        arma::uvec index = r_like_order(vec_eu_dis, noise);


        arma::vec ordered_Y;
        arma::mat ordered_Y_mat = conv_to<arma::mat>::from(Y).rows(index);

        ordered_Y = ordered_Y_mat;
        arma::vec U_1_vec(ordered_Y.n_rows);
        arma::vec U_2_vec(ordered_Y.n_rows);
        rowvec weight_vec(ordered_Y.n_rows);

        double w_1 = c/(c-1);
        double w_2 = -1/(c-1);

        U_1_vec = arma::reshape(ordered_Y,1,int(n)) * weight_mat_s_1.col(i);

        if(arma::accu(weight_mat_s_2.col(i)) == 0){
            // in this case s_2 is too large so we will get the 1-NN to use as the estimate
            // Rcout << "big s_2, using 1-NN: " << s_2(i) << std::endl;
            U_2_vec = get_1nn_reg(X, X_test_row, Y, 1);
        } else {
            U_2_vec = reshape(ordered_Y,1,int(n)) * weight_mat_s_2.col(i); // might need to convert this to mat?
        }

        arma::vec U_vec = w_1 * U_1_vec + w_2 * U_2_vec;
        // Rcout << "U_vec: " << U_vec << std::endl;
        // estimates.insert(i, sum(U_vec));
        estimates(i)=  sum(U_vec);
    }
    return(estimates);
}

// [[Rcpp::export]]
arma::vec tdnn_st_boot(arma::mat X, arma::vec Y, arma::mat X_test,
                           const arma::mat& weight_mat_s_1,
                           const arma::mat& weight_mat_s_2,
                           const arma::mat& weight_mat_s_1_plus_1,
                           const arma::mat& weight_mat_s_2_plus_1,
                           double c,
                           double n_prop){

    arma::vec estimates(X_test.n_rows);
    int n = X.n_rows;
    int p = X.n_cols;

    for(int i =0; i < X_test.n_rows; i++ ){
        arma::mat all_cols(p,1);
        all_cols.fill(1.0);

        // arma::mat all_rows;
        arma::mat X_dis;
        arma::mat EuDis;

        arma::mat X_test_row =  X_test.row(i);
        // all_rows = single_vec * X_test_row;
        arma::mat all_rows = arma::repmat(X_test_row,n,1);

        X_dis = X - all_rows;

        EuDis = (arma::pow(X_dis, 2)) * all_cols;

        arma::vec noise(n);
        double noise_val = arma::randn<double>();
        noise.fill(noise_val);

        arma::vec vec_eu_dis = conv_to<arma::vec>::from(EuDis);

        arma::uvec index = r_like_order(vec_eu_dis, noise);

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

        // the weight matrix is # train obs x # test obs so we want to use the
        // ith column of the weight mat for the ith test observation
        U_1_vec = reshape(ordered_Y,1,n) * weight_mat_s_1.col(i);
        U_1_1_vec = reshape(ordered_Y,1,n) * weight_mat_s_1_plus_1.col(i);
        if(arma::accu(weight_mat_s_2.col(i)) == 0){
            arma::vec nn_1_result = get_1nn_reg(X, X_test_row, Y, 1);
            U_2_vec = arma::as_scalar(nn_1_result);
            U_2_1_vec = arma::as_scalar(nn_1_result);


        } else {
            U_2_vec = reshape(ordered_Y,1,n) * weight_mat_s_2.col(i);
            U_2_1_vec = reshape(ordered_Y,1,n) * weight_mat_s_2_plus_1.col(i);
        }


        arma::vec U_vec = w_1 * U_1_vec + w_2 * U_2_vec;
        arma::vec U_vec_1 = w_1 * U_1_1_vec + w_2 * U_2_1_vec;
        // now take the average of the two estimates and use that as our final estimate
        arma::vec avg_est = (U_vec + U_vec_1)/ 2.0;
        // Rcout << avg_est << std::endl;
        estimates(i)=  arma::as_scalar(avg_est);
    }
    // Rcout << estimates << std::endl;
    return estimates;
}



struct BootstrapEstimate: public Worker {
    // input matrices to read from
    const arma::mat X;
    const arma::mat Y;
    const arma::mat X_test;
    const arma::vec s_choice;
    const arma::mat& weight_mat_s_1;
    const arma::mat& weight_mat_s_2;
    const arma::mat& weight_mat_s_1_plus_1;
    const arma::mat& weight_mat_s_2_plus_1;
    const double c;
    const double n_prop;

    // output matrix to write to
    // arma::mat boot_stats;
    RMatrix<double> boot_stats;

    // for each iteration, we need to pass everything to tdnn
    BootstrapEstimate(NumericMatrix boot_stats,
                      const arma::mat& X,
                      const arma::mat& Y,
                      const arma::mat& X_test,
                      const arma::mat& weight_mat_s_1,
                      const arma::mat& weight_mat_s_2,
                      const arma::mat& weight_mat_s_1_plus_1,
                      const arma::mat& weight_mat_s_2_plus_1,
                      const double c,
                      const double n_prop):
        X(X), Y(Y), X_test(X_test),
        weight_mat_s_1(weight_mat_s_1),
        weight_mat_s_2(weight_mat_s_2),
        weight_mat_s_1_plus_1(weight_mat_s_1_plus_1),
        weight_mat_s_2_plus_1(weight_mat_s_2_plus_1),
        c(c), n_prop(n_prop), boot_stats(boot_stats){}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {

            // sample observations with replacement
            arma::uvec boot_idx = sample_replace_index(X.n_rows);
            arma::mat X_boot = matrix_row_subset_idx(X, boot_idx);
            arma::mat Y_boot = matrix_row_subset_idx(Y, boot_idx);

            arma::vec est = tdnn_st_boot(X_boot, Y_boot, X_test,
                                         weight_mat_s_1,
                                         weight_mat_s_2,
                                         weight_mat_s_1_plus_1,
                                         weight_mat_s_2_plus_1,
                                         c, n_prop);
            // Rcout << est << std::endl;
            // boot_stats.column(i) = est;
            // boot_stats.col(i) = est;

            // NumericVector est_rcpp = NumericVector(est.begin(),est.end());
            // boot_stats(_, i) = est_rcpp;
            // RMatrix<double>::Column column = boot_stats.column(i);
            // size_t n = column.length();
            for(int j =0; j < X_test.n_rows; j++){
                boot_stats(j,i) = est(j);
            }
            // Rcout << boot_stats << std::endl;
        }
    }
};

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
NumericMatrix bootstrap_cpp_mt(const arma::mat& X,
                          const arma::mat& Y,
                          const arma::mat& X_test,
                          const arma::vec& s_choice,
                          const double c,
                          const double n_prop,
                          const int B,
                          Nullable<NumericVector> W0_ = R_NilValue){


    // Filter by W0
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



    // Infer n and p from our data after we've filtered for relevant features
    int n = X_subset.n_rows;
    int p = X_subset.n_cols;

    arma::vec ord = arma::linspace(1,n, n);
    arma::vec s_1 = s_choice;
    arma::vec s_2 = round_modified(s_1 * pow(c, - double(d) / 2.0));

    arma::vec s_1_1 = s_1 + 1;
    arma::vec s_2_1 = round_modified(s_1_1 * pow(c, - double(d) / 2.0));

    // Generate these matrices once since they won't change and just pass them to the workers
    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(n, ord, s_1, n_prop, false);
    arma::mat weight_mat_s_2 = weight_mat_lfac_s_2_filter(n, ord, s_2, n_prop, true);

    arma::mat weight_mat_s_1_plus_1 = weight_mat_lfac_s_2_filter(n, ord, s_1_1, n_prop, false);
    arma::mat weight_mat_s_2_plus_1 = weight_mat_lfac_s_2_filter(n, ord, s_2_1, n_prop, true);



    // initialize results matrix
    // arma::mat boot_stats(X_test_subset.n_rows, B);
    NumericMatrix boot_stats(X_test_subset.n_rows, B);

    // Initialize the struct
    BootstrapEstimate bstrap_est(boot_stats,
                                 X_subset,
                                 Y,
                                 X_test_subset,
                                 weight_mat_s_1,
                                 weight_mat_s_2,
                                 weight_mat_s_1_plus_1,
                                 weight_mat_s_2_plus_1,
                                 c,
                                 n_prop);

    parallelFor(0, B, bstrap_est);
    // Rcout<< boot_stats << std::endl;
    return(boot_stats);
};



#include "util.h"

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


struct BootstrapEstimate: public Worker {
    // input matrices to read from
    const arma::mat X;
    const arma::mat Y;
    const arma::mat X_test;
    const arma::vec s_choice;
    const arma::vec ord;
    const double c;
    const double n_prop;

    // output matrix to write to
    arma::mat boot_stats;
    // RMatrix<double> boot_stats;


    // for each iteration, we need to pass everything to de_dnn_st
    BootstrapEstimate(arma::mat boot_stats,
                      const arma::mat& X,
                      const arma::mat& Y,
                      const arma::mat& X_test,
                      const arma::vec& s_choice,
                      const arma::vec& ord,
                      const double c,
                      const double n_prop):
        X(X), Y(Y), X_test(X_test), s_choice(s_choice),
        ord(ord), c(c), n_prop(n_prop), boot_stats(boot_stats){}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            arma::vec a_pred = de_dnn_st_boot(X, Y,X_test,s_choice,ord,c,n_prop);
            arma::vec b_pred = de_dnn_st_boot(X, Y,X_test,s_choice + 1,ord,c,n_prop);
            arma::vec est = (a_pred + b_pred) / 2;
            boot_stats.col(i) = est;
            // NumericVector est_rcpp = NumericVector(est.begin(),est.end());
            // RMatrix<double>::Column column = boot_stats.column(i);
            // size_t n = column.length();
            // for(int j =0; j < n; j++){
            //     column[i] = est(i);
            // }
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
    int n = X.n_rows;
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

    arma::vec ord = arma::linspace(1,n, n);

    // initialize results matrix
    // NumericMatrix boot_stats(X_test_subset.n_rows, B);
    arma::mat boot_stats(X_test_subset.n_rows, B);
    // Rcout << "boot_stats.n_rows: " <<boot_stats.n_rows << std::endl;
    // Rcout << "boot_stats.n_cols: " <<boot_stats.n_cols << std::endl;
    // Initialize the struct
    BootstrapEstimate bstrap_est(boot_stats,
                                 X_subset,
                                 Y,
                                 X_test_subset,
                                 s_choice,
                                 ord,
                                 c,
                                 n_prop);

    parallelFor(0, B, bstrap_est);

    return(wrap(boot_stats));
};



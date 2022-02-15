#include "util.h"
#include "kd_tree.h"

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
                          const double C_s_2,
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
    int log_n = log(n);
    int s_2_val = round_modified(exp( C_s_2 * log_n * (double(d)/(double(d) + 8))));

    arma::vec ord = arma::linspace(1,n, n);
    arma::vec s_1 = s_choice;
    // arma::vec s_2 = round_modified(s_1 * pow(c, - double(d) / 2.0));
    // arma::vec s_2(s_1.n_elem, fill::value(round_modified(C_s_2 * pow(n, double(d) / (double(d) + 8)))));
    arma::vec s_2(s_1.n_elem, fill::value(s_2_val));

    arma::vec s_1_1 = s_1 + 1;
    // arma::vec s_2_1(s_1.n_elem, fill::value(round_modified(C_s_2 * pow(n, double(d) / (double(d) + 8)))));
    arma::vec s_2_1(s_1.n_elem, fill::value(s_2_val));

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


struct TrtEffectBootstrapEstimate: public Worker {
    // input matrices to read from

    const arma::mat X_trt;
    const arma::mat Y_trt;
    const arma::mat weight_mat_s_1_trt;
    const arma::mat weight_mat_s_2_trt;
    const arma::mat weight_mat_s_1_plus_1_trt;
    const arma::mat weight_mat_s_2_plus_1_trt;
    const double c_trt;
    const arma::mat X_ctl;
    const arma::mat Y_ctl;
    const arma::mat X_test_ctl;
    const arma::mat weight_mat_s_1_ctl;
    const arma::mat weight_mat_s_2_ctl;
    const arma::mat weight_mat_s_1_plus_1_ctl;
    const arma::mat weight_mat_s_2_plus_1_ctl;
    const double c_ctl;
    const arma::mat X_test;
    const double n_prop;

    // output matrix to write to
    // arma::mat boot_stats;
    RMatrix<double> boot_stats;

    // for each iteration, we need to pass everything to tdnn
    TrtEffectBootstrapEstimate(NumericMatrix boot_stats,
                      const arma::mat& X_trt,
                      const arma::mat& Y_trt,
                      const arma::mat& weight_mat_s_1_trt,
                      const arma::mat& weight_mat_s_2_trt,
                      const arma::mat& weight_mat_s_1_plus_1_trt,
                      const arma::mat& weight_mat_s_2_plus_1_trt,
                      const double c_trt,
                      const arma::mat& X_ctl,
                      const arma::mat& Y_ctl,
                      const arma::mat& weight_mat_s_1_ctl,
                      const arma::mat& weight_mat_s_2_ctl,
                      const arma::mat& weight_mat_s_1_plus_1_ctl,
                      const arma::mat& weight_mat_s_2_plus_1_ctl,
                      const double c_ctl,
                      const arma::mat& X_test,
                      const double n_prop):
        X_trt(X_trt), Y_trt(Y_trt),
        weight_mat_s_1_trt(weight_mat_s_1_trt),
        weight_mat_s_2_trt(weight_mat_s_2_trt),
        weight_mat_s_1_plus_1_trt(weight_mat_s_1_plus_1_trt),
        weight_mat_s_2_plus_1_trt(weight_mat_s_2_plus_1_trt),
        c_trt(c_trt),
        X_ctl(X_ctl), Y_ctl(Y_ctl),
        weight_mat_s_1_ctl(weight_mat_s_1_ctl),
        weight_mat_s_2_ctl(weight_mat_s_2_ctl),
        weight_mat_s_1_plus_1_ctl(weight_mat_s_1_plus_1_ctl),
        weight_mat_s_2_plus_1_ctl(weight_mat_s_2_plus_1_ctl),
        c_ctl(c_ctl),
        X_test(X_test),
        n_prop(n_prop), boot_stats(boot_stats){}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {

            // sample observations with replacement
            arma::uvec trt_boot_idx = sample_replace_index(X_trt.n_rows);
            arma::mat X_boot_trt = matrix_row_subset_idx(X_trt, trt_boot_idx);
            arma::mat Y_boot_trt = matrix_row_subset_idx(Y_trt, trt_boot_idx);

            arma::uvec ctl_boot_idx = sample_replace_index(X_ctl.n_rows);
            arma::mat X_boot_ctl = matrix_row_subset_idx(X_ctl, ctl_boot_idx);
            arma::mat Y_boot_ctl = matrix_row_subset_idx(Y_ctl, ctl_boot_idx);

            arma::vec trt_mu = tdnn_st_boot(X_boot_trt, Y_boot_trt, X_test,
                                         weight_mat_s_1_trt,
                                         weight_mat_s_2_trt,
                                         weight_mat_s_1_plus_1_trt,
                                         weight_mat_s_2_plus_1_trt,
                                         c_trt, n_prop);

            arma::vec ctl_mu = tdnn_st_boot(X_boot_ctl, Y_boot_ctl, X_test,
                                            weight_mat_s_1_ctl,
                                            weight_mat_s_2_ctl,
                                            weight_mat_s_1_plus_1_ctl,
                                            weight_mat_s_2_plus_1_ctl,
                                            c_ctl, n_prop);
            arma::vec trt_effect = trt_mu - ctl_mu;

            for(int j =0; j < X_test.n_rows; j++){
                boot_stats(j,i) = trt_effect(j);
            }
            // Rcout << boot_stats << std::endl;
        }
    }
};


std::tuple<arma::mat, arma::mat,arma::mat, arma::mat> make_weight_matrix(
        int n, int d, double n_prop, double C_s_2, arma::vec s_choice ){
    arma::vec ord = arma::linspace(1,n, n);

    int log_n = log(n);
    int s_2_val = round_modified(exp( C_s_2 * log_n * (double(d)/(double(d) + 8))));
    arma::vec s_1 = s_choice;
    arma::vec s_2(s_1.n_elem, fill::value(s_2_val));
    // arma::vec s_1 = s_choice;
    // arma::vec s_2 = round_modified(s_1 * pow(c, - double(d) / 2.0));

    arma::vec s_1_1 = s_1 + 1;
    arma::vec s_2_1(s_1.n_elem, fill::value(s_2_val));
    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(n, ord, s_1, n_prop, false);
    arma::mat weight_mat_s_2 = weight_mat_lfac_s_2_filter(n, ord, s_2, n_prop, true);

    arma::mat weight_mat_s_1_plus_1 = weight_mat_lfac_s_2_filter(n, ord, s_1_1, n_prop, false);
    arma::mat weight_mat_s_2_plus_1 = weight_mat_lfac_s_2_filter(n, ord, s_2_1, n_prop, true);
    // // Generate these matrices once since they won't change and just pass them to the workers
    return std::make_tuple(weight_mat_s_1, weight_mat_s_2,
                           weight_mat_s_1_plus_1, weight_mat_s_2_plus_1
    );
}


// [[Rcpp::export]]
NumericMatrix bootstrap_trt_effect_cpp_mt(const arma::mat& X,
                                          const arma::mat& Y,
                                          const arma::vec& W,
                                          const arma::mat& X_test,
                                          const arma::vec& s_choice_trt,
                                          const arma::vec& s_choice_ctl,
                                          const double c,
                                          const double n_prop,
                                          const double C_s_2,
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
    // int p = X_subset.n_cols;

    arma::uvec trt_idx = find(W == 1);
    arma::uvec ctl_idx = find(W == 0);

    arma::mat X_trt = X_subset.rows(trt_idx);
    arma::mat Y_trt = Y.rows(trt_idx);

    arma::mat X_ctl = X_subset.rows(ctl_idx);
    arma::mat Y_ctl = Y.rows(ctl_idx);


    arma::mat weight_mat_s_1_trt;
    arma::mat weight_mat_s_2_trt;

    arma::mat weight_mat_s_1_plus_1_trt;
    arma::mat weight_mat_s_2_plus_1_trt;

    arma::mat weight_mat_s_1_ctl;
    arma::mat weight_mat_s_2_ctl;

    arma::mat weight_mat_s_1_plus_1_ctl;
    arma::mat weight_mat_s_2_plus_1_ctl;

    std::tie(weight_mat_s_1_trt,weight_mat_s_2_trt, weight_mat_s_1_plus_1_trt,
             weight_mat_s_2_plus_1_trt) = make_weight_matrix(X_trt.n_rows, int(d),
             n_prop, C_s_2, s_choice_trt);

    std::tie(weight_mat_s_1_ctl,weight_mat_s_2_ctl, weight_mat_s_1_plus_1_ctl,
             weight_mat_s_2_plus_1_ctl) = make_weight_matrix(X_ctl.n_rows, int(d),
             n_prop, C_s_2, s_choice_ctl);

    // initialize results matrix
    // arma::mat boot_stats(X_test_subset.n_rows, B);
    NumericMatrix boot_stats(X_test_subset.n_rows, B);

    // Initialize the struct
    TrtEffectBootstrapEstimate bstrap_est(boot_stats,
                                 X_trt,
                                 Y_trt,
                                 weight_mat_s_1_trt,
                                 weight_mat_s_2_trt,
                                 weight_mat_s_1_plus_1_trt,
                                 weight_mat_s_2_plus_1_trt,
                                 c,
                                 X_ctl,
                                 Y_ctl,
                                 weight_mat_s_1_ctl,
                                 weight_mat_s_2_ctl,
                                 weight_mat_s_1_plus_1_ctl,
                                 weight_mat_s_2_plus_1_ctl,
                                 c,
                                 X_test_subset,
                                 n_prop);

    parallelFor(0, B, bstrap_est);
    // Rcout<< boot_stats << std::endl;
    return(boot_stats);
};



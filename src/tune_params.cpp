#include "tune_params.h"
#include <RcppArmadilloExtensions/sample.h>

double vec_dist(arma::rowvec x, arma::vec y)
{
    double d = sum(pow(x - y.t(), 2));
    return d;
}

// [[Rcpp::export]]
arma::vec pt_mat_dist(arma::mat X, arma::vec y)
{
    int n = X.n_rows;
    arma::vec distances(n);
    for (int i = 0; i < n; i++)
    {
        arma::rowvec x_obs = X.row(i);
        double dist_val = vec_dist(x_obs, y);
        distances(i) = dist_val;
    }
    return (distances);
}

// [[Rcpp::export]]
arma::vec make_ord_vec(int n)
{
    // This just creates a sequence 1:n and then reverses it
    NumericVector ord = seq_cpp(1, n);
    ord.attr("dim") = Dimension(n, 1);
    // don't need to reverse if we are using lfactorial
    // ord = n - ord;
    arma::vec ord_arma = as<arma::vec>(ord);
    // calculate euclidean distances for a fixed test point once here
    return (ord_arma);
}

// [[Rcpp::export]]
arma::rowvec make_ordered_Y_vec(const arma::mat &X_train,
                                const arma::vec &X_val_vec,
                                const arma::mat &Y_train,
                                int n)
{
    arma::vec noise(n);
    double noise_val = arma::randn<double>();
    noise.fill(noise_val);

    arma::vec vec_eu_dis = pt_mat_dist(X_train, X_val_vec);
    arma::uvec index = r_like_order(vec_eu_dis, noise);

    // order Y by the euclidean distance
    arma::rowvec ordered_Y = conv_to<arma::rowvec>::from(Y_train.rows(index));
    return (ordered_Y);
}

// [[Rcpp::export]]
arma::vec make_param_estimate(const arma::mat &X_train,
                              const arma::mat &Y_train,
                              const arma::mat &X_val,
                              const arma::rowvec &ordered_Y,
                              const arma::vec &ord_arma,
                              int n,
                              int p,
                              int log_n,
                              double c,
                              int s_1_val,
                              int s_2_val,
                              double n_prop)
{

    // int s_2_val = std::ceil(int(round_modified(exp(M * log_n * (double(p) / (double(p) + 8))))));
    // int s_1_val = std::ceil(int(round_modified(s_2_val * pow(c, double(p) / 2))));

    // need to choose s_1 through tuning
    arma::vec s_1(X_val.n_rows, fill::value(s_1_val));
    arma::vec s_1_1(s_1.n_elem, fill::value(s_1_val + 1));

    arma::vec s_2(s_1.n_elem, fill::value(s_2_val));
    arma::vec s_2_1(s_1.n_elem, arma::fill::value(ceil((s_1_val + 1) * 2)));

    // generate the weight matrices (technically vector because only one test observation)
    arma::mat weight_mat_s_1 = weight_mat_lfac_s_2_filter(n, ord_arma, s_1, n_prop, false);
    arma::mat weight_mat_s_2 = weight_mat_lfac_s_2_filter(n, ord_arma, s_2, n_prop, true);

    arma::mat weight_mat_s_1_plus_1 = weight_mat_lfac_s_2_filter(n, ord_arma, s_1_1, n_prop, false);
    arma::mat weight_mat_s_2_plus_1 = weight_mat_lfac_s_2_filter(n, ord_arma, s_2_1, n_prop, true);
    // Rcout << "weight mat num cols: "<< weight_mat_s_1.n_cols << std::endl;

    // generate weight vector here
    arma::vec U_1_vec(ordered_Y.n_elem);
    arma::vec U_2_vec;
    arma::vec U_1_1_vec(ordered_Y.n_elem);
    arma::vec U_2_1_vec;

    // double w_1 = c / (c - 1);
    // double w_2 = -1 / (c - 1);
    double w_2 = pow(c, 2 / double(p)) / (pow(c, 2 / double(p)) - 1);
    double w_1 = -1 / (pow(c, 2 / double(p)) - 1);

    // the weight matrix is # train obs x # test obs so we want to use the ith column of the weight mat for the ith test observation
    // NOTE: hard-coding the column to be the first column since we are doing LOO sampling
    U_1_vec = ordered_Y * weight_mat_s_1.col(0);
    U_1_1_vec = ordered_Y * weight_mat_s_1_plus_1.col(0);
    U_2_vec = ordered_Y * weight_mat_s_2.col(0);
    U_2_1_vec = ordered_Y * weight_mat_s_2_plus_1.col(0);

    arma::vec U_vec = w_1 * U_1_vec + w_2 * U_2_vec;
    arma::vec U_vec_1 = w_1 * U_1_1_vec + w_2 * U_2_1_vec;

    // Rcout << "U_1_vec: " << U_1_vec << std::endl;
    // Rcout << "U_2_vec: " << U_2_vec << std::endl;
    // Rcout << "U_1_1_vec: " << U_1_1_vec << std::endl;
    // Rcout << "U_2_1_vec: " << U_2_1_vec << std::endl;
    // Rcout << "w_1: " << w_1 << std::endl;
    // Rcout << "w_2: " << w_2 << std::endl;
    // now take the average of the two estimates and use that as our final estimate
    // this is our tdnn estimate for this given c and M value
    // arma::vec avg_est = (U_vec + U_vec_1) / 2.0;
    arma::vec avg_est = U_vec;
    return (avg_est);
}

struct ParamEstimates : public RcppParallel::Worker
{
    const arma::mat X_train;
    const arma::mat Y_train;
    const arma::mat X_val;
    const arma::mat Y_val;
    const arma::mat ordered_Y;
    const arma::mat ord_arma;
    const arma::mat param_mat;

    int n;
    int p;
    int log_n;
    double n_prop;


    // output vector to write to
    RcppParallel::RVector<double> estimates;

    ParamEstimates(const arma::mat &X_train,
                   const arma::mat &Y_train,
                   const arma::mat &X_val,
                   const arma::mat &ordered_Y,
                   const arma::mat &ord_arma,
                   const arma::mat &param_mat,
                   int n,
                   int p,
                   int log_n,
                   double n_prop,
                   NumericVector estimates) : X_train(X_train), Y_train(Y_train),
                                              X_val(X_val),
                                              ordered_Y(ordered_Y),
                                              ord_arma(ord_arma),
                                              param_mat(param_mat),
                                              n(n), p(p), log_n(log_n),
                                              n_prop(n_prop),
                                              estimates(estimates) {}

    void operator()(std::size_t begin, std::size_t end)
    {
        for (std::size_t i = begin; i < end; i++)
        {
            // loop through the parameter combinations
            double c = param_mat(i, 0);
            // double M = param_mat(i, 1);
            int s_1_val = int(param_mat(i, 1));
            int s_2_val = int(std::ceil(s_1_val * c));

            arma::vec avg_est = make_param_estimate(X_train, Y_train, X_val,
                                                    ordered_Y, ord_arma,
                                                    n, p, log_n, c,
                                                    s_1_val, s_2_val,
                                                    n_prop);
            estimates[i] = arma::as_scalar(avg_est);
        }
    }
};

// [[Rcpp::export]]
Rcpp::List loo_test(const arma::mat &X,
                    const arma::mat &Y,
                    const arma::mat &X_test,
                    const arma::mat &param_mat,
                    int B,
                    double n_prop,
                    NumericVector W0,
                    bool verbose)
{
    int n_params = param_mat.n_rows;
    // Store the results where each column is one Monte Carlo replication and each row is a parameter combination
    // Doing it this way since Armadillo is column major format
    arma::mat results_mat(n_params, B);
    arma::rowvec truth_vec(B);

    arma::uvec full_idx = seq_int(0, X.n_rows - 1);
    arma::uvec val_idx = Rcpp::RcppArmadillo::sample(full_idx, 1, false);
    arma::uvec train_idx = full_idx.elem(find(full_idx != as_scalar(val_idx)));

    arma::mat X_train = matrix_row_subset_idx(X, train_idx);
    arma::mat Y_train = matrix_row_subset_idx(Y, train_idx);
    // Rcout << "Able to subset train data" << std::endl;
    arma::mat X_val = matrix_row_subset_idx(X, val_idx);
    arma::vec X_val_vec = arma::conv_to<arma::vec>::from(X_val);
    arma::mat Y_val = matrix_row_subset_idx(Y, val_idx);

    int n = X_train.n_rows;
    int p = X_train.n_cols;
    int log_n = log(n);
    // Rcout << "Able to generate data for iter: "<< b << std::endl;

    arma::vec ord_arma = make_ord_vec(n);

    arma::rowvec ordered_Y = make_ordered_Y_vec(X_train, X_val_vec,
                                                Y_train, n);

    arma::vec param_estimates(n_params);
    for (int i = 0; i < n_params; i++)
    {
        // loop through the parameter combinations
        double c = param_mat(i, 0);
        // double M = param_mat(i, 1);
        int s_1_val = int(param_mat(i, 1));
        int s_2_val = int(std::ceil(s_1_val * c));

        arma::vec avg_est = make_param_estimate(X_train, Y_train, X_val,
                                                ordered_Y, ord_arma,
                                                n, p, log_n, c,
                                                s_1_val, s_2_val,
                                                n_prop);
        param_estimates(i) = arma::as_scalar(avg_est);
    }
    return (
        Rcpp::List::create(
            Named("param_estimates") = param_estimates,
            Named("ord_arma") = ord_arma,
            Named("ordered_Y") = ordered_Y,
            Named("val_idx") = val_idx,
            Named("train_idx") = train_idx,
            Named("X_train") = X_train,
            Named("Y_train") = Y_train,
            Named("X_val") = X_val,
            Named("Y_val") = Y_val));
}

// [[Rcpp::export]]
arma::vec tune_params(const arma::mat &X,
                      const arma::mat &Y,
                      const arma::mat &X_test,
                      const arma::mat &param_mat,
                      int B,
                      double n_prop,
                      NumericVector W0,
                      bool verbose)
{
    // arma::mat X_subset = matrix_subset_logical(X, as<arma::vec>(W0));
    // arma::mat X_test_subset = matrix_subset_logical(X_test, as<arma::vec>(W0));
    // int d = sum(W0);
    int n_params = param_mat.n_rows;
    // Store the results where each column is one Monte Carlo replication and each row is a parameter combination
    // Doing it this way since Armadillo is column major format
    arma::mat results_mat(n_params, B);
    arma::rowvec truth_vec(B);
    for (int b = 0; b < B; b++)
    {
        // generate LOO samples
        arma::uvec full_idx = seq_int(0, X.n_rows - 1);
        arma::uvec val_idx = Rcpp::RcppArmadillo::sample(full_idx, 1, false);
        arma::uvec train_idx = full_idx.elem(find(full_idx != as_scalar(val_idx)));

        arma::mat X_train = matrix_row_subset_idx(X, train_idx);
        arma::mat Y_train = matrix_row_subset_idx(Y, train_idx);
        // Rcout << "Able to subset train data" << std::endl;
        arma::mat X_val = matrix_row_subset_idx(X, val_idx);
        arma::vec X_val_vec = arma::conv_to<arma::vec>::from(X_val);
        arma::mat Y_val = matrix_row_subset_idx(Y, val_idx);

        int n = X_train.n_rows;
        int p = X_train.n_cols;
        int log_n = log(n);
        // Rcout << "Able to generate data for iter: "<< b << std::endl;

        arma::vec ord_arma = make_ord_vec(n);

        arma::rowvec ordered_Y = make_ordered_Y_vec(X_train, X_val_vec,
                                                    Y_train, n);

        // MT CODE
        NumericVector param_est_vec(n_params);
        ParamEstimates param_estimates(X_train, Y_train, X_val,
                                       ordered_Y, ord_arma, param_mat,
                                       n, p, log_n, n_prop, param_est_vec);
        RcppParallel::parallelFor(0, n_params, param_estimates);
        arma::vec param_est_vec_arma = as<arma::vec>(param_est_vec);
        results_mat.col(b) = pow(param_est_vec_arma - arma::as_scalar(Y_val), 2);

        // ST CODE
        // arma::vec param_estimates(n_params);
        // for (int i = 0; i < n_params; i++)
        // {
        //     // loop through the parameter combinations
        //     double c = param_mat(i, 0);
        //     // double M = param_mat(i, 1);
        //     int s_1_val = int(param_mat(i, 1));
        //     int s_2_val = int(std::ceil(s_1_val * c));
        //
        //     arma::vec avg_est = make_param_estimate(X_train, Y_train, X_val,
        //                                             ordered_Y, ord_arma,
        //                                             n, p, log_n, c,
        //                                             s_1_val, s_2_val,
        //                                             n_prop);
        //     param_estimates(i) = arma::as_scalar(avg_est);
        // }
        // results_mat.col(b) = pow(param_estimates - arma::as_scalar(Y_val), 2);
    }
    // now to find the best parameter combination, we will loop over the rows
    arma::vec mse_results = rowMeans_arma(results_mat);
    // arma::vec mse_results(n_params);
    // for (int i = 0; i < n_params; i++)
    // {
    //     arma::rowvec param_mc_est = results_mat.row(i);
    //     double mse = mean(pow((truth_vec - param_mc_est), 2));
    //     mse_results(i) = mse;
    // }
    int min_mse_idx = mse_results.index_min();
    // return(List::create(Named("c") = as_scalar(param_mat(min_mse_idx,0)),
    //                     Named("M") = as_scalar(param_mat(min_mse_idx,1))));
    arma::vec results_vec = {as_scalar(param_mat(min_mse_idx, 0)), as_scalar(param_mat(min_mse_idx, 1))};
    return (results_vec);
}

// [[Rcpp::export]]
Rcpp::List tune_params_debug(const arma::mat &X,
                             const arma::mat &Y,
                             const arma::mat &X_test,
                             const arma::mat &param_mat,
                             int B,
                             double n_prop,
                             NumericVector W0,
                             bool verbose)
{
    // arma::mat X_subset = matrix_subset_logical(X, as<arma::vec>(W0));
    // arma::mat X_test_subset = matrix_subset_logical(X_test, as<arma::vec>(W0));
    // int d = sum(W0);
    int n_params = param_mat.n_rows;
    // Store the results where each column is one Monte Carlo replication and each row is a parameter combination
    // Doing it this way since Armadillo is column major format
    arma::mat results_mat(n_params, B);
    arma::rowvec truth_vec(B);

    // last element will be the validation index
    arma::mat loo_idx_mat(X.n_rows, B);
    arma::mat x_val_mat(3, B);
    arma::mat param_est_mat(n_params, B);
    // arma::mat x_train_mat(X.n_rows-1,B);
    for (int b = 0; b < B; b++)
    {
        // generate LOO samples
        arma::uvec full_idx = seq_int(0, X.n_rows - 1);
        arma::uvec val_idx = Rcpp::RcppArmadillo::sample(full_idx, 1, false);
        arma::uvec train_idx = full_idx.elem(find(full_idx != as_scalar(val_idx)));
        arma::uvec all_idx = join_cols(train_idx, val_idx);
        loo_idx_mat.col(b) = conv_to<arma::vec>::from(all_idx);

        arma::mat X_train = matrix_row_subset_idx(X, train_idx);
        arma::mat Y_train = matrix_row_subset_idx(Y, train_idx);
        // Rcout << "Able to subset train data" << std::endl;
        arma::mat X_val = matrix_row_subset_idx(X, val_idx);
        arma::vec X_val_vec = arma::conv_to<arma::vec>::from(X_val);
        arma::mat Y_val = matrix_row_subset_idx(Y, val_idx);
        x_val_mat.col(b) = X_val_vec;

        int n = X_train.n_rows;
        int p = X_train.n_cols;
        int log_n = log(n);
        // Rcout << "Able to generate data for iter: "<< b << std::endl;

        arma::vec ord_arma = make_ord_vec(n);

        arma::rowvec ordered_Y = make_ordered_Y_vec(X_train, X_val_vec,
                                                    Y_train, n);

        // MT CODE
        // NumericVector param_est_vec(n_params);
        // ParamEstimates param_estimates(X_train, Y_train, X_val,
        //                                ordered_Y, ord_arma, param_mat,
        //                                n, p, log_n, n_prop, param_est_vec);
        // RcppParallel::parallelFor(0, n_params, param_estimates);
        // arma::vec param_est_vec_arma = as<arma::vec>(param_est_vec);
        // results_mat.col(b) = pow(param_est_vec_arma - arma::as_scalar(Y_val), 2);

        // ST CODE
        arma::vec param_estimates(n_params);
        for (int i = 0; i < n_params; i++)
        {
            // loop through the parameter combinations
            double c = param_mat(i, 0);
            // double M = param_mat(i, 1);
            int s_1_val = int(param_mat(i, 1));
            int s_2_val = int(std::ceil(s_1_val * c));

            arma::vec avg_est = make_param_estimate(X_train, Y_train, X_val,
                                                    ordered_Y, ord_arma,
                                                    n, p, log_n, c,
                                                    s_1_val, s_2_val,
                                                    n_prop);
            param_estimates(i) = arma::as_scalar(avg_est);
        }
        results_mat.col(b) = pow(param_estimates - arma::as_scalar(Y_val), 2);
        param_est_mat.col(b) = param_estimates;
        truth_vec(b) = arma::as_scalar(Y_val);
    }
    // now to find the best parameter combination, we will loop over the rows
    arma::vec mse_results = rowMeans_arma(results_mat);
    // arma::vec mse_results(n_params);
    // for (int i = 0; i < n_params; i++)
    // {
    //     arma::rowvec param_mc_est = results_mat.row(i);
    //     double mse = mean(pow((truth_vec - param_mc_est), 2));
    //     mse_results(i) = mse;
    // }
    int min_mse_idx = mse_results.index_min();
    arma::vec results_vec = {as_scalar(param_mat(min_mse_idx, 0)), as_scalar(param_mat(min_mse_idx, 1))};
    return (Rcpp::List::create(
        Named("param_vec") = results_vec,
        Named("mse_results") = mse_results,
        Named("results_mat") = results_mat,
        Named("param_est_mat") = param_est_mat,
        Named("truth_vec") = truth_vec,
        Named("idx_mat") = loo_idx_mat,
        Named("x_val_mat") = x_val_mat));
}

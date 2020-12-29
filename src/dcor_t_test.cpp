#include "util.h"


Rcpp::NumericMatrix calcPWD (const Rcpp::NumericMatrix & x){
    unsigned int outrows = x.nrow(), i = 0, j = 0;
    double d;
    Rcpp::NumericMatrix out(outrows,outrows);

    for (i = 0; i < outrows - 1; i++){
        Rcpp::NumericVector v1 = x.row(i);
        for (j = i + 1; j < outrows ; j ++){
            d = sqrt(sum(pow(v1-x.row(j), 2.0)));
            out(j,i)=d;
            out(i,j)=d;
        }
    }

    return out;
}


arma::mat calcPWD_arma (const arma::mat & x){
    unsigned int outrows = x.n_rows, i = 0, j = 0;
    double d;
    // Rcpp::NumericMatrix out(outrows,outrows);
    arma::mat out(outrows,outrows);

    for (i = 0; i < outrows - 1; i++){
        // Rcpp::NumericVector v1 = x.row(i);
        arma::vec v1 = x.row(i);
        for (j = i + 1; j < outrows ; j ++){
            d = std::sqrt(arma::sum(arma::pow(v1-x.row(j), 2.0)));
            out(j,i)=d;
            out(i,j)=d;
        }
    }

    return out;
}


arma::mat sweep_arma(arma::mat x, arma::mat y, int margin){
    int n = x.n_rows;
    int m = x.n_cols;

    arma::mat res(n, m);
    int i, j;
    if (margin == 1){
        // sweep over rows
        for (j = 0; j < m; j++){
            for (i = 0; i < n; i++){
                res(i, j) = x(i, j) - y[i];
            }
        }
    }
    if (margin == 2){
        // sweep over columns
        for (i = 0; i < n; i++){
            for (j = 0; j < m; j++){
                res(i, j) = x(i, j) - y[j];
            }
        }
    }
    return res;
}


NumericMatrix sweep(NumericMatrix x, NumericVector y, int margin){
    int n = x.nrow();
    int m = x.ncol();

    NumericMatrix res(n, m);
    int i, j;
    if (margin == 1){
        // sweep over rows
        for (j = 0; j < m; j++){
            for (i = 0; i < n; i++){
                res(i, j) = x(i, j) - y[i];
            }
        }
    }
    if (margin == 2){
        // sweep over columns
        for (i = 0; i < n; i++){
            for (j = 0; j < m; j++){
                res(i, j) = x(i, j) - y[j];
            }
        }
    }
    return res;
}



NumericMatrix Astar(NumericMatrix d){
    int n = d.nrow();
    // int p = d.ncol();
    NumericVector m = Rcpp::rowMeans(d);
    // assuming this gives us mean of a matrix??
    // Rcout << "get row means ok" << std::endl;
    double M = Rcpp::mean(d);
    NumericMatrix a = sweep(d, m, 1);
    NumericMatrix b = sweep(a, m, 2);
    NumericVector A = (b + M) - (d / double(n));
    // Rcout << A << std::endl;
    A.attr("dim") = Dimension(n,n);
    NumericMatrix A_mat = as<NumericMatrix>(A);
    NumericVector diag_fill = m - M;

    // Assuming square matrix
    for( int i = 0; i < n; i++){
        A_mat(i,i) = diag_fill[i];
    }
    return (double(n) / double(n - 1)) * A_mat;

}


List BCDCOR(NumericMatrix x, NumericMatrix y){
    NumericMatrix x_dist = calcPWD(x);
    NumericMatrix y_dist = calcPWD(y);
    int n = x_dist.nrow();
    double n_bias = double(n) / (double(n) - 2.0);
    NumericMatrix AA = Astar(x_dist);
    NumericMatrix BB = Astar(y_dist);
    arma::mat AA_BB = as<arma::mat>(AA) % as<arma::mat>(BB);
    arma::mat AA_AA = as<arma::mat>(AA) % as<arma::mat>(AA);
    arma::mat BB_BB = as<arma::mat>(BB) % as<arma::mat>(BB);
    double XY = accu(AA_BB) - n_bias * accu(AA_BB.diag());
    double XX = accu(AA_AA) - n_bias * accu(AA_AA.diag());
    double YY = accu(BB_BB) - n_bias * accu(BB_BB.diag());

    List res_list = List::create( Named("bcR") = XY/ sqrt(XX * YY),
                                  _["XY"] = XY/ pow(double(n), 2),
                                   _["XX"] = XX/ pow(double(n), 2),
                                    _["YY"]= YY/ pow(double(n), 2),
                                     _["n"]= n);

    return res_list;

}


double dcor_t_test(NumericMatrix x, NumericMatrix y){
    List stats = BCDCOR(x, y);
    double bcR = stats["bcR"];
    int n = stats["n"];
    double M = double(n) * (double(n) - 3)/2;
    double df = M - 1;
    double tstat = sqrt(M - 1) * bcR/sqrt(1 - pow(bcR,2));
    double pval = R::pt(tstat, df,true,false);
    return 1- pval;
}


Rcpp::NumericVector feature_screening(NumericMatrix x, NumericMatrix y, double alpha=0.001){

    int n = x.nrow();
    int num_col = x.ncol();
    double p = x.ncol();
    NumericVector res(num_col);
    for (int i = 0; i < num_col; i++){
        NumericVector x_col = (x(_, i));
        x_col.attr("dim") = Dimension(n,1);
        double dcor_test = dcor_t_test(as<NumericMatrix>(x_col), y);
        if (dcor_test < alpha/p){
            res(i) = 1;
        } else {
            res(i) = 0;
        }
    }
    return res;
}

arma::vec rowMeans_arma( arma::mat& d){
    int nRows = d.n_rows;
    arma::vec out(nRows);

    for( int i=0; i < nRows; i++ ) {
        arma::rowvec tmp = d.row(i);
        out(i) = arma::mean(tmp);
    }

    return out;
}



arma::mat Astar_parallel(arma::mat d){
    int n = d.n_rows;
    arma::vec m = rowMeans_arma(d);

    double M = arma::mean(d.as_row());
    arma::mat a = sweep_arma(d, m, 1);
    arma::mat b = sweep_arma(a, m, 2);
    arma::mat A = (b + M) - (d / double(n));
    // Rcout << A.n_cols << std::endl;
    // Rcout << A.n_rows << std::endl;
    arma::vec diag_fill = m - M;

    // Assuming square matrix
    for( int i = 0; i < n; i++){
        A(i,i) = diag_fill[i];
    }
    return (double(n) / double(n - 1)) * A;

}



std::tuple<double,int> BCDCOR_parallel(arma::mat x, arma::mat y){
    arma::mat x_dist = calcPWD_arma(x);
    arma::mat y_dist = calcPWD_arma(y);
    int n = x_dist.n_rows;
    double n_bias = double(n) / (double(n) - 2.0);
    arma::mat AA = Astar_parallel(x_dist);
    arma::mat BB = Astar_parallel(y_dist);
    arma::mat AA_BB = AA % BB;
    arma::mat AA_AA = AA % AA;
    arma::mat BB_BB = BB % BB;
    double XY = accu(AA_BB) - n_bias * accu(AA_BB.diag());
    double XX = accu(AA_AA) - n_bias * accu(AA_AA.diag());
    double YY = accu(BB_BB) - n_bias * accu(BB_BB.diag());

    return std::make_tuple(XY/ sqrt(XX * YY), n);
}

double beta_inc(double X, double A, double B){
    double A0 = 0;
    double B0 = 1;
    double A1 = 1;
    double B1 = 1;
    double M9 = 0;
    double A2 = 0;
    double C9;

    while (std::abs((A1 - A2)/ A1) > 0.00001) {
        A2 = A1;
        C9 = -(A + M9) * ( A + B + M9) * X / (A + 2 * M9)/(A + 2 * M9 + 1);
        A0 = A1+ C9 * A0;
        B0 = B1 + C9 * B0;
        M9 = M9+ 1;
        C9 = M9*(B-M9) * X /( A + 2  * M9 - 1)/(A + 2 * M9);
        A1 = A0 + C9 * A1;
        B1 = B0 + C9 * B1;
        A0 = A0/B1;
        B0 = B0/B1;
        A1 = A1/B1;
        B1 = 1;
    }
    return A1 / A;
}

double pt_raw(double tstat, double df){
    double A = df/ 2;
    double S = A + 0.5;
    double Z = df / (df + std::pow(tstat,2));

    double BT = std::exp(lgamma(S) - lgamma(0.5) - lgamma(A) + A *
                         std::log(Z) + 0.5 * std::log(1-Z) );
    double betacdf;
    if (Z<(A+1)/(S+2)) {
        betacdf = BT * beta_inc(Z, A, 0.5);
    } else {
        betacdf =1-BT* beta_inc(1-Z, 0.5, A);
    }
    double tcdf;
    if (tstat < 0) {
        tcdf = betacdf/2;
    } else {
        tcdf=1-betacdf/2;
    }
    return tcdf;
}


// [[Rcpp::export]]
double dcor_t_test_parallel(arma::mat x, arma::mat y){
    double bcR;
    int n;
    std::tie(bcR, n) = BCDCOR_parallel(x, y);
    double M = double(n) * (double(n) - 3)/2;
    double df = M - 1;
    double tstat = sqrt(M - 1) * bcR/sqrt(1 - pow(bcR,2));
    // double pval = R::pt(tstat, df,true,false);
    double pval = pt_raw(tstat, df);
    return 1- pval;
}


struct FeatureScreening : public Worker {
    const arma::mat x;
    const arma::mat y;
    const double alpha;

    const double n;
    const double p;
    const double num_col;

    RVector<double> res;

    FeatureScreening( const arma::mat x,
                      const arma::mat y,
                      const double alpha,
                      const int n,
                      const double p,
                      const int num_col,
                      NumericVector res
    ) : x(x), y(y), alpha(alpha), n(n), p(p), num_col(num_col), res(res) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; i++) {
            arma::mat x_col = x.col(i);
            double dcor_test = dcor_t_test_parallel(x_col, y);
            if (dcor_test < alpha/p){
                res[i] = 1;
            } else {
                res[i] = 0;
            }
        }
    }
};


// [[Rcpp::export]]
Rcpp::NumericVector feature_screening_parallel(arma::mat x,
                                               arma::mat y,
                                               double alpha=0.001){

    int n = x.n_rows;
    int num_col = x.n_cols;
    double p = x.n_cols;
    NumericVector res(num_col);

    FeatureScreening featureScreening(x, y,
                                    alpha, n,
                                    p, num_col, res);

    parallelFor(0, num_col, featureScreening);

    return res;
}

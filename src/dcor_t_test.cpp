#include "util.h"


// [[Rcpp::export]]
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

// [[Rcpp::export]]
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


// [[Rcpp::export]]
NumericMatrix Astar(NumericMatrix d){
    int n = d.nrow();
    // int p = d.ncol();
    NumericVector m = Rcpp::rowMeans(d);
    // assuming this gives us mean of a matrix??
    // Rcout << "get row means ok" << std::endl;
    double M = Rcpp::mean(d);
    NumericMatrix a = sweep(d, m, 1);
    NumericMatrix b = sweep(a, m, 2);
    // Rcout << "sweep ok ok" << std::endl;
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

// [[Rcpp::export]]
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

// [[Rcpp::export]]
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

// [[Rcpp::export]]
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

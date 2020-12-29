#include "util.h"

double nChoosek(double n, double k)
{
    if (k == 0) return 1.0;

    /*
        Extra computation saving for large R,
    using property:
        N choose R = N choose (N-R)
    */
        // if (k > n / 2.0) return nChoosek(n, n - k);

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

arma::mat matrix_subset_logical( const arma::mat & x,
                                const arma::vec & y, int mrgn) {
    // Assumes that y is 0/1 coded.
    // find() retrieves the integer index when y is equivalent 1.
    arma::mat ret_mat;
    uvec subset_vec = find(y == 1) ;
    if( mrgn == 1){
        ret_mat =  x.cols(find(y == 1) );
    } else{
        ret_mat = x.rows(find(y==1));
    }
    return ret_mat;

}



Rcpp::NumericMatrix matrix_subset_idx_rcpp(
        Rcpp::NumericMatrix x, Rcpp::IntegerVector y) {

    // Determine the number of observations
    int n_rows_out = y.size();

    // Create an output matrix
    Rcpp::NumericMatrix out = Rcpp::no_init(n_rows_out,x.ncol() );

    // Loop through each row and copy the data.
    for(unsigned int z = 0; z < n_rows_out; ++z) {
        out(z, Rcpp::_) = x(y[z], Rcpp::_);
    }

    return out;
}


arma::uvec seq_int(long int a, long int b){
    long int d = std::abs(b-a)+1;

    return conv_to<arma::uvec>::from(arma::linspace(a, b, d));
}


arma::mat matrix_subset_idx(const arma::mat& x,
                            const arma::uvec& y) {

    // y must be an integer between 0 and columns - 1
    // Allows for repeated draws from same columns.
    return x.cols( y );
}


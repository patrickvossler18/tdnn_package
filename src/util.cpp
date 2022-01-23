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

arma::mat matrix_row_subset_idx(const arma::mat& x,
                            const arma::uvec& y) {

    // y must be an integer between 0 and rows - 1
    // Allows for repeated draws from same rows.
    return x.rows( y );
}

arma::vec select_mat_elements(const arma::mat& x,
                                const arma::uvec& row_idx,
                                const arma::uvec& col_idx) {
    arma::vec out(x.n_rows);
    // assumes both idx vectors are the same length
    for(int i=0; i < row_idx.size(); i++){
        out(i) = x(row_idx(i), col_idx(i));
    }
    return out;
}

arma::vec vector_subset_idx(const arma::vec& x,
                            const arma::uvec& y) {

    // y must be an integer between 0 and columns - 1
    // Allows for repeated draws from same columns.
    return x.elem( y );
}


arma::uvec r_like_order(const arma::vec& x, const arma::vec& y){
    std::vector<double> eu_dis = conv_to<std::vector<double>>::from(x);
    std::vector<double> noise_vec = conv_to<std::vector<double>>::from(y);
    int n = y.size();

    vector<int> index(n, 0);
    for (int i = 0 ; i != index.size() ; i++) {
        index[i] = i;
    }
    std::stable_sort(index.begin(), index.end(),
                     [&](const int& a, const int& b) {
                         if (eu_dis[a] != eu_dis[b]){
                             return eu_dis[a] < eu_dis[b];
                         }
                         return noise_vec[a] < noise_vec[b];
                     }
    );

    return(conv_to<arma::uvec>::from(index));
}


// [[Rcpp::export]]
arma::mat weight_mat_lfac_s_2_filter(int n, const arma::vec& ord, const arma::vec& s_vec, double n_prop, bool is_s_2){
    arma::mat out(n, s_vec.size());

    // for the s_val of each test observation...
    for(int i=0; i < s_vec.size(); i++){
        // Rcout << i<< std::endl;
        arma::vec weight_vec(n);
        double s_val = arma::as_scalar(s_vec[i]);

        // if s_val is > n/2 then we're just going to use 1-NN
        if((is_s_2) & (s_val > (double(n) *  n_prop))){
            // Rcout << "s_val > n * n_prop" << std::endl;
            weight_vec.zeros();
            out.col(i) = weight_vec;

        } else{
            // use fact that lfactorial(x) = lgamma(x+1)
            arma::vec n_ord = arma::lgamma( ((double(n)- ord) + 1.0) ); // first term
            arma::vec n_ord_s = arma::lgamma( ((double(n) - ord - s_val + 1.0) + 1.0) ); // last term
            double n_s_1 = lgamma((double(n) - s_val) + 1.0);
            double lfact_n = lgamma(double(n) + 1.0);

            weight_vec = arma::exp(n_ord + n_s_1 - lfact_n - n_ord_s);

            out.col(i) = weight_vec * s_val;
        }
    }
    return(out);
}

arma::mat weight_mat_lfac(int n, const arma::vec& ord, const arma::vec& s_vec){
    arma::mat out(n, s_vec.size());

    // for the s_val of each test observation...
    for(int i=0; i < s_vec.size(); i++){
        arma::vec weight_vec(n);
        double s_val = arma::as_scalar(s_vec[i]);
        // use fact that lfactorial(x) = lgamma(x+1)
        arma::vec n_ord = arma::lgamma( ((double(n)- ord) + 1.0) ); // first term
        arma::vec n_ord_s = arma::lgamma( ((double(n) - ord - s_val + 1.0) + 1.0) ); // last term
        double n_s_1 = lgamma((double(n) - s_val) + 1.0);
        double lfact_n = lgamma(double(n) + 1.0);

        weight_vec = arma::exp(n_ord + n_s_1 - lfact_n - n_ord_s);

        out.col(i) = weight_vec * s_val;
    }
    return out;
}

// [[Rcpp::export]]
double round_modified(const double& x){
    NumericVector x_rcpp = as<NumericVector>(wrap(x));

    x_rcpp = Rcpp::round(x_rcpp, 0);
    return as<double>(x_rcpp);
}

// [[Rcpp::export]]
arma::vec round_modified_vec(const arma::vec& x){
    NumericVector x_rcpp = as<NumericVector>(wrap(x));
    x_rcpp = Rcpp::round(x_rcpp, 0);
    return as<arma::vec>(wrap(x_rcpp));
}

// [[Rcpp::export]]
arma::vec arma_round(const arma::vec& x){
    return arma::round(x);
}


arma::vec rowMeans_arma(const arma::mat& x) {
    int nr = x.n_rows;
    // NumericVector ans(nc);
    arma::vec ans(nr);
    for (int j = 0; j < nr; j++) {
        arma::vec col_tmp = x.row(j);
        double mean = arma::mean(col_tmp);
        ans(j) = mean;
    }
    return ans;
}

arma::vec colSums_arma(const arma::mat& x) {
    int nc = x.n_cols;
    // NumericVector ans(nc);
    arma::vec ans(nc);
    for (int j = 0; j < nc; j++) {
        arma::vec col_tmp = x.col(j);
        double sum = arma::sum(col_tmp);
        ans(j) = sum;
    }
    return ans;
}

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

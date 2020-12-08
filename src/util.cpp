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

arma::mat matrix_subset_logical(arma::mat x,
                                arma::vec y) {
    // Assumes that y is 0/1 coded.
    // find() retrieves the integer index when y is equivalent 1.
    uvec subset_vec = find(y == 1) ;
    return x.cols(find(y == 1) );
}

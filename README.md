
# tdnn: Two-scale Distributional Nearest Neighbors

<!-- badges: start -->
[![R-CMD-check](https://github.com/patrickvossler18/tdnn_package/workflows/R-CMD-check/badge.svg)](https://github.com/patrickvossler18/tdnn_package/actions)
<!-- badges: end -->
### This package is still in beta and not yet intended for regular use 

The `tdnn` R package provides an implementation of the two-scale distributional nearest neighbors algorithm proposed in [Demirkaya et al., 2022](https://arxiv.org/abs/1808.08469).

## Installation

### Development version

At this time the `tdnn` package must be compiled from source to be used. The `tdnn` package is able to compile from source using the standard C++ compiler toolchains for R in MacOS X and Linux only. See https://www.rstudio.com/products/rpackages/devtools/ for details of the developer toolchains needed for your platform.

Once you have installed the appropriate toolchain, you can install the development version of the package using `devtools`:

``` r
if (!require("devtools")) install.packages("devtools")
devtools::install_github("patrickvossler18/tdnn_package")
```

## Tests
``` r
library(testthat)
test_package("tdnn")
```

## Example


``` r
library(tdnn)

dgp_function = function(x){(x[1]-1)^2 + (x[2]+1)^3 - 3*x[3]}
set.seed(1234)
n = 100
p = 3

X <- matrix(rnorm(n * p), n, p)
epsi <- matrix(rnorm(n) , n, 1)
Y <- apply(X, MARGIN = 1, dgp_function) + epsi
X_test <- matrix(c(0.5,-0.5,0.5),1,p)

tdnn_est <-tdnn_reg_tune(X, Y, X_test)

```

## Acknowledgements

The `tdnn` package makes use of the [libnabo](https://github.com/ethz-asl/libnabo) C++ library for its 1-NN calculations and we thank the authors for such a performant implementation. `tdnn` would not exist if not for the [Rcpp](https://cran.r-project.org/package=Rcpp), [RcppArmadillo](https://cran.r-project.org/package=RcppArmadillo), and [RcppParallel](https://cran.r-project.org/package=RcppParallel) packages - many thanks to their authors! 

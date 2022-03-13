## usethis namespace: start
#' @useDynLib tdnn, .registration = TRUE
#' @importFrom RcppParallel RcppParallelLibs
#' @import Rcpp
## usethis namespace: end
NULL

## quiets concerns of R CMD check re: the s_1's that appear in pmap_df anonymous functions in tune_de_dnn.R
utils::globalVariables(c("s_1"))

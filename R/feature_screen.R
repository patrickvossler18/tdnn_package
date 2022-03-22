#' Screen features using distance correlation t-test
#'
#' @param X Matrix of covariates
#' @param Y Matrix of responses
#' @param alpha Threshold for the p-value from the distance correlation t-test. Default is 0.001
#' @importFrom energy dcor2d
#' @importFrom stats pt
#' @export
feature_screen <- function(X,Y, alpha = 0.001){
    n <- nrow(X)
    p <- ncol(X)
    Y_vec <- as.numeric(Y)
    sapply(1:p, function(i){
        X_i <- X[,i]
        bcR <- energy::dcor2d(X_i, Y_vec, type = "U")
        M <- n * (n-3) / 2
        df <- M - 1
        tstat <-  sqrt(M-1) * bcR / sqrt(1-bcR^2)
        as.numeric((1 - stats::pt(tstat, df=df)) < alpha / p)
    })

}

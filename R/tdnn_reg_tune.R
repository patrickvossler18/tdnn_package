#' Get a tuned tdnn estimate
#'
#' @param X Matrix of covariates
#' @param Y Matrix of responses
#' @param X_test Matrix of test observations for which we want to get estimates
#' @param W_0 Optional integer vector with 1 corresponding to columns that should be used for estimation. Default value is NULL
#' @param c Sequence of potential values for \eqn{c}. If a sequence is not provided, the default is 2.
#' @param B_NN Number of nearest neighbors to use during tuning. Default is 20.
#' @param n_prop If \eqn{s_{2} > n\dot \text{n_prop}}, default to using 1-NN estimate.
#' @param n_threads Number of threads to use when calculating bootstrap variance. Default is to use all available threads.
#' @param verbose Print which step the method is currently calculating in the console as well as debug information.
#' @importFrom glue glue
#' @import dplyr
#' @export

tdnn_reg_tune <- function(X,
                        Y,
                        X_test,
                        W_0 = NULL,
                        c_seq = 2,
                        B_NN = 20,
                        n_prop = 0.5,
                        estimate_variance = F,
                        bootstrap_iter = 1000,
                        n_threads = NULL,
                        verbose = F) {
    # Data checks before we doing anything else
    # Check X is a dataframe or matrix. If df, make it a matrix
    if (is.data.frame(X)) {
        warning(glue::glue("X is a dataframe. Attempting to convert to numeric matrix"))
        X <- as.matrix(X)
    }

    # Need all numeric columns for X matrix.
    if(!is.numeric(X)){
        stop(glue::glue("X is {glue::glue_collapse(class(X),', ')} but needs to be a matrix"))
    }

    if (!is.matrix(Y)) {
        stop(glue("Y is {glue::glue_collapse(class(Y),', ')} but needs to be a matrix"))
    }

    if (!is.matrix(X_test)) {
        stop(glue("X_test is {glue::glue_collapse(class(X_test),', ')} but needs to be a matrix"))
    }

    # Check that dimensions match up for X, Y, X_test, and W_0
    # need length of Y to match dim[1] of X
    # need length of W_0 to match dim[2] of X
    # X_test can be of different length but ncols of X and X_test must match
    dimension_checks <- c(
        dim(X)[1] == length(Y),
        dim(X)[2] == length(W_0),
        dim(X)[2] == dim(X_test)[2]
    )
    if (!all(dimension_checks)) {
        # get which checks failed
        failed_checks <- which(!(dimension_checks))
        fail_messages <- sapply(failed_checks, function(x) {
            switch(EXPR=x,
                   glue::glue("nrow of X ({dim(X)[1]}) does not equal length of Y: {length(Y)}"),
                   glue::glue("ncol of X ({dim(X)[2]}) does not equal length of W_0: {length(W_0)}"),
                   glue::glue("ncol of X ({dim(X)[2]}) does not equal ncol of X_test: {dim(X_test)[2]}")
            )
        })
        stop(glue::glue("Dimensions don't match: \n {toString(fail_messages)}"))
    }

    if (!is.null(W_0)) {
        stopifnot("The elements of W_0 must be either 0 or 1" = all(W_0 %in% 0:1))
        if (is.logical(W_0)) W_0 <- as.numeric(W_0)
    }

    if(!is.null(n_threads)){
        RcppParallel::setThreadOptions(numThreads = n_threads)
    }

    n <- nrow(X)
    tuned_tdnn_results <-
        tdnn:::tune_de_dnn_no_dist_cpp(
            X, Y, X_test,
            W_0, c=cc, B_NN=B_NN, n_prop=n_prop,
            estimate_variance= estimate_variance,
            bootstrap_iter=bootstrap_iter,
            debug=verbose)

    return(tuned_tdnn_results)
}

make_results_df <- function(truth, predictions, variance = NULL) {
    if (!is.null(variance)) {
        return(data.frame(truth = truth, predictions = predictions, variance = variance))
    } else{
        return(data.frame(truth = truth, predictions = predictions))
    }
}


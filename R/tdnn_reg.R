#' Estimate a regression function using the two-scale DNN estimator
#'
#' @param X matrix of covariates
#' @param Y matrix of responses
#' @param X_test matrix of test observations
#' @param W_0 Optional integer vector with 1 corresponding to columns that should be used to estimate the treatment effect. Default value is NULL
#' @param c Parameter that controls the size of the ratio of \eqn{s_1} and \eqn{s_2} through the equation
#' \eqn{s_2 = s_1 \cdot c^{d/2}}. Default value is 0.80. Also determines the weights of the s_1 and s_2 estimates
#' @param n_prop How large of a value relative to the sample size before we just use 1-NN to give us an estimate.
#' @param C_s_2 Constant term for s_2 = C_s_2 * n^(d/(d+8)) where d is ambient dimension and n is sample size. Default is 2
#' @param tuning_method Choose the method used to choose the subsample size \eqn{s} ("early stopping", "sequence").
#'  The default is the "early stopping" method which stops as soon as a sign change in the difference of the estimates is detected,
#'   while the "sequence" method calculates an estimate for each of the different s values
#'    (a sequence from 1 to 50) and then finds the s which causes a change in the sign of the derivative.
#' @param estimate_variance Bootstrap variance estimates using the \code{boot} library
#' @param verbose Print which step the method is currently calculating in the console.
#' @importFrom glue glue
#' @importFrom strex match_arg
#' @export
tdnn_reg <- function(X,
                     Y,
                     X_test,
                     W_0 = NULL,
                     c = 0.80,
                     n_prop = 0.5,
                     C_s_2 = 2.0,
                     tuning_method = "early stopping",
                     estimate_variance = F,
                     use_boot = F,
                     bootstrap_iter = 1000,
                     verbose = F,
                     n_threads = 4,
                       ...) {
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

    # Do argument matching for tuning method
    matched_tuning_method <- strex::match_arg(tuning_method, c("early stopping", "sequence"))

    RcppParallel::setThreadOptions(numThreads = n_threads)

    # After doing data checks, pass off to CPP code.
    deDNN <- est_reg_fn_mt_rcpp(
        X,
        Y,
        X_test,
        c = c,
        verbose = verbose,
        n_prop = n_prop,
        C_s_2 = C_s_2,
        W0_ = W_0
    )
    boot_vals <- NULL
    variance <- NULL
    if (estimate_variance) {
        if(verbose){
            message("starting variance estimation...")
        }
        if(use_boot){
            # prepare arguments to pass to boot function
            # bootstrap_iter -> R
            R <- bootstrap_iter

            boot_data <- data.frame(X, Y)
            d <- dim(X)[2]
            boot_estimates <- boot::boot(
                data = boot_data,
                statistic = boot_function_est_reg_fn,
                X_test = X_test,
                s_choice = deDNN$s,
                W_0 = W_0,
                c = c,
                n_prop = n_prop,
                C_s_2 = C_s_2,
                d = d,
                R = R,
                ...
            )

            variance <- apply(boot_estimates$t,2, var)
        } else {
            B <- bootstrap_iter
            if (verbose){
                message(paste0("B is ", B))
            }

            boot_estimates <- tdnn:::bootstrap_cpp_mt(X,Y,X_test,
                                                      s_choice= deDNN$s,
                                                      c=c,
                                                      n_prop=n_prop,
                                                      C_s_2 = C_s_2,
                                                      B=B,
                                                      W0_ = W_0)
            # boot_estimates <- tdnn:::bootstrap_reg_fn(X,Y,X_test,
            #                                           s_choice = deDNN$s,
            #                                           W_0=W_0,
            #                                           c=c,
            #                                           B=B,
            #                                           n_prop=n_prop,
            #                                           verbose=verbose)
            boot_vals <- boot_estimates
            variance <- apply(boot_estimates,1, var)
        }


        if(verbose){
            message("finished estimating variance")
        }


    }

    results <- structure(list(), class = "tdnn_regression_est")
    results[["deDNN_pred"]] <- deDNN$estimates
    results[["s_choice"]] <- deDNN$s
    results[["variance"]] <- variance
    results[["boot_vals"]] <- boot_vals
    return(results)
}

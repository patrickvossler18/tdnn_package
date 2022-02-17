#' Get a tuned tdnn estimate
#'
#' @param X Matrix of covariates
#' @param Y Matrix of responses
#' @param X_test Matrix of test observations for which we want to get estimates
#' @param tuning_iter Number of iterations for MC resampling when tuning parameters. Default is 200
#' @param W_0 Optional integer vector with 1 corresponding to columns that should be used for estimation. Default value is NULL
#' @param c_seq Sequence of potential values for \eqn{c}. If a sequence is not provided, the default is \code{seq(0.1, 0.9, 0.1)}
#' @param M_seq Sequence of potential values for \eqn{M}. If a sequence is not provided, the default is \code{seq(0.5, 2, 0.25)}
#' @param n_prop If \eqn{s_{2} > n\dot \text{n_prop}}, default to using 1-NN estimate.
#' @param n_threads Number of threads to use when calculating bootstrap variance. Default is to use all available threads.
#' @param verbose Print which step the method is currently calculating in the console.
#' @importFrom glue glue
#' @import dplyr
#' @export

tdnn_reg_cv <- function(X,
                        Y,
                        X_test,
                        tuning_iter= 200,
                        W_0 = NULL,
                        M_seq = NULL,
                        c_seq = NULL,
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
    # handle case where M_seq or c_seq are NULL
    if (is.null(M_seq)) {
        M_seq <- seq(0.5, 2, 0.25)
        M_seq_text <- deparse(quote(seq(0.5, 2, 0.25)))
        message(glue::glue("Using default M sequence: {M_seq_text}"))

    }

    if (is.null(c_seq)) {
        c_seq <- seq(0.1, 0.9, 0.1)
        c_seq_text <- deparse(quote(seq(0.1, 0.9, 0.1)))
        message(glue::glue("Using default c sequence: {c_seq_text}"))

    }
    # generate the matrix of tuning parameters
    param_mat <- as.matrix(tidyr::expand_grid(c_val = c_seq, M =  M_seq))
    if (verbose) {
        message("tuning parameters...")
    }
    tuned_tdnn_results <-
        tdnn:::tdnn_reg_cv_cpp(X, Y, X_test, param_mat, n_prop,
                               tuning_iter,
                               bootstrap_iter,
                               estimate_variance, verbose, W_0)

    return(tuned_tdnn_results)
}

make_results_df <- function(truth, predictions, variance = NULL) {
    if (!is.null(variance)) {
        return(data.frame(truth = truth, predictions = predictions, variance = variance))
    } else{
        return(data.frame(truth = truth, predictions = predictions))
    }
}

#' Tune tdnn_reg parameters c and M
#'
#' @param X_train matrix of covariates
#' @param Y_train matrix of responses
#' @param X_val matrix of validation observations
#' @param Y_val matrix of validation responses
#' @param W_0 Optional integer vector with 1 corresponding to columns that should be used to estimate the treatment effect. Default value is NULL
#' @param c_seq Sequence of potential values for \eqn{c}. If a sequence is not provided, the default is \code{seq(0.1, 0.9, 0.1)}
#' @param M_seq Sequence of potential values for \eqn{M}. If a sequence is not provided, the default is \code{seq(0.5, 2, 0.25)}
#' @param n_threads Number of threads to use when fitting the tdnn_reg method. Default is 4 threads.
#' @param verbose Print which step the method is currently calculating in the console.
#' @importFrom glue glue
#' @import dplyr
#' @export
tune_c_s2 <- function(X_train,
                      Y_train,
                      X_val,
                      Y_val,
                      W_0,
                      c_seq,
                      M_seq,
                      n_threads,
                      verbose,
                      make_summary) {
    seq_df <- tidyr::expand_grid(c_val = c_seq, M =  M_seq)

    tuning_results <- purrr::pmap(seq_df, function(c_val, M) {
        if (verbose) {
            print(glue::glue("c: {c_val}, M: {M}"))
        }
        tdnn_pred_rand_x <- tdnn:::est_reg_fn_mt_rcpp(
            X_train,
            Y_train,
            X_val,
            c = c_val,
            verbose = FALSE,
            n_prop = 0.5,
            C_s_2 = M,
            W0_ = W_0
        )
        list(truth = as.numeric(Y_val), predictions = as.numeric(tdnn_pred_rand_x$estimates), c=c_val, M=M)
    })

    if (make_summary) {
        tuning_results <- tuning_results %>% bind_rows() %>%
            group_by(c, M) %>%
            summarise(MSE = mean((predictions - Y_val) ^ 2)) %>%
            ungroup() %>%
            filter(MSE == min(MSE))
        if (verbose) {
            print(tuning_results)
        }
        return(list(c = tuning_results$c, M = tuning_results$M))
    } else{
        return(tuning_results)
    }
}




tdnn_reg_split <- function(X,
                           Y,
                           X_test,
                           W_0 = NULL,
                           M_seq = NULL,
                           c_seq = NULL,
                           validation_pct = 0.2,
                           n_prop = 0.5,
                           s_1_tuning_method = "early stopping",
                           estimate_variance = F,
                           bootstrap_iter = 1000,
                           n_threads = NULL,
                           verbose = F) {
    # handle case where M_seq or c_seq are NULL
    if (is.null(M_seq)) {
        M_seq <- seq(0.5, 2, 0.25)
        M_seq_text <- deparse(quote(seq(0.5, 2, 0.25)))
        message(glue::glue("Using default M sequence: {M_seq_text}"))

    }

    if (is.null(c_seq)) {
        c_seq <- seq(0.1, 0.9, 0.1)
        c_seq_text <- deparse(quote(seq(0.1, 0.9, 0.1)))
        message(glue::glue("Using default c sequence: {c_seq_text}"))

    }

    # split train data into train and validation
    n <- nrow(X)
    train_idx <- sample(n, n *(1-validation_pct))
    X_train <- X[train_idx, ]
    X_val <- X[-train_idx, ]

    Y_train <- as.matrix(Y[train_idx, ])
    Y_val <- as.matrix(Y[-train_idx, ])


    if (verbose) {
        message("tuning parameters...")
    }
    # returns a list with c and M values that minimized MSE for validation set
    tuned_params <- tune_c_s2(
        X_train,
        Y_train,
        X_val,
        Y_val,
        W_0,
        c_seq,
        M_seq,
        n_threads = n_threads,
        verbose = verbose,
        make_summary = TRUE
    )
    if (verbose) {
        message(
            glue::glue(
                "Estimating using c = {glue::glue_collapse(tuned_params$c)}, M={glue::glue_collapse(tuned_params$M)}..."
            )
        )
    }
    tuned_tdnn_results <- tdnn_reg(
        X,
        Y,
        X_test,
        W_0,
        c = tuned_params$c,
        C_s_2 = tuned_params$M,
        n_prop = n_prop,
        tuning_method = s_1_tuning_method,
        estimate_variance = estimate_variance,
        bootstrap_iter = bootstrap_iter,
        n_threads = n_threads,
        verbose = verbose
    )
    tuned_tdnn_results[["c"]] <- tuned_params$c
    tuned_tdnn_results[["M"]] <- tuned_params$M
    return(tuned_tdnn_results)
}

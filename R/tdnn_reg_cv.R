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

    tuning_results <- purrr::pmap_dfr(seq_df, function(c_val, M) {
        if (verbose) {
            print(glue::glue("c: {c_val}, M: {M}"))
        }
        tdnn_pred_rand_x <- tdnn_reg(
            X_train,
            Y_train,
            X_val,
            W_0 = W_0,
            n_prop = 0.5,
            c = c_val,
            C_s_2 = M,
            estimate_variance = FALSE,
            n_threads = n_threads,
            verbose = FALSE
        )
        results_df <- make_results_df(Y_val,
                        as.numeric(tdnn_pred_rand_x$deDNN_pred))
        colnames(results_df) <- c("truth", "predictions")
        results_df %>% mutate(c = c_val, M = M)
    }) %>% bind_rows()

    if (make_summary) {
        tuning_results <- tuning_results %>%
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

loo_sample <- function(num_rows, B = 200) {
    a <- map(seq(B), function(x) {
        idx <- seq(num_rows)
        held_out_idx <- sample(idx, 1)
        list(train_idx = idx[-held_out_idx], held_out_idx = held_out_idx)
    })
}

tdnn_reg_cv <- function(X,
                        Y,
                        X_test,
                        B,
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
    # generate our loo_samples
    loo_samples <- loo_sample(n, B)
    # loop through the loo_samples and tune for each
    if (verbose) {
        message("tuning parameters...")
    }
    tuned_params <- bind_rows(lapply(1:B, function(i) {
        if(verbose){
            print(glue::glue("i: {i}/{B}"))
        }

        train_idx <- loo_samples[[i]]$train_idx

        X_train <- X[train_idx, ]
        X_val <- matrix(X[-train_idx, ],1,ncol(X))

        Y_train <- as.matrix(Y[train_idx, ])
        Y_val <- as.matrix(Y[-train_idx, ])

        # returns dataframe with MSE for each parameter combo for this split
        tictoc::tic("tuning params on loo sample")
        tuned_params <- tune_c_s2(
            X_train,
            Y_train,
            X_val,
            Y_val,
            W_0,
            c_seq,
            M_seq,
            n_threads = n_threads,
            verbose = F,
            make_summary = F
        )
        tictoc::toc()

        return(tuned_params)
    }))
    tuned_params <- tuned_params %>%
        group_by(c,M) %>%
        summarise(MSE = mean((predictions - truth) ^ 2)) %>%
        ungroup() %>%
        filter(MSE == min(MSE))
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

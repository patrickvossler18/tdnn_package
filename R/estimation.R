#' Estimate a treatment effect using the two-scale DNN estimator
#'
#' @param X Matrix of covariates
#' @param W Matrix of treatment assignments
#' @param Y Matrix of responses
#' @param X_test Matrix of test observations
#' @param W_0 Optional integer vector with 1 corresponding to columns that should be used to estimate the treatment effect. Default value is NULL
#' @param c Parameter that controls the size of the ratio of \eqn{s_1} and \eqn{s_2} through the equation
#' \eqn{s_2 = s_1 \cdot c^{d/2}}. Default value is 0.80
#' @param tuning_method Choose the method used to choose the subsample size \eqn{s} ("early stopping", "sequence").
#'  The default is the "early stopping" method which stops as soon as a sign change in the difference of the estimates is detected,
#'   while the "sequence" method calculates an estimate for each of the different s values
#'    (a sequence from 1 to 50) and then finds the s which causes a change in the sign of the derivative.
#' @param estimate_var Boolean for estimating variance using bootstrap. Default value is False.
#' @param feature_screening Boolean for performing feature screening step. Default value is True. If W_0 is NULL and feature_screening is False, all columns of X are used.
#' using the energy package's (Rizzo and Szekely, 2019) `dcor.test`.
#' @param use_boot Boolean for whether to use the C++ bootstrap implemented by this package or the boot function from the boot package.
#' @param alpha Threshold value for multiple testing correction used in the feature screening step
#' @param B Number of bootstrap replicates used for calculating confidence intervals using the C++ bootstrap implemented by this package. Default B = 1000
#' @param n_threads Number of threads used for parallel C++ code. Default is 4 threads
#' @param n_cores_feature_screen Number of cores used for feature screening. Default is NULL and if the argument is NULL, only one core is used.
#' @param verbose Control verbosity of the status of the calculations involved in the function. Set to false by default.
#' @param ... Extra arguments to be passed to the boot function when estimating variance (e.g. ncpus, parallel, R)
#'
#' @importFrom glue glue
#' @importFrom strex match_arg
#' @export
est_effect <- function(X,
                       W,
                       Y,
                       X_test,
                       c = 0.8,
                       W_0 = NULL,
                       tuning_method = "early stopping",
                       estimate_var = F,
                       feature_screening = T,
                       use_boot = F,
                       alpha = 0.001,
                       B = 1000,
                       n_threads = 4,
                       n_prop = 0.5,
                       n_cores_feature_screen = NULL,
                       verbose = FALSE,
                       ...) {

  # Data checks before we doing anything else
  # Check X is a dataframe or matrix. If df, make it a matrix
  if (is.data.frame(X)) {
    warning(glue::glue("X is a dataframe. Attempting to convert to numeric matrix"))
    X <- as.matrix(X)
  }

  # Need all numeric columns for X matrix.
  # This might not be a necessary check since R coerces to numeric for matrix?
  # if (!all(sapply(X, is.numeric))) {
  #   non_num_cols <- which(!(sapply(X, is.numeric)))
  #   stop(glue::glue("Found non-numeric columns. Column indices: {non_num_cols}"))
  # }

  if (!is.matrix(Y)) {
    stop(glue("Y is of class {class(Y)[1]}, but needs to be a matrix"))
  }

  if (!is.matrix(X_test)) {
    stop(glue("X_test is of class {class(X_test)[1]}, but needs to be a matrix"))
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
  # if W_0 is null, length will be zero, but we don't want to stop here because of that.
  if (is.null(W_0)){
    dimension_checks[2] = T
  }

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

  # Do argument matching for tuning method
  matched_tuning_method <- strex::match_arg(tuning_method, c("early stopping", "sequence"))

  # If all of the checks have passed, set the number of threads
  RcppParallel::setThreadOptions(numThreads = n_threads)

  # Check if W_0 is null and feature screening is false.
  # If we are given a feature screening vector and feature_screening is true, we will always use the vector supplied by the user
  if (feature_screening) {
    if(verbose){
      message("starting feature screening...")
    }
    if (!is.null(W_0)) {
      message("Using user supplied feature screening vector")
      stopifnot("The elements of W_0 must be either 0 or 1" = all(W_0 %in% 0:1))
      if (is.logical(W_0)) W_0 <- as.numeric(W_0)
    } else {
      W_0 <- screen_features(X, Y, alpha = alpha, n_cores= n_cores_feature_screen)
    }
  } else if (!is.null(W_0)) {
    stopifnot("The elements of W_0 must be either 0 or 1" = all(W_0 %in% 0:1))
    if (is.logical(W_0)) W_0 <- as.numeric(W_0)

    if(verbose){
      message("finished feature screening")
    }
  }


  if(verbose){
    message("estimating effect_0...")
  }

  effect_0 <- est_reg_fn_mt_rcpp(
    X[W == 0,],
    matrix(Y[W == 0]),
    X_test,
    c = c,
    n_prop = n_prop,
    W0_ = W_0,
    verbose = verbose
  )
    # est_reg_fn(X[W == 0, ], matrix(Y[W == 0]), X_test, W_0, c, tuning_method, n_threads)
    # est_reg_fn(X[W == 0, ], matrix(Y[W == 0]), X_test, W_0, c, tuning_method)
  deDNN_pred_0 <- effect_0$estimates
  s_choice_0 <- effect_0$s
  if(verbose){
    message("finished estimating effect_0")
  }

  if(verbose){
    message("estimating effect_1...")

  }

  effect_1 <- est_reg_fn_mt_rcpp(
    X[W == 1,],
    matrix(Y[W == 1]),
    X_test,
    c = c,
    n_prop = n_prop,
    W0_ = W_0,
    verbose = verbose
  )
  # effect_1 <-
    # est_reg_fn(X[W == 1, ], matrix(Y[W == 1]), X_test, W_0, c, tuning_method, n_threads)
    # est_reg_fn(X[W == 1, ], matrix(Y[W == 1]), X_test, W_0, c, tuning_method)
  deDNN_pred_1 <- effect_1$estimates
  s_choice_1 <- effect_1$s

  if(verbose){
    message("finished estimating effect_1")
  }

  deDNN_pred <- as.numeric(deDNN_pred_1 - deDNN_pred_0)

  variance <- NULL
  var_obj <- NULL
  conf_int <- NULL
  if (estimate_var) {
    if(verbose){
      message("starting estimating variance...")
    }
    boot_est <-
      est_variance(X,
        W,
        Y,
        X_test,
        W_0,
        s_choice_0,
        s_choice_1,
        c,
        B = B,
        n_prop = n_prop,
        use_boot = use_boot,
        verbose = verbose,
        ...
      )
    if (use_boot) {

      # variance <- mean((boot_est$t - deDNN_pred)^2)
      var_obj <- boot_est
      var_t0 <- apply(var_obj$t,2, var)
      variance <- var_t0
      mean_t <- apply(var_obj$t,2, mean)
      merr <- sqrt(var_t0) * qnorm((1+0.95)/2)
      bias <- as.numeric(mean_t - var_obj$t0)
      conf_int <- cbind(deDNN_pred - bias - merr, deDNN_pred - bias + merr)
      # variance <- apply((boot_est$t - deDNN_pred)^2, 2, mean )
    } else {
      # boot_est has dim ncol(X_test) x B
      var_t0 <- apply(boot_est,1, var)
      mean_t <- apply(boot_est,1, mean)
      variance <- var_t0
      # to-do allow for different confidence levels
      merr <- sqrt(var_t0) * qnorm((1+0.95)/2)
      bias <- as.numeric(mean_t - deDNN_pred)
      conf_int <- cbind(deDNN_pred - bias - merr, deDNN_pred - bias + merr)
    }

    if(verbose){
      message("finished estimating variance")
    }


  }

  # Be considerate and reset RcppParallel thread options
  RcppParallel::setThreadOptions(numThreads = "auto")
  results_list <- structure(list(), class = "tdnn_causal_est")

  results_list[["estimate"]] <- deDNN_pred
  results_list[["s_choice"]] <- c(s_choice_0 = s_choice_0, s_choice_1 = s_choice_0)
  results_list[["variance"]] <- variance
  # results_list[["varobj"]] <- var_obj
  # results_list[["conf_int"]] <- conf_int

  results_list
}


est_variance <- function(X, W, Y, X_test, W_0, s_choice_0, s_choice_1,
                         c, B, n_prop, use_boot = F, ...) {
  args <- list(...)
  verbose <- ifelse(is.null(args[['verbose']]), FALSE, args[['verbose']])
  if (use_boot) {
    boot_data <- data.frame(X, W, Y)
    d <- dim(X)[2]

    boot_estimates <- boot::boot(
      data = boot_data,
      statistic = boot_function_trt_effect,
      s_choice_0 = s_choice_0,
      s_choice_1 = s_choice_1,
      W_0 = W_0,
      c = c,
      n_prop = n_prop,
      strata = W,
      ...
    )
  } else {
    boot_estimates <- bootstrap_cpp(X, Y, W, X_test,s_choice_0,
      s_choice_1,
      c=c,
      n_prop = n_prop,
      W_0 = W_0,
      B = B,
      verbose = verbose
    )
    # boot_estimates <- as.numeric(bootstrap_cpp_mt(X, Y, X_test, W,
    #                                    s_choice_0, s_choice_1, W0,
    #                                    c, B))
  }
  boot_estimates
}


screen_features <- function(X, Y, alpha, n_cores) {
  if (is.null(alpha)) {
    alpha <- 0.001
  }

  # feature_screening_parallel(X, Y, alpha)
  dcorT_parallel(X, Y, alpha, n_cores)
}

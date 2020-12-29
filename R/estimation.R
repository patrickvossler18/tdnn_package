#' Estimate a regression function using the two-scale DNN estimator
#'
#' @param X matrix of covariates
#' @param Y matrix of responses
#' @param X_test matrix of test observations
#' @param W_0 Optional integer vector with 1 corresponding to columns that should be used to estimate the treatment effect. Default value is NULL
#' @param c Parameter that controls the size of the ratio of \eqn{s_1} and \eqn{s_2} through the equation
#' \eqn{s_2 = s_1 \cdot c^{d/2}}. Default value is 0.80
#' @param tuning_method Choose the method used to choose the subsample size \eqn{s} ("greedy", "sequence").
#'  The default is the "greedy" method which stops as soon as a sign change in the difference of the estimates is detected,
#'   while the "sequence" method calculates an estimate for each of the different s values
#'    (a sequence from 1 to 50) and then finds the s which causes a change in the sign of the derivative.
#' @export
est_reg_fn <- function(X,
                 Y,
                 X_test,
                 W_0 = NULL,
                 c = 0.33,
                 tuning_method = "greedy") {
  # Data checks before we doing anything else
  # Check X is a dataframe or matrix. If df, make it a matrix
  if (is.data.frame(X)) {
    warning(glue::glue("X is a dataframe. Attempting to convert to numeric matrix"))
    X <- as.matrix(X)
  }

  # Need all numeric columns for X matrix.
  # This might not be a necessary check since R coerces to numeric for matrix?
  if (!all(sapply(X, is.numeric))) {
    non_num_cols <- which(!(sapply(X, is.numeric)))
    stop(glue::glue("Found non-numeric columns. Column indices: {non_num_cols}"))
  }

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

  if(!is.null(W_0)){
    stopifnot("The elements of W_0 must be either 0 or 1" = all(W_0 %in% 0:1))
    if (is.logical(W_0)) W_0 = as.numeric(W_0)
  }


    if(tuning_method == "greedy"){
      s_choice <- tuning(X, Y, X_test, c=c, W0_ = W_0)
    } else if( tuning_method == "sequence"){
      s_choice <- tuning_st(seq(1,50,1),X, Y, X_test, c=c, W0_ = W_0)
    }
    a_pred <- de_dnn(
      X,
      Y,
      X_test = X_test,
      s_sizes = s_choice,
      c = c,
      W0_ = W_0
    )$estimates

    b_pred <- de_dnn(
      X,
      Y,
      X_test = X_test,
      s_sizes = s_choice + 1,
      c = c,
      W0_ = W_0
    )$estimates

    deDNN_pred <- (a_pred + b_pred) / 2

    list(deDNN_pred = deDNN_pred, s_choice = s_choice)
}


#' Estimate a treatment effect using the two-scale DNN estimator
#'
#' @param X Matrix of covariates
#' @param W Matrix of treatment assignments
#' @param Y Matrix of responses
#' @param X_test Matrix of test observations
#' @param W_0 Optional integer vector with 1 corresponding to columns that should be used to estimate the treatment effect. Default value is NULL
#' @param c Parameter that controls the size of the ratio of \eqn{s_1} and \eqn{s_2} through the equation
#' \eqn{s_2 = s_1 \cdot c^{d/2}}. Default value is 0.80
#' @param tuning_method Choose the method used to choose the subsample size \eqn{s} ("greedy", "sequence").
#'  The default is the "greedy" method which stops as soon as a sign change in the difference of the estimates is detected,
#'   while the "sequence" method calculates an estimate for each of the different s values
#'    (a sequence from 1 to 50) and then finds the s which causes a change in the sign of the derivative.
#' @param estimate_var Boolean for estimating variance using bootstrap. Default value is False.
#' @param feature_screening Boolean for performing feature screening step. Default value is True. If W_0 is NULL and feature_screening is False, all columns of X are used.
#' using the energy package's (Rizzo and Szekely, 2019) `dcor.test`.
#' @param use_boot Boolean for whether to use the C++ bootstrap implemented by this package or the boot function from the boot package.
#' @param alpha Threshold value for multiple testing correction used in the feature screening step
#' @param B Number of bootstrap replicates used for calculating confidence intervals using the C++ bootstrap implemented by this package. Default B = 1000
#' @param ... Extra arguments to be passed to the boot function when estimating variance (e.g. ncpus, parallel, R)
#'
#' @importFrom glue glue
#' @export
est_effect <- function(X,
                       W,
                       Y,
                       X_test,
                       c=0.8,
                       W_0=NULL,
                       tuning_method = "greedy",
                       estimate_var = F,
                       feature_screening= T,
                       use_boot = F,
                       alpha=0.001,
                       B = 1000,
                       ...) {

  # Data checks before we doing anything else
  # Check X is a dataframe or matrix. If df, make it a matrix
  if (is.data.frame(X)) {
    warning(glue::glue("X is a dataframe. Attempting to convert to numeric matrix"))
    X <- as.matrix(X)
  }

  # Need all numeric columns for X matrix.
  # This might not be a necessary check since R coerces to numeric for matrix?
  if (!all(sapply(X, is.numeric))) {
    non_num_cols <- which(!(sapply(X, is.numeric)))
    stop(glue::glue("Found non-numeric columns. Column indices: {non_num_cols}"))
  }

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

  # Check if W_0 is null and feature screening is false.
  # If we are given a feature screening vector and feature_screening is true, we will always use the vector supplied by the user
  if(feature_screening){
    if(!is.null(W_0)){
      message("Using user supplied feature screening vector")
      stopifnot("The elements of W_0 must be either 0 or 1" = all(W_0 %in% 0:1))
      if (is.logical(W_0)) W_0 = as.numeric(W_0)
    } else{
      W_0 <- screen_features(X, Y, alpha=alpha)
    }
  } else if(!is.null(W_0)){
    stopifnot("The elements of W_0 must be either 0 or 1" = all(W_0 %in% 0:1))
    if (is.logical(W_0)) W_0 = as.numeric(W_0)
  }




  effect_0 <-
    est_reg_fn(X[W == 0, ], matrix(Y[W == 0]), X_test, W_0, c, tuning_method)
  deDNN_pred_0 <- effect_0$deDNN_pred
  s_choice_0 <- effect_0$s_choice

  effect_1 <-
    est_reg_fn(X[W == 1, ], matrix(Y[W == 1]), X_test, W_0, c, tuning_method)
  deDNN_pred_1 <- effect_1$deDNN_pred
  s_choice_1 <- effect_1$s_choice

  deDNN_pred <- deDNN_pred_1 - deDNN_pred_0

  if( estimate_var){
    boot_est <-
      est_variance(X,
                   W,
                   Y,
                   X_test,
                   W_0,
                   s_choice_0,
                   s_choice_1,
                   c,
                   B=B,
                   use_boot = use_boot,
                   ...)

    list(estimate = deDNN_pred,
         variance =  if (use_boot)
           mean((boot_est$t - deDNN_pred) ^ 2)
         else
           mean((boot_est - deDNN_pred) ^ 2))
  } else{
      list(estimate = deDNN_pred)
  }

}


est_variance <- function(X, W, Y, X_test, W_0, s_choice_0, s_choice_1,
                         c, B, use_boot=F, ...){
    if(use_boot){
      boot_data <- data.frame(X, W, Y)
      d <- dim(X)[2]
      boot_function_cpp <- function(dat, idx, s_choice_0, s_choice_1, W_0, c) {
        # subsample the indices and of those split in to treated and control groups
        X <- as.matrix(dat[idx, 1:d])
        W <- dat$W[idx]
        Y <- dat$Y[idx]

        # split into groups
        trt_est <- td_dnn(X[W == 1, ],
                          matrix(Y[W == 1]),
                          X_test = X_test,
                          s_choice = s_choice_1,
                          c = c,
                          W_0 = W_0
        )
        ctrl_est <- td_dnn(X[W == 0, ],
                           matrix(Y[W == 0]),
                           X_test = X_test,
                           s_choice = s_choice_0,
                           c = c,
                           W_0 = W_0
        )
        trt_est - ctrl_est
      }
      boot_estimates <- boot::boot(
        data = boot_data,
        statistic = boot_function_cpp,
        s_choice_0 = s_choice_0,
        s_choice_1 = s_choice_1,
        W_0 = W_0,
        c = c,
        ...
      )
    } else{
      boot_estimates <- bootstrap_cpp(X, X_test,Y, W, W_0,s_choice_0,
                                             s_choice_1, c, B=B)
    }
    boot_estimates
}


screen_features <- function(X, Y, alpha){
  if(is.null(alpha)){alpha = 0.001}
  feature_screening_parallel(X, Y, alpha)
  # feature_screening(X, Y, alpha)
  # p0 <- ncol(X)
  # sapply(1:p0, function(i){
  #     as.numeric(energy::dcor.test(X[, i], Y)$p.value < alpha/p0)
  # })
}

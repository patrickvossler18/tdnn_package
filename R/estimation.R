#' Estimate a treatment effect using the two-scale DNN estimator
#'
#' @param X matrix of covariates
#' @param Y matrix of responses
#' @param X_test matrix of test observations
#' @param W_0 optional Boolean feature screening vector
#' @param t max size of tuning sequence. Default is 50
#'
#' @importFrom glue glue
tdnn <- function(X,
                 Y,
                 X_test,
                 W_0 = NULL,
                 t = 50) {
    s_choice0 <- tuning(seq(1, t, 1), X, Y, X_test, 2, W0_ = W_0)

  deDNN_pred <- td_dnn(X,
    Y,
    X_test = X_test,
    s_choice = s_choice0,
    W_0 = W_0
  )

  list(deDNN_pred = deDNN_pred, s_choice = s_choice0)
}


#' Estimate a treatment effect using the two-scale DNN estimator
#'
#' @param X Matrix of covariates
#' @param W Matrix of treatment assignments
#' @param Y Matrix of responses
#' @param X_test Matrix of test observations
#' @param W_0 Optional boolean feature screening vector
#' @param t Max size of tuning sequence. Default is 50
#' @param estimate_var Boolean for estimating variance using bootstrap
#' @param ... Extra arguments to be passed to boot when estimating variance (e.g. ncpus, parallel, R)
#'
#' @importFrom glue glue
#' @export
est_effect <- function(X,
                       W,
                       Y,
                       X_test,
                       W_0,
                       t= 50,
                       estimate_var = F,
                       ...) {
  effect_0 <-
      td_dnn(X[W == 0, ], matrix(Y[W == 0]), X_test, W_0, t)
  deDNN_pred_0 <- effect_0$deDNN_pred
  s_choice_0 <- effect_0$s_choice

  effect_1 <-
      td_dnn(X[W == 1, ], matrix(Y[W == 1]), X_test, W_0, t)
  deDNN_pred_1 <- effect_1$deDNN_pred
  s_choice_1 <- effect_1$s_choice

  deDNN_pred <- deDNN_pred_1 - deDNN_pred_0

  if( estimate_var){
      boot_est <- est_variance(X, W, Y, X_test, W_0, s_choice_0, s_choice_1, ...)

     list(estimate = deDNN_pred,
          variance =  mean((boot_est$t - deDNN_pred)^2))
  } else{
      list(estimate = deDNN_pred)
  }

}


est_variance <- function(X, W, Y, X_test, W_0, s_choice_0, s_choice_1, ...){

    boot_data <- data.frame(X, W, Y)
    d <- dim(X)[2]

    boot_function_cpp <- function(dat, idx, s_choice_0, s_choice_1, W_0) {
        # subsample the indices and of those split in to treated and control groups
        X <- as.matrix(dat[idx, 1:d])
        W <- dat$W[idx]
        Y <- dat$Y[idx]

        # split into groups
        trt_est <- td_dnn(X[W == 0, ],
                          Y[W == 0],
                          X_test = X_test,
                          s_choice = s_choice_1,
                          W_0 = W_0
        )
        ctrl_est <- td_dnn(X[W == 1, ],
                           Y[W == 1],
                           X_test = X_test,
                           s_choice = s_choice_0,
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
        ...
    )
    boot_estimates
}

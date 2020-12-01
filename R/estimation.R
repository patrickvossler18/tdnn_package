est_effect <- function(X,
                       Y,
                       X_test,
                       W_0 = NULL,
                       cpp = T,
                       t = 50) {
  s_choice0 <- tuning(seq(1, t, 1), X, Y, X_test, 2, W0_ = W_0)

  deDNN_pred <- td_dnn(X,
    Y,
    X_test = X_test,
    s_choice = s_choice0,
    W_0 = W_0
  )

  return(list(deDNN_pred = deDNN_pred, s_choice = s_choice0))
}


est_variance <- function(X,
                         W,
                         Y,
                         X_test,
                         W_0,
                         multicore = F,
                         ncpus = NULL,
                         cpp = T,
                         num_replicates = 1000) {
  effect_0 <-
    est_effect(X[W == 0, ], matrix(Y[W == 0]), X_test, W_0, cpp = cpp)
  deDNN_pred_0 <- effect_0$deDNN_pred
  s_choice_0 <- effect_0$s_choice

  effect_1 <-
    est_effect(X[W == 1, ], matrix(Y[W == 1]), X_test, W_0,
      cpp =
        cpp
    )
  deDNN_pred_1 <- effect_1$deDNN_pred
  s_choice_1 <- effect_1$s_choice

  deDNN_pred <- deDNN_pred_1 - deDNN_pred_0

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
  if (multicore) {
    use_parallel <- "multicore"
    ncpus <- ncpus
  } else {
    use_parallel <- NULL
    ncpus <- NULL
  }
  boot_estimates <- boot::boot(
    data = boot_data,
    statistic = boot_function_cpp,
    R = num_replicates,
    s_choice_0 = s_choice_0,
    s_choice_1 = s_choice_1,
    W_0 = W_0,
    parallel = use_parallel,
    ncpus = ncpus
  )

  return (mean((boot_estimates$t - deDNN_pred)^2))

}

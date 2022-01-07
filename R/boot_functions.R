boot_function_est_reg_fn <- function(dat, idx, X_test, s_choice, W_0, c, n_prop, d, verbose) {
    # subsample the indices and of those split in to treated and control groups
    X <- as.matrix(dat[idx, 1:d])
    Y <- dat$Y[idx]

    tdnn:::tdnn(
        X,
        Y,
        X_test = X_test,
        s_sizes = s_choice,
        s_sizes_1 = s_choice + 1,
        c = c,
        n_prop = n_prop,
        W0_ = W_0
    )
}


boot_function_trt_effect <- function(dat, idx, s_choice_0, s_choice_1, W_0, c, n_prop,verbose) {
    # subsample the indices and of those split in to treated and control groups
    X <- as.matrix(dat[idx, 1:d])
    W <- dat$W[idx]
    Y <- dat$Y[idx]

    # split into groups
    trt_est <- tdnn:::tdnn(X[W == 1, ],
                           matrix(Y[W == 1]),
                           X_test = X_test,
                           s_sizes = s_choice_1,
                           s_sizes_1 = s_choice_1 + 1,
                           c = c,
                           n_prop = n_prop,
                           W0_ = W_0)
    ctl_est <- tdnn:::tdnn(X[W == 0, ],
                           matrix(Y[W == 0]),
                           X_test = X_test,
                           s_sizes = s_choice_0,
                           s_sizes_1 = s_choice_0 + 1,
                           c = c,
                           n_prop = n_prop,
                           W0_ = W_0)
    trt_est - ctl_est
}

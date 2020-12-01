est_effect = function(X, Y, Xtest, W0 = NULL, cpp = T, t=50) {
    s.choice0 <- tuning(seq(1, t, 1), X, Y, Xtest, 2, W0_ = W0)

    deDNN_pred = td_dnn(X,
                        Y,
                        X_test = Xtest,
                        s.choice = s.choice0,
                        W0 = W0)

    return(list(deDNN.pred = deDNN_pred, s.choice = s.choice0))
}


est_variance <- function(X, W, Y, X_test, W0,
                         multicore=F, ncpus=NULL, cpp=T,
                         num_replicates= 1000){

    effect_0 = est_effect(X[W == 0,],matrix(Y[W == 0]), X_test, W0, cpp = cpp)
    deDNN.pred_0 = effect_0$deDNN.pred
    s.choice_0 = effect_0$s.choice

    effect_1 = est_effect(X[W == 1,],matrix(Y[W == 1]),X_test, W0, cpp=cpp)
    deDNN.pred_1 = effect_1$deDNN.pred
    s.choice_1 = effect_1$s.choice

    deDNN.pred = deDNN.pred_1 - deDNN.pred_0

    boot_data = data.frame(X,W,Y)

    boot_function_cpp = function(dat, idx, s.choice_0, s.choice_1, W0){
        # subsample the indices and of those split in to treated and control groups
        X = as.matrix(dat[idx,1:d])
        W = dat$W[idx]
        Y = dat$Y[idx]

        #split into groups
        trt_est <- td_dnn(X[W == 0, ],
                          Y[W == 0],
                          X_test = X_test,
                          s.choice = s.choice_1,
                          W0 = W0)
        ctrl_est <- td_dnn(X[W == 1, ],
                           Y[W == 1],
                           X_test = X_test,
                           s.choice = s.choice_0,
                           W0 = W0)
        trt_est - ctrl_est

    }
    if(multicore){
        use_parallel = "multicore"
        ncpus = ncpus
    }else{
        use_parallel = NULL
        ncpus = NULL
    }
    boot_estimates = boot::boot(data = boot_data, statistic = boot_function_cpp,
                                R = num_replicates, s.choice_0 = s.choice_0,
                                s.choice_1 = s.choice_1, W0 = W0,
                                parallel = use_parallel, ncpus=ncpus)
}

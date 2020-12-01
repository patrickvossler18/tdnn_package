library(Rcpp)
library(foreach)
library(doParallel)
library(dplyr)
library(tidyr)
library(boot)
library(grf)
library(energy)
library(tictoc)
library(purrr)
library(Rcpp)
library(glue)


Sys.setenv("USE_CXX11" = "yes")
sourceCpp("dnn.cpp")


td_dnn <- function(X, Y, X_test, s.choice, W0) {
    # Data checks before we doing anything else
    # Check X is a dataframe or matrix. If df, make it a matrix
    if(is.data.frame(X)){
        warning(glue("X is a dataframe. Attempting to convert to numeric matrix"))
        X <- as.matrix(X)
    }
    
    # Need all numeric columns for X matrix. 
    # This might not be a necessary check since R coerces to numeric for matrix?
    if(!all(sapply(X, is.numeric))){
        
        non_num_cols <- which(!(sapply(X, is.numeric)))
        stop(glue("Found non-numeric columns. Column indices: {non_num_cols}"))
    }
    
    # Check if Y is a matrix
    if(! is.matrix(Y)){
        stop(glue("Y is of class {class(Y)[1]}, but needs to be a matrix"))
        # Y = matrix(Y)
    }
    
    # Check that dimensions match up for X, Y, X_test, and W0
    # need length of Y to match dim[1] of X
    # need length of W0 to match dim[2] of X
    # X_test can be of different length but ncols of X and X_test must match
    dimension_checks <- c(dim(X)[1] == length(Y), 
                          dim(X)[2] == length(W0), 
                          dim(X)[2] == dim(X_test)[2])
    if(!all(dimension_checks)){
        # get which checks failed
        failed_checks <- which(!(dimension_checks))
        fail_messages <- sapply(failed_checks, function(x){switch(EXPR=x,
               glue("nrow of X ({dim(X)[1]}) does not equal length of Y: {length(Y)}"),
               glue("ncol of X ({dim(X)[2]}) does not equal length of W0: {length(W0)}"),
               glue("ncol of X ({dim(X)[2]}) does not equal ncol of X_test: {dim(X_test)[2]}")
               )})
        stop(glue("Dimensions don't match: \n {toString(fail_messages)}"))
    }
    
    X_fs = X[, which(W0 == 1)]
    X_test_fs = matrix(X_test[which(W0 == 1)], nrow = 1)
    
    a.pred = de_dnn(
        X_fs,
        Y,
        X_test = X_test_fs,
        s_size = s.choice,
        bc_p = 2
    )$estimates
    
    b.pred = de_dnn(
        X_fs,
        Y,
        X_test = X_test_fs,
        s_size = s.choice + 1,
        bc_p = 2
    )$estimates
    
    t.pred = (a.pred + b.pred) / 2
    return(t.pred)
}


est_effect = function(X, Y, Xtest, W0=NULL, cpp=T){
    Dataset = data.frame(X, Y)
    
    t = 50
    s.choice0 <- tuning(seq(1, t, 1), X, Y, Xtest, 2, W0_=W0)
    
    if(cpp){
        deDNN_pred = td_dnn(X, Y, X.test = Xtest, 
                            s.choice = s.choice0, W0 = W0)
    }
    else{
        deDNN_pred = td.dnn(Dataset, X.test = Xtest, s.choice = s.choice0, W0 = W0)
    }
    
    
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
                          X.test = X_test,
                          s.choice = s.choice_1,
                          W0 = W0)
        ctrl_est <- td_dnn(X[W == 1, ],
                           Y[W == 1],
                           X.test = X_test,
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



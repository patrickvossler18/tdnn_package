# source_file("old_implementation.R")

set.seed(1234)
library(tidyverse)
library(parallel)
library(kknn)

dgp_function = function(x){
    # This function will take a row of the matrix as input and return the 
    # transformed value. To use this on a matrix, we can use the apply function
    # with margin=1
    (x[1]-1)^2 + (x[2]+1)^3 - 3*x[3]
    # (x[1])^2 + (x[2])^3 - 3*x[3]
}


n=1000
p = 3
X = matrix(rnorm(n * p), n, p)
epsi = matrix(rnorm(n) , n, 1)
Y = apply(X, MARGIN = 1, dgp_function) + epsi

fixed_test_vector = c(2, -2, 2) # because using normally distributed data
X_test_fixed <- matrix(fixed_test_vector, 1, p)
mu <- dgp_function(fixed_test_vector)

# generate our B LOO samples
B = 20
X.dis = X - kronecker(matrix(1, n, 1), X_test_fixed)
EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
B.index = (sort(EuDis,index.return = T)$ix)[1:B]
s_seq = seq(10, 150, 1)
scale_p = 1

# Get estimates for all parameter values for all B LOO samples
boot_reps <- lapply(1:B, function(b) {
    # get the bth LOO sample indices
    # Make train data and validation data
    X_train <- X[-B.index[b],]
    Y_train <- as.matrix(Y[-B.index[b],])
    
    X_val <- matrix(X[B.index[b],], 1, ncol(X))
    Y_val <- as.matrix(Y[B.index[b],])
    
    n_train = nrow(X_train)
    p_train = ncol(X_train)
    
    X_train_dis = X_train - kronecker(matrix(1, n_train, 1), X_val)
    EuDis_train = (X_train_dis ^ 2) %*% matrix(1, p_train, 1)
    index.order = sort(EuDis_train, index = T)$ix
    ordered_Y_train <- Y_train[index.order]
    
    dnn.res = pmap_df(tidyr::expand_grid(c = 1, s =  s_seq), function(c, s){
        est.dnn = dnn_ord(ordered_Y_train, n_train, p_train, s)
        bind_rows(list(data.frame(
            MSE.dnn = (est.dnn - Y_val)^2*exp(- sum((X_val - X_test_fixed)^2) / scale_p),
            s = s,
            y_val = Y_val
        )
        ))
    })
    return(list(dnn.res = dnn.res))
})
boot_rep_results_dnn =  matrix(0, nrow = length(s_seq), ncol = 3)
for(i in 1:B) {
    u = boot_reps[[i]]
    boot_rep_results_dnn = boot_rep_results_dnn + u$dnn.res
}
boot_rep_results_dnn = boot_rep_results_dnn/B

dnn.min = boot_rep_results_dnn[which.min(boot_rep_results_dnn[, 1]), ]

test_that("C++ dnn tuning gives same estimate and s_1 value for a fixed test vector",{

    dnn_R_est <- dnn0(X, Y, X_test_fixed, dnn.min$s)
    dnn_R_s_1 <- dnn.min$s

    dnn_cpp <- tdnn:::tune_dnn_no_dist_thread(X,Y,X_test_fixed,
    s_seq = s_seq,W0_ = rep(1,p))
    dnn_cpp_est <- as.numeric(dnn_cpp$estimate_loo)
    dnn_cpp_s_1 <- as.numeric(dnn_cpp$s_1_B_NN)

    expect_equal(dnn_R_est, dnn_cpp_est)
    expect_equal(dnn_R_s_1, dnn_cpp_s_1)
})


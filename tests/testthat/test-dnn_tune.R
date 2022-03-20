dnn = function(ordered_Y, n, p, s.size = 2) {
    # Weight
    ord1 = matrix(1:(n - s.size +1) , n - s.size +1 , 1)
    # (n-k s-1) over (n s)
    weight1 = rbind(s.size * exp(lgamma(n - ord1 + 1) + lgamma(n - s.size + 1) - lgamma(n+1) - lgamma(n - ord1 - s.size + 2)), matrix(0, nrow = s.size -1, ncol = 1))    # choose(n - ord, s.size - 1) / choose(n, s.size)
    U1 = sum(ordered_Y * weight1)
    return(U1)
}


de.dnn <- function(
    ordered_Y,
    n,
    p,
    s.size = 2,
    bc.p = 2) {
    # Weight
    ord1 = matrix(1:(n - s.size +1) , n - s.size +1 , 1)
    ord2 = matrix(1:(n - ceiling(bc.p * s.size) +1) , n - ceiling(bc.p * s.size) +1 , 1)
    # (n-k s-1) over (n s)
    weight1 = rbind(s.size * exp(lgamma(n - ord1 + 1) + lgamma(n - s.size + 1) - lgamma(n+1) - lgamma(n - ord1 - s.size + 2)), matrix(0, nrow = s.size -1, ncol = 1))    # choose(n - ord, s.size - 1) / choose(n, s.size)
    weight2 = rbind(ceiling(bc.p * s.size) * exp(lgamma(n - ord2 + 1) + lgamma(n - ceiling(bc.p * s.size) + 1) - lgamma(n+1) - lgamma(n - ord2 - ceiling(bc.p * s.size) + 2)), matrix(0, nrow = ceiling(bc.p * s.size) -1, ncol = 1))  #choose(n - ord, bc.p * s.size - 1) / choose(n, bc.p * s.size)
    # Distance
    # X.dis = X - kronecker(matrix(1, n, 1), X.test)
    # EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
    # Ascending small->large
    # noise = matrix(rnorm(1), n, 1)
    # TempD = data.frame(EuDis, Y, noise)[order(EuDis, noise),]
    # Estimator
    U1 = sum(ordered_Y * weight1)
    U2 = sum(ordered_Y * weight2)
    Magic = solve(matrix(c(1, 1, 1, (1 / bc.p) ^ (2 / p)), 2, 2)) %*% matrix(c(1, 0), 2, 1)
    U = Magic[1, 1] * U1 + Magic[2, 1] * U2
    return(U)
}

de.dnn0 <- function(X,
                    Y,
                    X.test,
                    s.size = 2,
                    bc.p = 2) {
    n = nrow(X)
    p = ncol(X)
    # Weight
    ord1 = matrix(1:(n - s.size +1) , n - s.size +1 , 1)
    ord2 = matrix(1:(n - ceiling(bc.p * s.size) +1) , n - ceiling(bc.p * s.size) +1 , 1)
    # (n-k s-1) over (n s)
    weight1 = rbind(s.size * exp(lgamma(n - ord1 + 1) + lgamma(n - s.size + 1) - lgamma(n+1) - lgamma(n - ord1 - s.size + 2)), matrix(0, nrow = s.size -1, ncol = 1))    # choose(n - ord, s.size - 1) / choose(n, s.size)
    weight2 = rbind(ceiling(bc.p * s.size) * exp(lgamma(n - ord2 + 1) + lgamma(n - ceiling(bc.p * s.size) + 1) - lgamma(n+1) - lgamma(n - ord2 - ceiling(bc.p * s.size) + 2)), matrix(0, nrow = ceiling(bc.p * s.size) -1, ncol = 1))  #choose(n - ord, bc.p * s.size - 1) / choose(n, bc.p * s.size)
    # Distance
    X.dis = X - kronecker(matrix(1, n, 1), X.test)
    EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
    # Ascending small->large
    noise = matrix(rnorm(1), n, 1)
    TempD = data.frame(EuDis, Y, noise)[order(EuDis, noise),]
    # Estimator
    U1 = sum(TempD$Y * weight1)
    U2 = sum(TempD$Y * weight2)
    Magic = solve(matrix(c(1, 1, 1, (1 / bc.p) ^ (2 / p)), 2, 2)) %*% matrix(c(1, 0), 2, 1)
    U = Magic[1, 1] * U1 + Magic[2, 1] * U2
    return(U)
}

dnn0 <- function(X,
                 Y,
                 X.test,
                 s.size = 2) {
    n = nrow(X)
    p = ncol(X)
    # Weight
    ord1 = matrix(1:(n - s.size +1) , n - s.size +1 , 1)
    # (n-k s-1) over (n s)
    weight1 = rbind(s.size * exp(lgamma(n - ord1 + 1) + lgamma(n - s.size + 1) - lgamma(n+1) - lgamma(n - ord1 - s.size + 2)), matrix(0, nrow = s.size -1, ncol = 1))    # choose(n - ord, s.size - 1) / choose(n, s.size)
    # Distance
    X.dis = X - kronecker(matrix(1, n, 1), X.test)
    EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
    # Ascending small->large
    noise = matrix(rnorm(1), n, 1)
    TempD = data.frame(EuDis, Y, noise)[order(EuDis, noise),]
    # Estimator
    U1 = sum(TempD$Y * weight1)
    return(U1)
}



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
        est.dnn = dnn(ordered_Y_train, n_train, p_train, s)
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


dnn0(X, Y, X_test_fixed, dnn.min$s)
dnn.min$s

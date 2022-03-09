#' @export
de.dnn <- function(X,
                   Y,
                   X.test,
                   s.size = 2,
                   bc.p = 2) {
    n = nrow(X)
    p = ncol(X)
    # Weight
    ord = matrix(1:n , n , 1)
    # (n-k s-1) over (n s)
    weight1 = choose(n - ord, s.size - 1) / choose(n, s.size)
    weight2 = choose(n - ord, bc.p * s.size - 1) / choose(n, bc.p * s.size)
    # Distance
    X.dis = X - kronecker(matrix(1, n, 1), X.test)
    EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
    # Ascending small->large
    noise = matrix(rnorm(1), n, 1)
    TempD = data.frame(EuDis, Y, noise)[order(EuDis, noise),]
    # Estimator
    U1 = sum(TempD$Y * weight1)
    U2 = sum(TempD$Y * weight2)
    Magic = solve(matrix(c(1, 1, 1, (1 / bc.p) ^ (2 / min(
        p, 3
    ))), 2, 2)) %*% matrix(c(1, 0), 2, 1)
    U = Magic[1, 1] * U1 + Magic[2, 1] * U2
    return(U)
}

#' @export
tune_s <- function(X,Y,X_test, s_seq, c){
    t <- length(s_seq)
    tuning = matrix(0, length(s_seq), 1)
    tuning = sapply(s_seq, function(s) {
        de.dnn(X,Y, X_test, s.size = s + 1,bc.p = c)
    })

    s.choice = which(diff(abs(diff(tuning) / tuning[1:t - 1])) > -0.01)[1] + 3
    return(s.choice)
}

#' @export
tune_de_dnn <- function(X,Y,X_test, c = 2, B_NN=20, scale_p=1){
    n <- nrow(X)
    p <- ncol(X)
    fixed_c <- 2

    # get tuned s from MSE curvature method
    s_1_seq_curve <- seq(1,round(sqrt(n)),1)
    s_curve = tune_s(X,Y,X_test, s_1_seq_curve, fixed_c)
    estimate_curve = de.dnn(X, Y, X_test, s.size = s_curve, bc.p = 2)

    s_1_seq <- seq(s_curve,s_curve*2,1)
    param_df <- tidyr::expand_grid(c = fixed_c, s_1 =  s_1_seq)

    # generate our B nearest neighbor observations
    X.dis = X - kronecker(matrix(1, n, 1), X_test)
    EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
    B.index = (sort(EuDis,index.return = T)$ix)[1:B_NN]

    B_NN_estimates <- purrr::map_df(1:B_NN, function(b){
        X_train <- X[-B.index[b],]
        Y_train <- as.matrix(Y[-B.index[b],])

        X_val <- matrix(X[B.index[b],], 1, ncol(X))
        Y_val <- as.matrix(Y[B.index[b],])

        # for each LOO sample, calculate estimates for each s_1 value
        # this loops through the parameter combinations and returns a data frame with the results
        purrr::pmap_df(param_df, function(c, s_1) {
            param_estimate = de.dnn(X_train, Y_train, X_val, s_1, c)
            neighbor_weights = exp(- sum((X_val - X_test)^2) / scale_p)
            weighted_estimate = param_estimate*sqrt(neighbor_weights)
            weighted_y_val = Y_val*sqrt(neighbor_weights)
            data.frame(
                estimate = param_estimate,
                s_1 = s_1,
                c = c,
                y_val = Y_val,
                neighbor_weights = neighbor_weights,
                weighted_estimate = weighted_estimate,
                weighted_y_val= weighted_y_val,
                loss = (weighted_estimate - weighted_y_val)^2
            )
        })
    })
    tuned_mse <- B_NN_estimates %>% dplyr::group_by(s_1,c) %>%
        dplyr::summarize(tuned_mse = mean(loss)) %>% dplyr::pull(tuned_mse)
    choose_s1 = min(s_1_seq[tuned_mse <=  (1 + 0.01) * min(tuned_mse) ])
    tuned_estimate <- de.dnn(X, Y, X_test, choose_s1, fixed_c)
    list(estimate_loo = tuned_estimate,
         s_1_B_NN = choose_s1,
         estimate_curve = estimate_curve,
         s_1_curve = s_curve
         )
}

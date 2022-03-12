#' @export
#' @importFrom purrr pmap_df
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

de.dnn_no_dist <- function(
                   ordered_Y,
                   n,
                   p,
                   s.size = 2,
                   bc.p = 2) {
    # Weight
    ord = matrix(1:n , n , 1)
    # (n-k s-1) over (n s)
    weight1 = choose(n - ord, s.size - 1) / choose(n, s.size)
    weight2 = choose(n - ord, bc.p * s.size - 1) / choose(n, bc.p * s.size)
    # Distance
    # X.dis = X - kronecker(matrix(1, n, 1), X.test)
    # EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
    # Ascending small->large
    # noise = matrix(rnorm(1), n, 1)
    # TempD = data.frame(EuDis, Y, noise)[order(EuDis, noise),]
    # Estimator
    U1 = sum(ordered_Y * weight1)
    U2 = sum(ordered_Y * weight2)
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

tune_s_no_dist <- function(ordered_Y,n,p, s_seq, c){
    t <- length(s_seq)
    tuning = matrix(0, length(s_seq), 1)
    tuning = sapply(s_seq, function(s) {
        de.dnn_no_dist(ordered_Y,n,p, s.size = s + 1,bc.p = c)
    })

    s.choice = which(diff(abs(diff(tuning) / tuning[1:t - 1])) > -0.01)[1] + 3
    return(s.choice)
}

#' @export
tune_de_dnn <- function(X,Y,X_test, c = 2, B_NN=20, scale_p=1, debug=F){
    n <- nrow(X)
    p <- ncol(X)
    fixed_c <- 2

    # generate our B nearest neighbor observations
    X.dis = X - kronecker(matrix(1, n, 1), X_test)
    EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
    B.index = (sort(EuDis,index.return = T)$ix)[1:B_NN]



    # get tuned s from MSE curvature method
    s_1_seq_curve <- seq(1,ceiling(sqrt(n)),1)
    s_curve = tune_s(X,Y,X_test, s_1_seq_curve, fixed_c)
    estimate_curve = de.dnn(X, Y, X_test, s.size = s_curve, bc.p = 2)

    s_1_seq <- seq(s_curve,s_curve*2,1)
    param_df <- tidyr::expand_grid(c = fixed_c, s_1 =  s_1_seq)

    B_NN_estimates <- purrr::map_df(1:B_NN, function(b){
        X_train <- X[-B.index[b],]
        Y_train <- as.matrix(Y[-B.index[b],])

        X_val <- matrix(X[B.index[b],], 1, ncol(X))
        Y_val <- as.matrix(Y[B.index[b],])

        neighbor_weights = exp(- sum((X_val - X_test)^2) / scale_p)
        weighted_y_val = Y_val*sqrt(neighbor_weights)


        # for each LOO sample, calculate estimates for each s_1 value
        # this loops through the parameter combinations and returns a data frame with the results
        purrr::pmap_df(param_df, function(c, s_1) {
            param_estimate = de.dnn(X_train, Y_train, X_val, s_1, c)
            weighted_estimate = param_estimate*sqrt(neighbor_weights)
            if(debug){
                return(data.frame(
                    estimate = param_estimate,
                    s_1 = s_1,
                    c = c,
                    y_val = Y_val,
                    neighbor_weights = neighbor_weights,
                    weighted_estimate = weighted_estimate,
                    weighted_y_val= weighted_y_val,
                    loss = (weighted_estimate - weighted_y_val)^2
                ))
            } else{
                return(
                    data.frame(
                        s_1 = s_1,
                        c = c,
                        loss = (weighted_estimate - weighted_y_val)^2
                    )
                )
            }
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


#' @export
tune_de_dnn_no_dist <- function(X,Y,X_test, c = 2, B_NN=20, scale_p=1, debug=F){
    n <- nrow(X)
    p <- ncol(X)
    fixed_c <- 2

    # generate our B nearest neighbor observations
    X.dis = X - kronecker(matrix(1, n, 1), X_test)
    EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
    noise = matrix(rnorm(1), n, 1)
    ordered_Y <- Y[order(EuDis,noise)]
    B.index = (sort(EuDis,index.return = T)$ix)[1:B_NN]



    # get tuned s from MSE curvature method
    s_1_seq_curve <- seq(1,ceiling(sqrt(n)),1)
    s_curve = tune_s_no_dist(ordered_Y,n,p, s_1_seq_curve, fixed_c)
    estimate_curve = de.dnn_no_dist(ordered_Y,n,p,s_curve,bc.p=2)

    s_1_seq <- seq(s_curve,s_curve*2,1)
    param_df <- tidyr::expand_grid(c = fixed_c, s_1 =  s_1_seq)


    B_NN_estimates <- purrr::map_df(1:B_NN, function(b){
        X_train <- X[-B.index[b],]
        Y_train <- as.matrix(Y[-B.index[b],])

        X_val <- matrix(X[B.index[b],], 1, ncol(X))
        Y_val <- as.matrix(Y[B.index[b],])

        n_train = nrow(X_train)
        p_train = ncol(X_train)

        X_train_dis = X_train - kronecker(matrix(1, n_train, 1), X_val)
        EuDis_train = (X_train_dis ^ 2) %*% matrix(1, p_train, 1)
        noise_train = matrix(rnorm(1), n_train, 1)
        ordered_Y_train <- Y_train[order(EuDis_train,noise_train)]

        neighbor_weights = exp(- sum((X_val - X_test)^2) / scale_p)
        weighted_y_val = Y_val*sqrt(neighbor_weights)

        # this loops through the parameter combinations and returns a data frame with the results
        purrr::pmap_df(param_df, function(c, s_1) {
            param_estimate = de.dnn_no_dist(ordered_Y_train,n_train, p_train, s_1, c)
            weighted_estimate = param_estimate*sqrt(neighbor_weights)
            if(debug){
                return(data.frame(
                    estimate = param_estimate,
                    s_1 = s_1,
                    c = c,
                    y_val = Y_val,
                    neighbor_weights = neighbor_weights,
                    weighted_estimate = weighted_estimate,
                    weighted_y_val= weighted_y_val,
                    loss = (weighted_estimate - weighted_y_val)^2
                ))
            } else{
                return(
                    data.frame(
                        s_1 = s_1,
                        c = c,
                        loss = (weighted_estimate - weighted_y_val)^2
                    )
                )
            }
        })
    })
    tuned_mse <- B_NN_estimates %>% dplyr::group_by(s_1,c) %>%
        dplyr::summarize(tuned_mse = mean(loss)) %>% dplyr::pull(tuned_mse)
    choose_s1 = min(s_1_seq[tuned_mse <=  (1 + 0.01) * min(tuned_mse) ])
    tuned_estimate <- de.dnn_no_dist(ordered_Y,n,p, choose_s1, fixed_c)
    list(estimate_loo = tuned_estimate,
         s_1_B_NN = choose_s1,
         estimate_curve = estimate_curve,
         s_1_curve = s_curve
    )
}

# calculate distance matrix where each column corresponds to a test observation
#' @export
calc_dist_mat <- function(A,B){
    M = nrow(A)
    N = nrow(B)
    A_dots <- rowSums(A*A) %*% matrix(1,1,N)
    B_dots <- matrix(rowSums(B*B),M,N, byrow = T)
    A_dots + B_dots - 2*A %*% t(B)

}

#' function that allows for multiple test observations
#' @export
tune_de_dnn_no_dist_test_mat <- function(X,Y,X_test, c = 2, B_NN=20, scale_p=1, debug=F){
    n <- nrow(X)
    p <- ncol(X)
    fixed_c <- 2

    # generate Euclidean distance matrix once
    EuDis_mat <- calc_dist_mat(X,X_test)

    # loop through the test observations and calculate our estimate
    # this gives a list of lists where the ith element contains the prediction for the ith test observation
    X_test_predictions <- lapply(1:nrow(X_test), function(i){
        X_test_i <- matrix(X_test[i,],1,ncol(X_test))
        # get EuDis column vector
        EuDis <- EuDis_mat[,i]
        noise = matrix(rnorm(1), n, 1)
        ordered_Y <- Y[order(EuDis,noise)]
        B.index = (sort(EuDis,index.return = T)$ix)[1:B_NN]
        # get tuned s from MSE curvature method
        s_1_seq_curve <- seq(1,ceiling(sqrt(n)),1)
        s_curve = tune_s_no_dist(ordered_Y,n,p, s_1_seq_curve, fixed_c)
        estimate_curve = de.dnn_no_dist(ordered_Y,n,p,s_curve,bc.p=2)

        s_1_seq <- seq(s_curve,s_curve*2,1)
        param_df <- tidyr::expand_grid(c = fixed_c, s_1 =  s_1_seq)


        B_NN_estimates <- purrr::map_df(1:B_NN, function(b){
            X_train <- X[-B.index[b],]
            Y_train <- as.matrix(Y[-B.index[b],])

            X_val <- matrix(X[B.index[b],], 1, ncol(X))
            Y_val <- as.matrix(Y[B.index[b],])

            n_train = nrow(X_train)
            p_train = ncol(X_train)

            X_train_dis = X_train - kronecker(matrix(1, n_train, 1), X_val)
            EuDis_train = (X_train_dis ^ 2) %*% matrix(1, p_train, 1)
            noise_train = matrix(rnorm(1), n_train, 1)
            ordered_Y_train <- Y_train[order(EuDis_train,noise_train)]

            neighbor_weights = exp(- sum((X_val - X_test_i)^2) / scale_p)
            weighted_y_val = Y_val*sqrt(neighbor_weights)

            # this loops through the parameter combinations and returns a data frame with the results
            purrr::pmap_df(param_df, function(c, s_1) {
                param_estimate = de.dnn_no_dist(ordered_Y_train,n_train, p_train, s_1, c)
                weighted_estimate = param_estimate*sqrt(neighbor_weights)
                if(debug){
                    return(data.frame(
                        estimate = param_estimate,
                        s_1 = s_1,
                        c = c,
                        y_val = Y_val,
                        neighbor_weights = neighbor_weights,
                        weighted_estimate = weighted_estimate,
                        weighted_y_val= weighted_y_val,
                        loss = (weighted_estimate - weighted_y_val)^2
                    ))
                } else{
                    return(
                        data.frame(
                            s_1 = s_1,
                            c = c,
                            loss = (weighted_estimate - weighted_y_val)^2
                        )
                    )
                }
            })
        })
        tuned_mse <- B_NN_estimates %>% dplyr::group_by(s_1,c) %>%
            dplyr::summarize(tuned_mse = mean(loss)) %>% dplyr::pull(tuned_mse)
        choose_s1 = min(s_1_seq[tuned_mse <=  (1 + 0.01) * min(tuned_mse) ])
        tuned_estimate <- de.dnn_no_dist(ordered_Y,n,p, choose_s1, fixed_c)
        list(estimate_loo = tuned_estimate,
             s_1_B_NN = choose_s1,
             estimate_curve = estimate_curve,
             s_1_curve = s_curve
        )
    })

    return(X_test_predictions)

}


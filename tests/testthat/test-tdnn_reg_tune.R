
dgp_function = function(x){
    # This function will take a row of the matrix as input and return the
    # transformed value. To use this on a matrix, we can use the apply function
    # with margin=1
    (x[1]-1)^2 + (x[2]+1)^3 - 3*x[3]
}

n <- 100
s_1_seq <- seq(1,200,1)
p = 3


fixed_test_vector = c(0.5,-0.5,0.5)
X_test_fixed <- matrix(fixed_test_vector,1,p)
mu <- dgp_function(fixed_test_vector)
fixed_c <- 2
param_df <- tidyr::expand_grid(c = fixed_c, s_1 =  s_1_seq)

k_grid <- seq(1,200,1)

n_reps <- 500


set.seed(1234)


X = matrix(rnorm(n * p), n, p)
epsi = matrix(rnorm(n) , n, 1)
Y = apply(X, MARGIN = 1, dgp_function) + epsi

n_test <- 10
X_test_mat <- matrix(rep(X_test_fixed,n_test), n_test, p)

s_1_seq_curve <- seq(1,ceiling(sqrt(nrow(X))),1)

# generate our B nearest neighbor observations
X.dis = X - kronecker(matrix(1, n, 1), X_test_fixed)
EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
noise = matrix(rnorm(1), n, 1)
# TempD = data.frame(EuDis, Y, noise)[order(EuDis, noise),]
ordered_Y <- Y[order(EuDis,noise)]

B_NN = 20
B.index = (sort(EuDis,index.return = T)$ix)[1:B_NN]

# 2.09 ms vs 7.18 ms
# microbenchmark::microbenchmark(
#     tdnn:::tune_s_no_dist(ordered_Y, n, p, s_1_seq_curve, fixed_c),
#     tdnn:::tune_s(X,Y,X_test_fixed,s_1_seq_curve, fixed_c)
# )

s_curve_no_dist <- tdnn:::tune_s_no_dist(ordered_Y, n, p, s_1_seq_curve, fixed_c)
s_curve <- tdnn:::tune_s(X,Y,X_test_fixed,s_1_seq_curve, fixed_c)
s_curve_no_dist_cpp <- tdnn:::tuning_ord_Y(X,Y,X_test_fixed,as.matrix(ordered_Y),fixed_c,0.5)

test_that("We get the same s using the MSE curvature method using our C++ implementation that we get with our R implementation",
          {
              expect_equal(s_curve_no_dist, s_curve)
              expect_equal(as.numeric(s_curve_no_dist_cpp), s_curve)
})

test_that("We get the same estimate using the tuned s from the MSE",{
    curve_est_no_dist <- tdnn:::de.dnn_no_dist(ordered_Y,n,p,s_curve_no_dist, fixed_c)
    curve_est <- tdnn:::de.dnn(X,Y,X_test_fixed,s_curve, fixed_c)
    curve_est_no_dist_cpp <- tdnn:::tdnn_ord_y(X,Y,X_test_fixed,as.matrix(ordered_Y),s_curve_no_dist_cpp,c = fixed_c,n_prop = 0.5)

    expect_equal(as.numeric(curve_est_no_dist_cpp), curve_est)
    expect_equal(curve_est_no_dist, curve_est)
})


# 42 ms vs 218 ms
# microbenchmark::microbenchmark(
#     tdnn:::de.dnn_no_dist(ordered_Y,n,p,s_curve_no_dist, fixed_c),
#     tdnn:::de.dnn(X,Y,X_test_fixed,s_curve, fixed_c),
#     tdnn:::tdnn_ord_y(X,Y,X_test_fixed,as.matrix(ordered_Y),s_curve_no_dist_cpp,c = fixed_c,n_prop = 0.5)
# )

test_that("C++ code gives same estimate as R code for de-DNN using ordered Y",{
    expect_equal(
        as.numeric(tdnn:::tdnn_ord_y(X,Y,X_test_fixed, as.matrix(ordered_Y),c = fixed_c,n_prop = 0.5,
                                     s_1 = s_curve_no_dist )),
        tdnn:::de.dnn_no_dist(ordered_Y,n,p,s_curve_no_dist, fixed_c)
    )
})

test_that("Our C++ ordered Y matrix matches the matrix from R",{
    ordered_Y_mat_R <- sapply(1:n_test, function(i){
        X.dis = X - kronecker(matrix(1, n, 1), matrix(X_test_mat[i,],1,ncol(X)))
        EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
        noise = matrix(rnorm(1), n, 1)
        # TempD = data.frame(EuDis, Y, noise)[order(EuDis, noise),]
        Y[order(EuDis,noise)]

    })

    ordered_Y_mat_cpp <- tdnn:::make_ordered_Y_mat_debug(X,Y,X_test_mat)
    expect_equal(ordered_Y_mat_cpp, ordered_Y_mat_R)
})


debug=F
X_test_i <- matrix(X_test_mat[1,],1,ncol(X_test_mat))
scale_p=1
c = c(2,3)
s_1 = 2
B_NN = 1
s_1_tuning_seq = seq(2,4,1)
param_df <- tidyr::expand_grid(c = c, s_1 =  s_1_tuning_seq)
B_NN_estimates_R <- purrr::map_df(1:B_NN, function(b){
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
        param_estimate = tdnn:::de.dnn_no_dist(ordered_Y_train,n_train, p_train, s_1, c)
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

test_that("our C++ B-NN estimates match our R B-NN estimates for vector of c values",{
    B_NN_estimates_cpp <- tdnn:::make_B_NN_estimates(X,Y,t(X_test_i),
                                                     top_B = B.index-1,
                                                     s_tmp = 2,c_vec = c, B_NN = B_NN,debug = F)
    expect_equal(

        as.vector(t(B_NN_estimates_cpp)),
        B_NN_estimates_R$loss
    )
})


test_that("our C++ B-NN estimates match our R B-NN estimates for fixed c",{
    c = 2
    s_1 = 2
    B_NN = 1
    s_1_tuning_seq = seq(2,4,1)
    param_df <- tidyr::expand_grid(c = c, s_1 =  s_1_tuning_seq)
    B_NN_estimates_R <- purrr::map_df(1:B_NN, function(b){
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
            param_estimate = tdnn:::de.dnn_no_dist(ordered_Y_train,n_train, p_train, s_1, c)
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
    B_NN_estimates_cpp <- tdnn:::make_B_NN_estimates(X,Y,t(X_test_i),
                                                     top_B = B.index-1,
                                                     s_tmp = 2,c_vec = 2, B_NN = B_NN,debug = F)
    expect_equal(

        as.vector(t(B_NN_estimates_cpp)),
        B_NN_estimates_R$loss
    )
})

test_that("for a vector of c values, our choice of s_1 from B_NN tuning in C++ gives same answer as the R code",{
    c_vals = c(2,3)
    B_NN_estimates_cpp <- tdnn:::make_B_NN_estimates(X,Y,t(X_test_i),
                                                     top_B = B.index-1,
                                                     s_tmp = 2,c_vec = c_vals,
                                                     B_NN = B_NN)
    # R version
    best_params <- B_NN_estimates_R %>% dplyr::group_by(c, s_1) %>%
        dplyr::summarize(tuned_mse = mean(.data$loss)) %>%
        group_modify(~.x %>% filter(tuned_mse <= (1 + 0.01) * min(tuned_mse))) %>%
        filter(s_1 == min(s_1)) %>% ungroup() %>% filter(tuned_mse == min(tuned_mse))
    best_s1 = best_params %>% pull(s_1)
    best_c = best_params %>% pull(c)
    # C++ version
    best_params_cpp <- as.numeric(tdnn:::choose_s_1_c_val(2,c_vals,B_NN_estimates_cpp))
    best_c_cpp = best_params_cpp[1]
    best_s1_cpp = best_params_cpp[2]
    expect_equal(
        c(best_s1, best_c),
        c(best_s1_cpp, best_c_cpp)
    )

})



B_NN_estimates_cpp <- tdnn:::make_B_NN_estimates(X,Y,t(X_test_i),top_B = B.index-1,s_tmp = 2,c_vec = 2, B_NN = B_NN)
tuned_mse <- B_NN_estimates_R %>% dplyr::group_by(s_1,c) %>%
    dplyr::summarize(tuned_mse = mean(loss)) %>% dplyr::pull(tuned_mse)
choose_s1 = min(s_1_tuning_seq[tuned_mse <=  (1 + 0.01) * min(tuned_mse) ])
choose_s1_cpp = tdnn:::choose_s_1_val(matrix(B_NN_estimates_cpp,length(s_1_tuning_seq),1),s_1_tuning_seq)

test_that("for a fixed c, our choice of s_1 from the B_NN tuning in C++ gives same answer as R code",{
    expect_equal(
        as.numeric(choose_s1_cpp),
        choose_s1
    )
})

test_that("for a fixed c, tuned estimates from C++ match tuned estimates from full R function",{
    expect_equal(
        as.numeric(tdnn:::tune_de_dnn_no_dist_cpp(X,Y,X_test_fixed,W0_ = rep(1,p),c = 2, B_NN = 20,scale_p = 1,n_prop=0.5)$estimate_loo),
        tdnn:::tune_de_dnn_no_dist(X,Y,X_test_fixed,c = 2, B_NN = 20,scale_p = 1, debug=F)$estimate_loo,
    )
})


n_test <- 2
X_test_fixed_2 <- matrix(c(0.1,0.2,0.3),1,p)
X_test_mat <- matrix(c(X_test_fixed,c(0.1,0.2,0.3)), n_test, p, byrow = T)
ordered_Y_mat <- tdnn:::make_ordered_Y_mat_debug(X,Y,X_test_mat)



test_that("The full tuned C++ estimate matches the tuned R estimate",{
    tune_de_dnn_R_mat <- tdnn:::tune_de_dnn_no_dist_test_mat(X,Y,X_test_mat,c = 2,B_NN = 20, scale_p = 1)

    tune_de_dnn_cpp_mat <- tdnn:::tune_de_dnn_no_dist_cpp(X,Y,X_test_mat,W0_ = rep(1,p),c = 2,
                                                          B_NN = 20,scale_p = 1,n_prop=0.5, debug = F)
    tune_de_dnn_R_mat_loo <- bind_rows(tune_de_dnn_R_mat)

    expect_equal(as.numeric(tune_de_dnn_cpp_mat$estimate_loo), tune_de_dnn_R_mat_loo %>% pull(estimate_loo))
    expect_equal( as.numeric(tune_de_dnn_cpp_mat$s_1_B_NN), tune_de_dnn_R_mat_loo %>% pull(s_1_B_NN))

})


test_that("Bootstrap estimate method matches the R estimate and the C++ estimate for fixed s and c", {
    de_dnn_est_R <- tdnn:::de.dnn_no_dist(ordered_Y,n,p,s_curve_no_dist, fixed_c)
    de_dnn_ord_est_cpp <- as.numeric(tdnn:::tdnn_ord_y(X,Y,X_test_fixed, as.matrix(ordered_Y),c = fixed_c,n_prop = 0.5,
                                                       s_1 = s_curve_no_dist ))
    ord = seq(1,n,1)
    weight_mat_s_1 <- tdnn:::weight_mat_lfac_s_2_filter(n, ord,s_vec = s_curve_no_dist,n_prop = 0.5,is_s_2 = F)
    weight_mat_s_2 <- tdnn:::weight_mat_lfac_s_2_filter(n, ord,s_vec= ceiling(s_curve_no_dist * fixed_c),n_prop = 0.5,is_s_2 = T)
    de_dnn_boot_est_cpp <- as.numeric(tdnn:::tdnn_st_boot(X,Y,X_test_fixed,weight_mat_s_1, weight_mat_s_2,c = fixed_c,n_prop = 0.5))

    expect_equal(de_dnn_boot_est_cpp,de_dnn_est_R)
    expect_equal(de_dnn_boot_est_cpp,de_dnn_ord_est_cpp)

})

test_that("Bootstrap estimate matches tuned C++ estimate when using the same s vector for matrix test input", {
    tune_de_dnn_cpp_mat <- tdnn:::tune_de_dnn_no_dist_cpp(X,Y,X_test_mat,W0_ = rep(1,p),c = 2,
                                                          B_NN = 20,scale_p = 1,n_prop=0.5, debug = F)

    s_vec <- as.numeric(tune_de_dnn_cpp_mat$s_1_B_NN)
    ord = seq(1,n,1)
    weight_mat_s_1 <- tdnn:::weight_mat_lfac_s_2_filter(n, ord,s_vec = s_vec,n_prop = 0.5,is_s_2 = F)
    weight_mat_s_2 <- tdnn:::weight_mat_lfac_s_2_filter(n, ord,s_vec= ceiling(s_vec * fixed_c),n_prop = 0.5,is_s_2 = T)
    tune_de_dnn_cpp_mat <- tdnn:::tune_de_dnn_no_dist_cpp(X,Y,X_test_mat,W0_ = rep(1,p),c = 2,
                                                          B_NN = 20,scale_p = 1,n_prop=0.5, debug = F)

    expect_equal(as.numeric(tdnn:::tdnn_st_boot(X,Y,X_test_mat,weight_mat_s_1,
                                                weight_mat_s_2,
                                                c = rep(fixed_c,nrow(X_test_mat)),n_prop = 0.5)),
                 as.numeric(tune_de_dnn_cpp_mat$estimate_loo))

})


test_that("tdnn tune reg R function gives same estimate as using the C++ function directly", {
    X_test_rand <- matrix(rnorm(100*p),100,p)
    expect_equal(
        tdnn::tdnn_reg_tune(X,Y,X_test_rand,W_0 = rep(1,p),c_val = fixed_c,B_NN = 20,n_prop = 0.5,estimate_variance = F, bootstrap_iter = 100),
        tdnn:::tune_de_dnn_no_dist_cpp(X,Y,X_test_rand,W0_ = rep(1,p),c = fixed_c,
                                       B_NN = 20,scale_p = 1,n_prop=0.5, debug = F)
    )
})

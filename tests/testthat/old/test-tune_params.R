seed_val <- 1234
set.seed(seed_val)

p = 3
n = 1000
n_test = 1000
B = 200
M = 2
c_val = 0.1


grid_start_end = c(-1, 1)
rand_test_points = matrix(c(
    seq(grid_start_end[1], grid_start_end[2], length.out = n_test),
    seq(grid_start_end[1], grid_start_end[2], length.out = n_test),
    seq(grid_start_end[1], grid_start_end[2], length.out = n_test)
), n_test, p)

dgp_function = function(x) {
    (x[1] - 1) ^ 2 + (x[2] + 1) ^ 3 - 3 * x[3]
}

loo_sample <- function(num_rows, B = 200) {
    a <- map(seq(B), function(x) {
        idx <- seq(num_rows)
        held_out_idx <- sample(idx, 1)
        list(train_idx = idx[-held_out_idx], held_out_idx = held_out_idx)
    })
}

# Generate data
X = matrix(runif(n * p, grid_start_end[1], grid_start_end[2]), n, p)
X_test_random <- rand_test_points
epsi = matrix(rnorm(n) , n, 1)
Y = apply(X, MARGIN = 1, dgp_function) + epsi
W_0 = rep(1, p)

# get a single LOO sample
train_idx <- loo_sample(n, 1)$train_idx
X_train <- X[train_idx,]
X_val <- matrix(X[-train_idx,], 1, ncol(X))
Y_train <- as.matrix(Y[train_idx,])
Y_val <- as.matrix(Y[-train_idx,])


test_that("test that make_param_estimates give same estimate as normal tdnn function",
          {
              ordered_Y <-
                  tdnn:::make_ordered_Y_vec(X_train, as.numeric(X_val), Y_train, nrow(X_train))
              ord_arma <-
                  tdnn:::make_ord_vec(nrow(X_train))
              param_est <-
                  tdnn:::make_param_estimate(
                      X_train,
                      Y_train,
                      X_val,
                      ordered_Y,
                      ord_arma,
                      nrow(X_train),
                      ncol(X_train),
                      log(nrow(X_train)),
                      c = 0.2,
                      C_s_2 = 2,
                      n_prop = 0.5,
                      W0 = W_0
                  )

              normal_est <- tdnn:::est_reg_fn_mt_rcpp(
                  X_train,
                  Y_train,
                  X_val,
                  c = 0.2,
                  verbose = FALSE,
                  n_prop = 0.5,
                  C_s_2 = 2,
                  W0_ = W_0
              )
              expect_equal(as.numeric(normal_est$estimates), param_est)

          })

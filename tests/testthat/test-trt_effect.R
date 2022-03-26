set.seed(1234)
d <- 20

impt_idx <- c(3, 5, 7)
fixed_test_vector <- rep(0.5, d)
fixed_test_vector[impt_idx] <- c(0.2, 0.4, 0.6)

dgp_function <- function(x, idx) {
    prod(sapply(idx, function(i) {
        (1 + 1 / (1 + exp(-20 * (x[i] - 1 / 3))))
    }))
}

baseline_function <- function(x) {
    x[3]^2 + x[7]
}
c <- 0.8
C_s_2 <- 2.0
n0 <- 1000
p0 <- d
n <- n0
p <- p0
X <- matrix(runif(n0 * p0), n0, p0)
epsi <- matrix(rnorm(n0), n0, 1)
W <- rbinom(n0, 1, 0.5) # treatment condition



Xtest_fixed <- matrix(fixed_test_vector, 1, p0)


Ytest <- apply(Xtest_fixed, MARGIN = 1, dgp_function, idx = impt_idx)
Y <- (W - 0.5) * apply(X, MARGIN = 1, dgp_function, idx = impt_idx) + epsi

Xtest <- Xtest_fixed

X_trt <- X[W==1,]
Y_trt <- Y[W==1]
X_ctl <- X[W==0,]
Y_ctl <- Y[W==0]

W0 <- rep(0, p0)
W0[impt_idx] <- 1

tune_trt_effect_results <- tdnn:::tune_treatment_effect_thread(X,Y,W,Xtest,W0,c = 2,verbose =F, estimate_variance = F)



mu_trt <- tdnn:::tune_de_dnn_no_dist_vary_c_cpp_thread(X_trt,Y_trt,Xtest,W0,c = 2)
mu_ctl <- tdnn:::tune_de_dnn_no_dist_vary_c_cpp_thread(X_ctl,Y_ctl,Xtest,W0,c = 2)
trt_effect <- mu_trt$estimate_loo - mu_ctl$estimate_loo

test_that("treatment effect estimate from tuning trt effect function matches estimate from separately tuning treatment and control mus. Also test that individual estimates are equal",
          {
              expect_equal(tune_trt_effect_results$treatment_effect, trt_effect)
              expect_equal(tune_trt_effect_results$estimate_trt, mu_trt$estimate_loo)
              expect_equal(tune_trt_effect_results$estimate_ctl, mu_ctl$estimate_loo)
})

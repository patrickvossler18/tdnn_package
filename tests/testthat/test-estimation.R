# source_file("old_implementation.R")

# Set up simulation setting
set.seed(1234)
d <- 20

impt_idx <- c(3, 5, 7)
fixed_test_vector <- rep(0.5, d)
fixed_test_vector[impt_idx] <- c(0.2, 0.4, 0.6)

dgp_function <- function(x, idx) {
  # This function will take a row of the matrix as input and return the
  # transformed value. To use this on a matrix, we can use the apply function
  # with margin=1
  prod(sapply(idx, function(i) {
    (1 + 1 / (1 + exp(-20 * (x[i] - 1 / 3))))
  }))
  # (1 + 1 / (1 + exp(-20 * (x[1] - 1/3)))) * (1 + 1 / (1 + exp(-20 * (x[2] - 1/3))))
}

baseline_function <- function(x) {
  x[3]^2 + x[7]
}
c <- 0.8
n0 <- 1000
p0 <- d
n <- n0
p <- p0
X <- matrix(runif(n0 * p0), n0, p0)
epsi <- matrix(rnorm(n0), n0, 1)
W <- rbinom(n0, 1, 0.5) # treatment condition



Xtest_fixed <- matrix(fixed_test_vector, 1, p0)

random_test_vector <- rep(0.5, p0)
random_test_vector[impt_idx] <- runif(length(impt_idx))
# random_test_vector = rnorm(p0)


Xtest_random <- matrix(runif(1000 * p0), 1000, p0)
Ytest <- apply(Xtest_random, MARGIN = 1, dgp_function, idx = impt_idx)

Ytest <- apply(Xtest_fixed, MARGIN = 1, dgp_function, idx = impt_idx)
Y <- (W - 0.5) * apply(X, MARGIN = 1, dgp_function, idx = impt_idx) + epsi

Xtest <- Xtest_fixed

W0 <- rep(0, p0)
W0[impt_idx] <- 1
# W0 = rep(1,p0)

# euclidean distance matrix
eu_dist_mat <- tdnn:::make_pdist_mat(X,Xtest,W0)
eu_dist_mat_rnd <- tdnn:::make_pdist_mat(X,Xtest_random,W0)
# d is # of features after screening
d <- sum(W0)


context("make sure that estimation function gives same output as original implementation")

test_that("est_effect gives same estimate as est_effect_old", {
  cpp_version <- as.numeric(tdnn::est_effect(X, W, Y, Xtest, W0, feature_screening = F, c = c, estimate_var = F)$estimate)
  r_version <- est_effect_old(X, W, Y, W0, Xtest, c = c)$deDNN_pred
  expect_equal(cpp_version, r_version)
})


context("tuning functions should all give the same choice of s")
test_that("early_stopping, single-threaded, and original implementation all give the same s value", {
  normal <- as.numeric(tdnn:::tuning_st(seq(1, 50, 1), eu_dist_mat, Y, c = c, n = nrow(X),
                             n_test_obs = nrow(Xtest), W0_ = W0,d = d))
  early_stopping <- tdnn:::tuning(X, Y, Xtest, W0_ = W0, c = c)

  t <- 50
  tuning_mat <- matrix(0, t, 1)
  Dataset <- data.frame(X, Y)
  for (s in seq(1, t, 1)) {
    tuning_mat[s] <- de.dnn(Dataset, X.test = Xtest, s.size = s + 1, W0 = W0, c = c)
  }
  base_r <- which(diff(abs(diff(tuning_mat) / tuning_mat[1:t - 1])) > -0.01)[1] + 3
  expect_true(all.equal(normal, early_stopping, base_r))
})





##### est_effect #####
test_that("we show a warning if X is a df instead of a matrix for est_effect", {
  X_df <- data.frame(X)
  expect_warning(
    tdnn::est_effect(
      X_df,
      W,
      Y,
      Xtest,
      W0,
      feature_screening = F,
      c = c,
      estimate_var = F
    )$estimate
  )
})

test_that("throw error if given non-numeric columns in X matrix", {
  X_chr <- mapply(X, FUN = as.character)
  expect_error(
    tdnn::est_effect(
      X_chr,
      W,
      Y,
      Xtest,
      W0,
      feature_screening = F,
      c = c,
      estimate_var = F
    )$estimate
  )
})

test_that("throw error if Y is not a matrix", {
  expect_error(
    tdnn::est_effect(
      X,
      W,
      as.numeric(Y),
      Xtest,
      W0,
      feature_screening = F,
      c = c,
      estimate_var = F
    )$estimate
  )
})

test_that("throw error if X test is not a matrix", {
  expect_error(
    tdnn::est_effect(
      X_df,
      W,
      Y,
      as.numeric(Xtest),
      W0,
      feature_screening = F,
      c = c,
      estimate_var = F
    )$estimate
  )
})

test_that("throw error if W_0 is not an integer vector of ones and zeros.", {
  W_0_wrong <- W0
  W_0_wrong[2] <- 3L
  expect_error(
    tdnn::est_effect(
      X,
      W,
      Y,
      Xtest,
      W_0_wrong,
      feature_screening = F,
      c = c,
      estimate_var = F
    )$estimate
  )
})


test_that("throw error if the number of columns of X doesn't match X_test", {
  Xtest_wrong <- matrix(rnorm((ncol(X) - 1) * 10), (ncol(X) - 1), 10)
  expect_error(
    tdnn::est_effect(
      X,
      W,
      Y,
      Xtest_wrong,
      W0,
      feature_screening = F,
      c = c,
      estimate_var = F
    )$estimate
  )
})

test_that("throw error if the number of rows of X don't match the length of Y", {
  expect_error(
    tdnn::est_effect(
      X,
      W,
      Y[0:(nrow(X) - 1)],
      Xtest,
      W0,
      feature_screening = F,
      c = c,
      estimate_var = F
    )$estimate
  )
})


test_that("throw error if the length of W doesn't match ncol(X)", {
  expect_error(
    tdnn::est_effect(
      X,
      W,
      Y,
      Xtest,
      W0[0:(ncol(X) - 1)],
      feature_screening = F,
      c = c,
      estimate_var = F
    )$estimate
  )
})

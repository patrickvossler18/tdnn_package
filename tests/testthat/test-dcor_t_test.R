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
}

baseline_function <- function(x) {
  x[3]^2 + x[7]
}

n0 <- 1000
p0 <- d
n <- n0
p <- p0
X <- matrix(runif(n0 * p0), n0, p0)
epsi <- matrix(rnorm(n0), n0, 1)
W <- rbinom(n0, 1, 0.5) # treatment condition

c <- 0.33

Xtest_fixed <- matrix(fixed_test_vector, 1, p0)

random_test_vector <- rep(0.5, p0)
random_test_vector[impt_idx] <- runif(length(impt_idx))


Xtest_random <- matrix(runif(10 * p0), 10, p0)
Ytest <- apply(Xtest_random, MARGIN = 1, dgp_function, idx = impt_idx)

Ytest <- apply(Xtest_fixed, MARGIN = 1, dgp_function, idx = impt_idx)
Y <- (W - 0.5) * apply(X, MARGIN = 1, dgp_function, idx = impt_idx) + epsi

Xtest <- Xtest_fixed

W0 <- rep(0, p0)
W0[impt_idx] <- 1


context("our dcor t-test implementation should give the same results as the energy package")

test_that("for each column of X, our dcor t-test should match the energy version", {
  expect_equal(
    tdnn:::feature_screening_parallel(X, Y),
    sapply(1:p0, function(i) {
      as.numeric(energy::dcorT.test(X[, i], Y)$p.value < 0.001 / p0)
    })
  )
})


test_that("manual implementation of pt matches the R implementation", {
  expect_equal(tdnn:::pt_raw(2, 10), pt(2, 10))
})

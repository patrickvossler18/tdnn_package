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


Xtest_random <- matrix(runif(1000 * p0), 1000, p0)
Ytest <- apply(Xtest_random, MARGIN = 1, dgp_function, idx = impt_idx)

Ytest <- apply(Xtest_fixed, MARGIN = 1, dgp_function, idx = impt_idx)
Y <- (W - 0.5) * apply(X, MARGIN = 1, dgp_function, idx = impt_idx) + epsi

Xtest <- Xtest_fixed

W0 <- rep(0, p0)
W0[impt_idx] <- 1

context("make sure that tdnn_reg function gives same output as original R implementation")

a <- est_reg_fn_old(X,Y,W0 = W0, Xtest = Xtest, c=c)
b <- tdnn::tdnn_reg(X,Y,Xtest,W0,c=c, n_prop=0.5)

test_that("tdnn_reg estimate is the same estimate as the R implementation version",
          {
              expect_equal(a$deDNN.pred,as.numeric(b$deDNN_pred))
          }
)

test_that("tdnn_reg estimate is the same s value as the R implementation version",
          {
              expect_equal(a$s.choice,b$s_choice)
          }
)


context(
    "throw informative errors if we get data in the wrong format instead of ambiguous Rcpp errors"
)

test_that("we show a warning if X is a df instead of a matrix for tdnn_reg", {
    X_df <- data.frame(X)
    expect_warning(
        tdnn::tdnn_reg(X_df,
                         Y,
                         Xtest,
                         W0)
    )
})

test_that("throw error if given non-numeric columns in X matrix for tdnn_reg", {
    X_chr <- mapply(X, FUN = as.character)
    expect_error(
        tdnn::tdnn_reg(X_chr,
                         Y,
                         Xtest,
                         W0)
    )
})


test_that("throw error if Y is not a matrix for tdnn_reg", {
    expect_error(
        tdnn::tdnn_reg(X,
                         as.numeric(Y),
                         Xtest,
                         W0)
    )
})

test_that("throw error if X test is not a matrix for tdnn_reg", {
    expect_error(
        tdnn::tdnn_reg(X,
                         Y,
                         as.numeric(Xtest),
                         W0)
    )
})

test_that("throw error if W_0 is not an integer vector of ones and zeros for tdnn_reg.", {
    W_0_wrong <- W0
    W_0_wrong[2] <- 3L
    expect_error(
        tdnn::tdnn_reg(X,
                         Y,
                         Xtest,
                         W_0_wrong)
    )
})


test_that("throw error if the number of columns of X doesn't match X_test for tdnn_reg.", {
    Xtest_wrong <- matrix(rnorm((ncol(X) - 1) * 10), (ncol(X) - 1), 10)
    expect_error(
        tdnn::tdnn_reg(X,
                         Y,
                         Xtest_wrong,
                         W0)
    )
})

test_that("throw error if the number of rows of X don't match the length of Y for tdnn_reg.", {
    expect_error(
        tdnn::tdnn_reg(X,
                         Y[0:(nrow(X) - 1)],
                         Xtest,
                         W0)
    )
})


test_that("throw error if the length of W doesn't match ncol(X)", {
    expect_error(
        tdnn::tdnn_reg(X,
                         Y,
                         Xtest,
                         W0[0:(ncol(X) - 1)])
    )
})

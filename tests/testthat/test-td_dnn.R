# source_file("old_implementation.R")

# Set up simulation setting
set.seed(1234)
d = 20

impt_idx = c(3, 5, 7)
fixed_test_vector = rep(0.5,d)
fixed_test_vector[impt_idx] = c(0.2, 0.4, 0.6)

dgp_function = function(x,idx){
    # This function will take a row of the matrix as input and return the
    # transformed value. To use this on a matrix, we can use the apply function
    # with margin=1
    prod(sapply(idx, function(i){
        (1 + 1 / (1 + exp(-20 * (x[i] - 1/3))))
    }))
    # (1 + 1 / (1 + exp(-20 * (x[1] - 1/3)))) * (1 + 1 / (1 + exp(-20 * (x[2] - 1/3))))
}

baseline_function = function(x){
    x[3]^2 + x[7]
}

n0 = 1000
p0 = d
n = n0
p = p0
X = matrix(runif(n0 * p0), n0, p0)
epsi = matrix(rnorm(n0) , n0, 1)
W <- rbinom(n0, 1, 0.5) #treatment condition

c = 0.33

Xtest_fixed = matrix(fixed_test_vector, 1, p0)

random_test_vector = rep(0.5,p0)
random_test_vector[impt_idx] = runif(length(impt_idx))
# random_test_vector = rnorm(p0)


Xtest_random = matrix(runif(1000 * p0), 1000, p0)
Ytest =   apply(Xtest_random, MARGIN = 1, dgp_function, idx = impt_idx)

Ytest =   apply(Xtest_fixed, MARGIN = 1, dgp_function, idx = impt_idx)
Y =  (W - 0.5 ) * apply(X, MARGIN = 1, dgp_function, idx = impt_idx) + epsi

Xtest = Xtest_fixed

W0 = rep(0, p0)
W0[impt_idx] = 1


context("fixed s, fixed test pt")
test_that("single-threaded version of de_dnn gives same estimate as R version for fixed s and single fixed test point",
          {
              expect_equal(as.numeric(tdnn:::de_dnn_st(X,Y,Xtest,s_sizes = 2,W0_=W0, c = c)),
              de.dnn(data.frame(X, Y),X.test = Xtest,s.size = 2,W0 = W0, c=c))
          })

test_that("multi-threaded version of de_dnn gives same estimate as R version for fixed s and single fixed test point",
          {
              expect_equal(tdnn:::de_dnn(X,Y,Xtest,s_sizes = 2, W0_=W0, c = c)$estimates,
                           de.dnn(data.frame(X, Y),X.test = Xtest,s.size = 2,W0 = W0, c = c))
          })


context("tuning using greedy tuning")
test_that("greedy version of tuning algo matches original R implementation's choice of s",
          {
              t = 50
              tuning_mat = matrix(0, t, 1)
              Dataset = data.frame(X, Y)
              for (s in seq(1, t, 1)) {
                  tuning_mat[s] = de.dnn(Dataset, X.test = Xtest, s.size = s + 1, W0 = W0, c = c)
              }
              orig_s <- which(diff(abs(diff(tuning_mat) / tuning_mat[1:t - 1])) > -0.01)[1] + 3
              greedy_s <- tuning( X, Y, Xtest, W0_=W0, c = c)
              expect_equal(greedy_s, orig_s)
          })

context("multi-threaded de_dnn same as single-threaded de_dnn")
test_that(
    "multi-threaded de_dnn gives same estimates as single-threaded de_dnn for vector of test inputs",
    {
        a <- as.numeric(tdnn:::de_dnn_st(
            X,
            Y,
            X_test = Xtest_random,
            s_sizes = rep(2, nrow(Xtest_random)),
            W0_ = W0, c = c
        ))
        b <- de_dnn(
            X,
            Y,
            X_test = Xtest_random,
            s_sizes = rep(2, nrow(Xtest_random)),
            W0_ = W0, c = c
        )$estimates
        expect_equal(b, a)
    }
)



dnn_ord = function(ordered_Y, n, p, s.size = 2) {
    # Weight
    ord1 = matrix(1:(n - s.size +1) , n - s.size +1 , 1)
    # (n-k s-1) over (n s)
    weight1 = rbind(s.size * exp(lgamma(n - ord1 + 1) + lgamma(n - s.size + 1) - lgamma(n+1) - lgamma(n - ord1 - s.size + 2)), matrix(0, nrow = s.size -1, ncol = 1))    # choose(n - ord, s.size - 1) / choose(n, s.size)
    U1 = sum(ordered_Y * weight1)
    return(U1)
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


de.dnn <-
  function(Dataset,
           s.size = 2,
           bc.p = 2,
           c,
           X.test,
           W0) {
    n <- nrow(Dataset)
    p <- ncol(Dataset) - 1

    X <- as.matrix(Dataset[, c(1:p)])
    Y <- Dataset$Y
    # Weight
    ord <- matrix(1:n, n, 1)

    d <- sum(W0)
    w_1 <- c / (c - 1)
    w_2 <- -1 / (c - 1)
    C_s_2 <- 2.0
    s_1 <- s.size
    # s_2 <- round(C_s_2 * n^(d/(d+8)))
    # s_2 <- round(s_1 * (c^(-d / 2)))
    s_2 <- ceiling(s_1 * c)


    # Distance
    X.dis <- (X - kronecker(matrix(1, n, 1), X.test)) * (matrix(1, n, 1) %*% W0)
    EuDis <- (X.dis^2) %*% matrix(1, p, 1)
    # Ascending small->large
    noise <- matrix(rnorm(1), n, 1)
    TempD <- data.frame(EuDis, Y, noise)[order(EuDis, noise), ]


    # new method using c
    weight1 <- choose(n - ord, s_1 - 1) / choose(n, s_1)
    weight2 <- choose(n - ord, s_2 - 1) / choose(n, s_2)


    # Estimator
    U1 <- sum(TempD$Y * weight1)
    U2 <- sum(TempD$Y * weight2)


    U <- w_1 * U1 + w_2 * U2

    return(U)
  }

td.dnn <- function(Dataset, X.test, s.choice, c, W0) {
  a.pred <- de.dnn(Dataset,
    X.test = X.test,
    s.size = s.choice,
    c = c,
    W0 = W0
  )
  b.pred <- de.dnn(Dataset,
    X.test = X.test,
    s.size = s.choice + 1,
    c = c,
    W0 = W0
  )
  t.pred <- (a.pred + b.pred) / 2
  return(t.pred)
}

est_reg_fn_old <- function(X, Y, W0, Xtest, c) {
  Dataset <- data.frame(X, Y)

  t <- 50
  tuning <- matrix(0, t, 1)

  for (s in seq(1, t, 1)) {
    tuning[s] <- de.dnn(Dataset,
      X.test = Xtest,
      s.size = s + 1,
      c = c,
      W0 = W0
    )
  }
  s.choice0 <- which(diff(abs(diff(tuning) / tuning[1:t - 1])) > -0.01)[1] + 3


  deDNN.pred <- td.dnn(Dataset,
    X.test = Xtest,
    s.choice = s.choice0,
    c = c,
    W0 = W0
  )

  return(list(deDNN.pred = deDNN.pred, s.choice = s.choice0))
}


est_effect_old <- function(X, W, Y, W0, X_test, c) {
  results_0 <- est_reg_fn_old(X[W == 0, ], Y[W == 0], W0, X_test, c)
  deDNN.pred_0 <- results_0$deDNN.pred
  s.choice_0 <- results_0$s.choice

  results_1 <- est_reg_fn_old(X[W == 1, ], Y[W == 1], W0, X_test, c)
  deDNN.pred_1 <- results_1$deDNN.pred
  s.choice_1 <- results_1$s.choice

  deDNN.pred <- deDNN.pred_1 - deDNN.pred_0

  return(list(
    deDNN_pred = deDNN.pred,
    s_choice0 = s.choice_0,
    s_choice1 = s.choice_1
  ))
}

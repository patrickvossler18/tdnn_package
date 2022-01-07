td_dnn <- function(X, Y, X_test, s_choice, W_0, c, n_prop) {
  a_pred <- de_dnn(
    X,
    Y,
    X_test = X_test,
    s_sizes = s_choice,
    c = c,
    n_prop = n_prop,
    W0_ = W_0
  )$estimates

  b_pred <- de_dnn(
    X,
    Y,
    X_test = X_test,
    s_sizes = s_choice + 1,
    c = c,
    n_prop = n_prop,
    W0_ = W_0
  )$estimates

  (a_pred + b_pred) / 2
}

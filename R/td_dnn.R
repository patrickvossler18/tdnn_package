td_dnn <- function(X, Y, X_test, s_choice, W_0, c) {
  # Data checks before we doing anything else
  # Check X is a dataframe or matrix. If df, make it a matrix
  if (is.data.frame(X)) {
    warning(glue::glue("X is a dataframe. Attempting to convert to numeric matrix"))
    X <- as.matrix(X)
  }

  # Need all numeric columns for X matrix.
  # This might not be a necessary check since R coerces to numeric for matrix?
  if (!all(sapply(X, is.numeric))) {
    non_num_cols <- which(!(sapply(X, is.numeric)))
    stop(glue::glue("Found non-numeric columns. Column indices: {non_num_cols}"))
  }

  if (!is.matrix(Y)) {
    stop(glue("Y is of class {class(Y)[1]}, but needs to be a matrix"))
  }

    if (!is.matrix(X_test)) {
        stop(glue("X_test is of class {class(X_test)[1]}, but needs to be a matrix"))
    }

  # Check that dimensions match up for X, Y, X_test, and W_0
  # need length of Y to match dim[1] of X
  # need length of W_0 to match dim[2] of X
  # X_test can be of different length but ncols of X and X_test must match
  dimension_checks <- c(
    dim(X)[1] == length(Y),
    dim(X)[2] == length(W_0),
    dim(X)[2] == dim(X_test)[2]
  )
  if (!all(dimension_checks)) {
    # get which checks failed
    failed_checks <- which(!(dimension_checks))
    fail_messages <- sapply(failed_checks, function(x) {
      switch(EXPR=x,
        glue::glue("nrow of X ({dim(X)[1]}) does not equal length of Y: {length(Y)}"),
        glue::glue("ncol of X ({dim(X)[2]}) does not equal length of W_0: {length(W_0)}"),
        glue::glue("ncol of X ({dim(X)[2]}) does not equal ncol of X_test: {dim(X_test)[2]}")
      )
    })
    stop(glue::glue("Dimensions don't match: \n {toString(fail_messages)}"))
  }


  a_pred <- de_dnn(
    X,
    Y,
    X_test = X_test,
    s_sizes = s_choice,
    c = c,
    W0_ = W_0
  )$estimates

  b_pred <- de_dnn(
    X,
    Y,
    X_test = X_test,
    s_sizes = s_choice + 1,
    c = c,
    W0_ = W_0
  )$estimates

  (a_pred + b_pred) / 2
}

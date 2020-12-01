td_dnn <- function(X, Y, X_test, s.choice, W0) {
    # Data checks before we doing anything else
    # Check X is a dataframe or matrix. If df, make it a matrix
    if(is.data.frame(X)){
        warning(glue::glue("X is a dataframe. Attempting to convert to numeric matrix"))
        X <- as.matrix(X)
    }

    # Need all numeric columns for X matrix.
    # This might not be a necessary check since R coerces to numeric for matrix?
    if(!all(sapply(X, is.numeric))){

        non_num_cols <- which(!(sapply(X, is.numeric)))
        stop(glue::glue("Found non-numeric columns. Column indices: {non_num_cols}"))
    }

    # Check if Y is a matrix
    if(! is.matrix(Y)){
        stop(glue("Y is of class {class(Y)[1]}, but needs to be a matrix"))
        # Y = matrix(Y)
    }

    # Check that dimensions match up for X, Y, X_test, and W0
    # need length of Y to match dim[1] of X
    # need length of W0 to match dim[2] of X
    # X_test can be of different length but ncols of X and X_test must match
    dimension_checks <- c(dim(X)[1] == length(Y),
                          dim(X)[2] == length(W0),
                          dim(X)[2] == dim(X_test)[2])
    if(!all(dimension_checks)){
        # get which checks failed
        failed_checks <- which(!(dimension_checks))
        fail_messages <- sapply(failed_checks, function(x){switch(EXPR=x,
                                                                  glue::glue("nrow of X ({dim(X)[1]}) does not equal length of Y: {length(Y)}"),
                                                                  glue::glue("ncol of X ({dim(X)[2]}) does not equal length of W0: {length(W0)}"),
                                                                  glue::glue("ncol of X ({dim(X)[2]}) does not equal ncol of X_test: {dim(X_test)[2]}")
        )})
        stop(glue::glue("Dimensions don't match: \n {toString(fail_messages)}"))
    }

    X_fs = X[, which(W0 == 1)]
    X_test_fs = matrix(X_test[which(W0 == 1)], nrow = 1)

    a.pred = de_dnn(
        X_fs,
        Y,
        X_test = X_test_fs,
        s_size = s.choice,
        bc_p = 2
    )$estimates

    b.pred = de_dnn(
        X_fs,
        Y,
        X_test = X_test_fs,
        s_size = s.choice + 1,
        bc_p = 2
    )$estimates

    t.pred = (a.pred + b.pred) / 2
    return(t.pred)
}

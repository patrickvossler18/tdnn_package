est_effect = function(X, Y, Xtest, W0 = NULL, cpp = T) {
    t = 50
    s.choice0 <- tuning(seq(1, t, 1), X, Y, Xtest, 2, W0_ = W0)

    deDNN_pred = td_dnn(X,
                        Y,
                        X_test = Xtest,
                        s.choice = s.choice0,
                        W0 = W0)

    return(list(deDNN.pred = deDNN_pred, s.choice = s.choice0))
}

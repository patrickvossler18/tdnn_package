de.dnn <-
    function(Dataset,
             s.size = 2,
             bc.p = 2,
             X.test,
             W0) {
        n = nrow(Dataset)
        p = ncol(Dataset) - 1

        X = as.matrix(Dataset[, c(1:p)])
        Y = Dataset$Y
        # Weight
        ord = matrix(1:n , n , 1)
        # (n-k s-1) over (n s)
        weight1 = choose(n - ord, s.size - 1) / choose(n, s.size)
        weight2 = choose(n - ord, bc.p * s.size - 1) / choose(n, bc.p * s.size)
        # Distance
        X.dis = (X - kronecker(matrix(1, n, 1), X.test))*(matrix(1, n, 1) %*% W0)
        EuDis = (X.dis ^ 2) %*% matrix(1, p, 1)
        # Ascending small->large
        noise = matrix(rnorm(1), n, 1)
        TempD = data.frame(EuDis, Y, noise)[order(EuDis, noise), ]
        # Estimator
        U1 = sum(TempD$Y * weight1)
        U2 = sum(TempD$Y * weight2)
        Magic = solve(matrix(c(1, 1, 1, (1 / bc.p) ^ (2 / min(
            sum(W0), 3
        ))), 2, 2)) %*% matrix(c(1, 0), 2, 1)
        U = Magic[1, 1] * U1 + Magic[2, 1] * U2
        return(U)
    }

td.dnn <- function(Dataset,X.test, s.choice, W0){
    a.pred = de.dnn(Dataset,X.test = X.test, s.size = s.choice, W0=W0)
    b.pred = de.dnn(Dataset, X.test = X.test, s.size = s.choice + 1, W0=W0)
    t.pred = (a.pred + b.pred)/2
    return(t.pred)
}


compute_weight_sums <- function(y, weights){
    n_samples <- length(y)

    weight_sums <- matrix(0, n_samples, ncol(weights))

    # Buffer that contains the indices of current and last iterations
    indexes <- matrix(seq(2*n_samples), 2, n_samples, byrow=T)
    indexes[2,] <- 0 # do we need this for R?

    previous_indices <- indexes[1,]
    current_indices <- indexes[2,]


    merged_subarray_len <- 1

    while(merged_subarray_len < n_samples){
        gap <-  2 * merged_subarray_len
        indices_idx <- 1 # this is zero in the python version

        # apply(weights[previous_indices,], 2, cumsum)
        weights_cumsum <- do.call(rbind,
                list(matrix(0, nrow = 1, ncol = ncol(weights)),
                     apply(weights[previous_indices,], 2, cumsum)))
        # weights_cumsum <- c(0,cumsum(weights[previous_indices]))

        # Select the subarrays in pairs
        for(subarray_pair_idx in seq(1, n_samples, gap)){
            subarray_1_idx <- subarray_pair_idx
            subarray_2_idx <- subarray_pair_idx + merged_subarray_len

            subarray_1_idx_last <- min(
                subarray_1_idx + merged_subarray_len - 1, n_samples)

            subarray_2_idx_last = min(
                subarray_2_idx + merged_subarray_len - 1, n_samples)

            # Merge the subarrays
            while ( subarray_1_idx <= subarray_1_idx_last &
                    subarray_2_idx <= subarray_2_idx_last){
                previous_index_1 <- previous_indices[subarray_1_idx]
                previous_index_2 <- previous_indices[subarray_2_idx]

                if(y[previous_index_1] >= y[previous_index_2]){
                    current_indices[indices_idx] <-  previous_index_1
                    subarray_1_idx <- subarray_1_idx + 1
                } else{
                    current_indices[indices_idx] <-  previous_index_2
                    subarray_2_idx <- subarray_2_idx + 1

                    weight_sums[previous_index_2,] <- weight_sums[previous_index_2,] +
                        (weights_cumsum[subarray_1_idx_last + 1,] -
                             weights_cumsum[subarray_1_idx,])
                }
                indices_idx <- indices_idx + 1
            }
            # Join the remaining elements of one of the arrays (already sorted)
            if(subarray_1_idx <= subarray_1_idx_last){
                n_remaining <- subarray_1_idx_last - subarray_1_idx + 1
                indices_idx_next <- indices_idx + n_remaining
                # python version adds 1 to the end index, but I think that's only necessary for zero-indexing
                # For the R version we need to add 1 to indices_idx
                # print(paste0("(indices_idx):indices_idx_next: ",(indices_idx):indices_idx_next))
                # print(paste0("subarray_1_idx:subarray_1_idx_last: ",subarray_1_idx:subarray_1_idx_last))

                # Try forcing the update array length to match the current indices length
                if(length(subarray_1_idx:subarray_1_idx_last) != length((indices_idx):indices_idx_next)){
                    matched_length <- min(length((indices_idx):indices_idx_next),
                                          length(subarray_1_idx:subarray_1_idx_last))

                    current_indices_idx <- (indices_idx):indices_idx_next
                    prev_indices_idx <- subarray_1_idx:subarray_1_idx_last

                    current_indices_idx <- current_indices_idx[1:matched_length]
                    prev_indices_idx <- prev_indices_idx[1:matched_length]

                    current_indices[current_indices_idx] <- (previous_indices[prev_indices_idx])
                } else{
                    current_indices[(indices_idx):indices_idx_next] <- (previous_indices[subarray_1_idx:subarray_1_idx_last])
                }
                # current_indices[(indices_idx):indices_idx_next] <- (previous_indices[subarray_1_idx:subarray_1_idx_last])
                indices_idx <- indices_idx_next
            } else if (subarray_2_idx <= subarray_2_idx_last){
                n_remaining <- subarray_2_idx_last - subarray_2_idx + 1
                indices_idx_next <- indices_idx + n_remaining
                # python version adds 1 to the end index, but I think that's only necessary for zero-indexing
                # print(paste0("(indices_idx):indices_idx_next: ",(indices_idx):indices_idx_next))
                # print(paste0("subarray_2_idx:subarray_2_idx_last: ",subarray_2_idx:subarray_2_idx_last))
                # current_indices[(indices_idx):indices_idx_next] <- (previous_indices[subarray_2_idx:subarray_2_idx_last])

                if(length(subarray_2_idx:subarray_2_idx_last) != length((indices_idx):indices_idx_next)){
                    matched_length <- min(length((indices_idx):indices_idx_next),
                                          length(subarray_2_idx:subarray_2_idx_last))

                    current_indices_idx <- (indices_idx):indices_idx_next
                    prev_indices_idx <- subarray_2_idx:subarray_2_idx_last

                    current_indices_idx <- current_indices_idx[1:matched_length]
                    prev_indices_idx <- prev_indices_idx[1:matched_length]

                    current_indices[current_indices_idx] <- (previous_indices[prev_indices_idx])
                } else{
                    current_indices[(indices_idx):indices_idx_next] <- (previous_indices[subarray_2_idx:subarray_2_idx_last])
                }


                indices_idx <- indices_idx_next
            }
        }
        merged_subarray_len <- gap
        # swap buffers with tmp variable since we can't do tuple unpacking like Python
        current_tmp <- current_indices
        prev_tmp <- previous_indices

        previous_indices <- current_tmp
        current_indices <- prev_tmp
    }
    return(weight_sums)
}

computeaijbij_term <- function(x,y){

    n <- length(x)
    weights <- do.call("cbind",list(rep(1,length(y)), y, x, x * y))
    weight_sums <- compute_weight_sums(y, weights)

    term_1 <- as.numeric(t(x * y) %*% weight_sums[,1])
    term_2 <- as.numeric(t(x) %*% weight_sums[,2])
    term_3 <- as.numeric(t(y) %*% weight_sums[,3])
    term_4 <- as.numeric(sum(weight_sums[,4]))

    # First term in the equation
    sums_term <- term_1 - term_2 - term_3 + term_4

    # Second term in the equation
    sum_x = as.numeric(sum(x))
    sum_y = as.numeric(sum(y))
    cov_term = as.numeric(n * t(x) %*% y - sum(sum_x * y + sum_y * x) + sum_x * sum_y)

    d = 4 * sums_term - 2 * cov_term

    return(d)
}

compute_row_sums <- function(x){
    # assumes x is sorted

    n_samples <- length(x)

    term_1 <-  (2 * seq(1, n_samples) - n_samples) * x

    sums <- cumsum(x)

    term_2 <- sums[length(sums)] - 2 * sums

    return(term_1 + term_2)
}

generate_dcov_sqr <- function(x,y,unbiased=T){
    n <- length(x)

    ordered_indices <- order(x)
    x <- x[ordered_indices]
    y <- y[ordered_indices]

    aijbij <- computeaijbij_term(x,y)
    a_i <- compute_row_sums(x)

    ordered_indices_y <- order(y)
    b_i_perm <- compute_row_sums(y[ordered_indices_y])
    # Not sure what they are doing at this part in Python version
    b_i <- numeric(length(b_i_perm))
    b_i[ordered_indices_y] <- b_i_perm

    a_dot_dot <-  sum(a_i)
    b_dot_dot <- sum(b_i)
    sum_ab <- as.numeric(t(a_i) %*% b_i)

    if(unbiased){
        d3 <-  n - 3
        d2 <-  n - 2
        d1 <-  n - 1

    } else{
        d1 <- n
        d2 <- n
        d3 <- n
    }

    d_cov <- as.numeric(aijbij / n / d3 - 2 * sum_ab /n / d2 / d3 + a_dot_dot /n * b_dot_dot / d1 / d2 / d3)

    return(d_cov)
}


generate_bcdcor <- function(x,y){
    generate_dcov_sqr(x,y, unbiased = T)/
        sqrt(generate_dcov_sqr(x,x, unbiased = T) * generate_dcov_sqr(y,y, unbiased = T))
}

generate_bcdcor_T_pval <- function(x,y){
    bcdcor <- generate_bcdcor(x,y)
    n <- nrow(x)
    M <- n * (n-3) / 2
    df <- M - 1
    tstat <- sqrt(M-1) * bcdcor / sqrt(1- bcdcor^2)
    1 - stats::pt(tstat, df=df)
}

dcorT_test_mergesort <- function(x,y){
    dname <- paste(deparse(substitute(x)),"and",
                   deparse(substitute(y)))
    bcdcor <- generate_bcdcor(x,y)
    n <- nrow(x)
    M <- n * (n-3) / 2
    df <- M - 1
    names(df) <- "df"
    tstat <- sqrt(M-1) * bcdcor / sqrt(1- bcdcor^2)
    names(tstat) <- "T"
    estimate <- bcdcor
    names(estimate) <- "Bias corrected dcor"
    pval <- 1 - stats::pt(tstat, df=df)
    method <- "dcor t-test of independence for high dimension using merge sort algorithm"
    rval <- list(statistic = tstat, parameter = df, p.value = pval,
                 estimate=estimate, method=method, data.name=dname)
    class(rval) <- "htest"
    return(rval)
}


dcorT_parallel <- function(X, Y, alpha, n_cores=NULL){
    p <- ncol(X)

    # if null, only use one core
    if(is.null(n_cores)){
        return(sapply(1:ncol(X), function(i){
            p_val <- generate_bcdcor_T_pval(as.matrix(X[,i]), Y)
            as.numeric(p_val < alpha / p)
        }))
    } else{
        alp <- alpha # trying this to see if it allows us to properly export alpha to the cluster
        clust <- parallel::makeCluster(n_cores)
        # Look for these objects in the function environment
        parallel::clusterExport(clust, c("Y", "alp", "p"),envir = environment())
        res <- parallel::parSapply(clust,1:ncol(X), function(i){
            p_val <- generate_bcdcor_T_pval(as.matrix(X[,i]), Y)
            as.numeric(p_val < alp / p)
        })
        # Stop cluster to avoid warning messages
        parallel::stopCluster(clust)
        return(res)
    }


}




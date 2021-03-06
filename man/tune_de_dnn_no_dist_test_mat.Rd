% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tune_de_dnn.R
\name{tune_de_dnn_no_dist_test_mat}
\alias{tune_de_dnn_no_dist_test_mat}
\title{Calculate tuned de-DNN estimates using the R implementation
This version of the algorithm attempts to calculate the Euclidean distance matrix as
few times as possible.}
\usage{
tune_de_dnn_no_dist_test_mat(
  X,
  Y,
  X_test,
  c = 2,
  B_NN = 20,
  scale_p = 1,
  debug = F
)
}
\arguments{
\item{X}{Matrix of covariates}

\item{Y}{Matrix of responses}

\item{X_test}{Matrix of test observations for which we want to get an estimates}

\item{c}{Ratio between the subsampling scales: c = s_2/s_1}

\item{B_NN}{Top B_NN nearest neighbors used for tuning. Default is 20}

\item{scale_p}{Parameter for normalizing weights of the the top B_NN nearest neighbors. Default is 1.}

\item{debug}{Boolean flag for returning data frame with extra debug values}
}
\description{
Calculate tuned de-DNN estimates using the R implementation
This version of the algorithm attempts to calculate the Euclidean distance matrix as
few times as possible.
}

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_THOMASGRUNKEMEIER_P_H
#define WH_THOMASGRUNKEMEIER_P_H

double WH_ThomasGrunkemeier_P(double lambda, const Eigen::VectorXd& N, const Eigen::VectorXd& D) {
  int len = N.size();
  double product = 1.0;
  for (int i = 0; i < len; ++i) {
    product *= (1.0 - D(i) / (N(i) + lambda));
  }
  return product;
}

#endif
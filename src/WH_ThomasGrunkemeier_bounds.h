// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_THOMASGRUNKEMEIER_BOUNDS_H
#define WH_THOMASGRUNKEMEIER_BOUNDS_H

#include "WH_ThomasGrunkemeier_LRT.h"

double WH_ThomasGrunkemeier_lower(double theta, const Eigen::VectorXd& N,
                                  const Eigen::VectorXd& D, double L, 
                                  double L_min, int maxit = 100) { // L < 0
  double tol = 0.01;
  for (int i = 0; i < maxit; ++i) {
    if (L < L_min) {
      L = L_min + tol;
      tol *= 0.5;
    }
    double value = WH_ThomasGrunkemeier_LRT(L, N, D, theta);
    if (value < 0.0) return L;
    double slope = (theta - value) / L;
    double step  = (value - theta) / slope;
    L -= step;
  }
  throw std::runtime_error("[WH_ThomasGrunkemeier_lower] not found");
}

double WH_ThomasGrunkemeier_upper(double theta, const Eigen::VectorXd& N,
                                  const Eigen::VectorXd& D, double L, 
                                  int maxit = 100) { // L > 0
  for (int i = 0; i < maxit; ++i) {
    double value = WH_ThomasGrunkemeier_LRT(L, N, D, theta);
    if (value < 0.0) return L;
    double slope = (theta - value) / L;
    double step = (value - theta) / slope;
    L -= step;
  }
  throw std::runtime_error("[WH_ThomasGrunkemeier_upper] not found");
}

#endif
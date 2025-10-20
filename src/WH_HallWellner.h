// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_HALLWELLNER_H
#define WH_HALLWELLNER_H

double WH_HallWellner_G(double a, double lambda, double tol_G = 1e-10, int maxit_G = 10000) {
  double lambda2 = lambda * lambda;
  double r = lambda * std::sqrt((1.0 - a) / a);
  double d = 1.0 / (1.0 - a);
  double sum = 0.0;
  for (int k = 1; k <= maxit_G; ++k) {
    double coefficient = std::pow(-1.0, k) * std::exp(-2.0 * k * k * lambda2);
    double phi_difference = WH_cpstdnorm(r * (2.0 * k - d)) - WH_cpstdnorm(r * (2.0 * k + 2.0));
    double summand = coefficient * phi_difference;
    sum += summand;
    if (std::fabs(summand) < tol_G) return 1.0 - 2.0 * WH_cpstdnorm(lambda * std::pow(a * (1.0 - a), -0.5)) + 2.0 * sum;
  }
  throw std::runtime_error("[WH_HallWellner_G] non-convergence");
}

double WH_HallWellner_K(double x, double alpha = 0.05, double tol_G = 1e-10, int maxit_G = 10000, double tol_K = 1e-10, int maxit_K = 10000) {
  double lower = 0.0, upper = 5.0;
  for (int i = 0; i < maxit_K; ++i) {
    double mid = 0.5 * (lower + upper);
    double value = WH_HallWellner_G(x, mid, tol_G, maxit_G) - 1.0 + alpha;
    if (std::fabs(value) < tol_K || (upper - lower) < tol_K * (std::fabs(upper) + std::fabs(lower)) / 2.0) {
      return mid;
    }
    if (value > 0.0) upper = mid; else lower = mid;
  }
  throw std::runtime_error("[WH_HallWellner_K] non-convergence");
}

#endif

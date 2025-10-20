// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_HOLLANDERMCKEAGUE_H
#define WH_HOLLANDERMCKEAGUE_H

#include "WH_distr_chisq.h"
#include "WH_HallWellner.h"

// inputs must be at event-time steps, not all-time steps
double WH_HollanderMcKeague_alpha(double n, const Eigen::VectorXd& event,
                                  const Eigen::VectorXd& surv,
                                  const Eigen::VectorXd& SE, double alpha = 0.05,
                                  double tol_G = 1e-10, int maxit_G = 10000,
                                  double tol_K = 1e-10, int maxit_K = 10000) {
  int index = -1;
  for (int i = event.size() - 1; i >= 0; --i) {
    if (event(i) > 0.0) {
      index = i;
      break;
    }
  }
  if (index < 0) throw std::runtime_error("[WH_HollanderMcKeague_alpha] no events");
  double sigma = std::sqrt(n) * SE(index) / surv(index);
  double sigma2 = sigma * sigma;
  double K = WH_HallWellner_K(sigma2 / (1.0 + sigma2), alpha, tol_G, maxit_G, tol_K, maxit_K);
  double C = K * (1.0 + sigma2) / sigma;
  return 1.0 - WH_pchisq(C * C, 1.0);
}

Eigen::MatrixXd WH_HollanderMcKeague_summary(const Eigen::VectorXd& time,
                                             const Eigen::VectorXi& risk,
                                             const Eigen::VectorXi& event) {
  int n = time.size();
  int m = 0;
  for (int i = 0; i < n; ++i) if (event(i) > 0) ++m;
  Eigen::MatrixXd output(m, 3);
  int j = 0;
  double Greenwood_summand = 0.0;
  double surv = 1.0;
  for (int i = 0; i < n; ++i) {
    if (risk(i) == 0) continue;
    double p = static_cast<double>(event(i)) / static_cast<double>(risk(i));
    surv *= (1.0 - p);
    Greenwood_summand += p / (risk(i) * (1.0 - p));
    if (event(i) > 0) {
      double SE = surv * std::sqrt(Greenwood_summand);
      output(j, 0) = event(i);
      output(j, 1) = surv;
      output(j, 2) = SE;
      ++j;
    }
  }
  return output;
}

#endif

// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_HOLLANDERMCKEAGUE_H
#define WH_HOLLANDERMCKEAGUE_H

#include "WH_distr_chisq.h"
#include "WH_HallWellner.h"


//' Hollander–McKeague pointwise significance level
//'
//' Computes the *pointwise* \ifelse{latex}{\out{$\mathit{\alpha}_{\mathit{p}}$}}{\ifelse{html}{\out{<i>&alpha;</i><sub><i>p</i></sub>}}{*alpha_p*}} for Hollander–McKeague confidence bands.
//' @param risk Integer vector of numbers at risk at each unique event time.
//' @param event Integer vector of numbers of events at each unique event time.
//' @param surv Numeric vector of Kaplan–Meier survival estimates at each unique
//' event time.
//' @param SE Numeric vector of standard errors for each `surv`.
//' @param alpha Optional significance level. Default = `0.05`.
//' @param tol_G Optional convergence tolerance for the infinite series
//' computing the Hall–Wellner² distribution function *G*(*a*, *λ*). Default =
//' `1e-10`.
//' @param maxit_G Optional maximum number of iterations for the infinite series
//' computing the Hall–Wellner distribution function *G*(*a*, *λ*). Default =
//' `10000`.
//' @param tol_K Optional convergence tolerance for the bisection root-finding
//' of the Hall–Wellner quantile function *K*(*x*, *a*). Default = `1e-10`.
//' @param maxit_K Optional maximum number of iterations for the bisection
//' root-finding of the Hall–Wellner quantile function *K*(*x*, *a*). Default =
//' `10000`.
//' @return The pointwise significance level \ifelse{latex}{\out{$\mathit{\alpha}_{\mathit{p}}$}}{\ifelse{html}{\out{<i>&alpha;</i><sub><i>p</i></sub>}}{*alpha_p*}}.
//' @references
//' 1. Hollander, M. and McKeague, I.W., 1997. Likelihood ratio-based confidence
//' bands for survival functions. *Journal of the American Statistical
//' Association*, 92(437), pp. 215–226.
//' 2. Hall, W.J. and Wellner, J.A., 1980. Confidence bands for a survival curve
//' from censored data. *Biometrika*, 67(1), pp. 133–143.
//' @export
// [[Rcpp::export]]
double WH_HollanderMcKeague_alpha(const Eigen::VectorXi& risk, const Eigen::VectorXd& event,
                                  const Eigen::VectorXd& surv, const Eigen::VectorXd& SE,
                                  double alpha = 0.05,
                                  double tol_G = 1e-10, int maxit_G = 10000,
                                  double tol_K = 1e-10, int maxit_K = 10000) {
  if ((event.array() == 0).any()) {
    throw std::invalid_argument("[WH_HollanderMcKeague_alpha] inputs must be at event-time steps, not all-time steps");
  }
  double n = risk[0];
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

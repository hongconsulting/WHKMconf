// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>
#include "WH_distr_chisq.h"
#include "WH_distr_norm.h"
#include "WH_select_bool.h"
#include "WH_HollanderMcKeague.h"
#include "WH_Nair.h"
#include "WH_ThomasGrunkemeier_bounds.h"
#include "WH_ThomasGrunkemeier_LRT.h"
#include "WH_ThomasGrunkemeier_P.h"
#include "WH_unique.h"

Eigen::MatrixXd WH_KMexpand(const Eigen::VectorXd& time, const Eigen::VectorXi& risk, const Eigen::VectorXi& event) {
  int n = time.size();
  Eigen::VectorXi lost(n);
  for (int i = 0; i < n - 1; ++i) lost(i) = risk(i) - risk(i + 1);
  lost(n - 1) = risk(n - 1);
  Eigen::VectorXi censor = lost - event;
  int total = lost.sum();
  Eigen::MatrixXd output(total, 2);
  int index = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < censor(i); ++j) {
      output(index, 0) = time(i);
      output(index, 1) = 0.0;
      ++index;
    }
    for (int j = 0; j < event(i); ++j) {
      output(index, 0) = time(i);
      output(index, 1) = 1.0;
      ++index;
    }
  }
  return output;
}

Eigen::VectorXd WH_ThomasGrunkemeier_CI(const Eigen::VectorXd& time,
                                        const Eigen::VectorXi& status,
                                        double t0, double alpha = 0.05) {
  Eigen::VectorXd output(2);
  Eigen::VectorXd fail_time = WH_unique(WH_select_bool(time, status.array() == 1));
  int k = fail_time.size();
  Eigen::VectorXd D = Eigen::VectorXd::Ones(k);
  Eigen::VectorXd N = D;
  int j_max = 0;
  for (int i = 0; i < k; ++i) {
    D(i) = ((time.array() == fail_time(i)) && (status.array() == 1)).count();
    N(i) = (time.array() >= fail_time(i)).count();
    if (t0 >= fail_time(i)) j_max = i + 1;
  }
  Eigen::VectorXd N_sub = N.head(j_max);
  Eigen::VectorXd D_sub = D.head(j_max);
  double theta = WH_qchisq(1.0 - alpha, 1.0);
  if (j_max == 0 || D_sub.sum() == 0.0) {
    double lower = std::exp(-theta / (2.0 * N_sub.maxCoeff()));
    double upper = 1.0;
    output(0) = lower;
    output(1) = upper;
    return output;
  }
  double T = ((N_sub - D_sub).cwiseInverse() - N_sub.cwiseInverse()).sum();
  double L_start = -std::sqrt(theta / T);
  double L_min = D(j_max - 1) - N(j_max - 1);
  double L_lower0 = WH_ThomasGrunkemeier_lower(theta, N_sub, D_sub, L_start, L_min);
  double L_upper0 = WH_ThomasGrunkemeier_upper(theta, N_sub, D_sub, -L_lower0);
  double L_lower = WH_ThomasGrunkemeier_LRT_bisect(N_sub, D_sub, theta, L_lower0, 0.0);
  double L_upper = WH_ThomasGrunkemeier_LRT_bisect(N_sub, D_sub, theta, 0.0, L_upper0);
  output(0) = WH_ThomasGrunkemeier_P(L_lower, N_sub, D_sub);
  output(1) = WH_ThomasGrunkemeier_P(L_upper, N_sub, D_sub);
  return output;
}

////////////////////////////////////////////////////////////////////////////////

//' Nair confidence bands
//'
//' Computes the Nair's log-transformed equal-precision simultaneous confidence
//' band¹ for the Kaplan–Meier survival estimator at each unique event time.
//' @param time Numeric vector of unique event times.
//' @param surv Numeric vector of Kaplan–Meier survival estimates at each `time`.
//' @param SE Numeric vector of standard errors for each `surv`.
//' @param risk Integer vector of numbers at risk at each `time`.
//' @param event Integer vector of numbers of events at each `time`.
//' @param alpha Optional significance level. Default = `0.05`.
//' @param verbose Optional logical; if `TRUE`, prints additional diagnostic
//' information. Default = `TRUE`.
//' @param tol Optional convergence tolerance for the Borokov–Sycheva
//' approximation. Default = `1e-10`.
//' @param maxit Optional maximum number of iterations for the Borokov–Sycheva
//' approximation. Default = `10000`.
//' @param adapt Optional logical; if `TRUE`, computation of \ifelse{latex}{\out{$\mathit{e}_{\mathit{\alpha}}$}}{\ifelse{html}{\out{<i>e</i><sub><i>&alpha;</i></sub>}}{*e_alpha*}} automatically
//' falls back to the Monte Carlo method if the Borokov–Sycheva approximation
//' fails to converge. Default = `TRUE`.
//' @param MC_step Optional number of variance-stabilized Brownian bridge
//' discretization steps for the Monte Carlo fallback. Default = `1e5`.
//' @param MC_rep Optional number of replicates for the Monte Carlo fallback.
//' Default = `1e5`.
//' @param MC_seed Optional random seed for the Monte Carlo fallback. Default =
//' `24601`.
//' @param e_override Optional numeric value overriding the critical value \ifelse{latex}{\out{$\mathit{e}_{\mathit{\alpha}}$}}{\ifelse{html}{\out{<i>e</i><sub><i>&alpha;</i></sub>}}{*e_alpha*}}.
//' @return A numeric matrix with 2 columns: lower limit and upper limit.
//' @details
//' The equal-precision critical value \ifelse{latex}{\out{$\mathit{e}_{\mathit{\alpha}}$}}{\ifelse{html}{\out{<i>e</i><sub><i>&alpha;</i></sub>}}{*e_alpha*}} is calculated using Borokov–Sycheva
//' approximation² (`WH_e_alpha()`). This method aligns more closely with
//' Monte Carlo simulations using 100,000 Brownian bridge discretization steps
//' and 100,000 replicates (`WH_e_alpha_MC()`) compared to the pre-computed tables
//' published in Klein and Moeschberger (2003)³.
//' @references
//' 1. Nair, V.N., 1984. Conﬁdence bands for survival functions with censored
//' data: a comparative study. *Technometrics*, 26, pp. 265–275.
//' 2. Borokov, A.A. and Sycheva, N.M., 1968. On asymptotically optimal
//' non-parametric criteria. *Theory of Probability & Its Applications*, 13(3),
//' pp. 359–393.
//' 3. Klein, J.P. and Moeschberger, M.L., 2003. Appendix C: Statistical Tables.
//' In *Survival Analysis: Techniques for Censored and Truncated Data*, pp.
//' 455–482. New York: Springer New York.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd WH_Nair(const Eigen::VectorXd& time, const Eigen::VectorXd& surv,
                        const Eigen::VectorXd& SE, const Eigen::VectorXi& risk,
                        const Eigen::VectorXi& event, double alpha = 0.05,
                        bool verbose = true, double tol = 1e-10, int maxit = 10000,
                        bool adapt = true, int MC_step = 1e5, int MC_rep = 1e5,
                        int MC_seed = 24601, double e_override = 0.0) {
  int n_step = surv.size();
  int i_first = 0;
  while (i_first < n_step && event(i_first) == 0) ++i_first;
  int i_last = n_step - 1;
  while (i_last > i_first && event(i_last) == 0) --i_last;
  double S_first = surv(i_first);
  double S_last = surv(i_last);
  double n = risk[0];
  Eigen::VectorXd variance = (SE.array() / surv.array()).square();
  double variance_lower = variance(i_first);
  double variance_upper = variance(i_last);
  double lower = n * variance_lower / (1.0 + n * variance_lower);
  double upper = n * variance_upper / (1.0 + n * variance_upper);
  double e;
  if (e_override != 0.0) {
    e = e_override;
  } else {
    if (verbose) Rcpp::Rcout << "[WH_Nair] lower = " << lower << ", upper = " << upper << std::endl;
    if (adapt) {
      try {
        e = WH_e_alpha(lower, upper, alpha, tol, maxit);
      } catch (const std::exception&) {
        if (verbose) Rcpp::Rcout << "[WH_Nair] Monte Carlo fallback" << std::endl;
        e = WH_e_alpha_MC(lower, upper, alpha, MC_step, MC_rep, MC_seed);
      }
    } else {
      e = WH_e_alpha(lower, upper, alpha, tol, maxit);
    }
  }
  Eigen::VectorXd D = ((e * variance.array().sqrt()) / surv.array().log()).exp();
  Eigen::MatrixXd output(surv.size(), 2);
  output.col(0) = surv.array().pow(1.0 / D.array());
  output.col(1) = surv.array().pow(D.array());
  return output;
}

//' Rothman confidence intervals
//'
//' Computes Rothman binomial pointwise confidence intervals¹ for the
//' Kaplan–Meier survival estimator at each unique event time.
//' @param surv Numeric vector of Kaplan–Meier survival probabilities.
//' @param risk Integer vector of numbers at risk at each change in `surv`.
//' @param event Integer vector of event counts at each change in `surv`.
//' @param alpha Significance level. Default = `0.05`.
//' @return A numeric matrix with 2 columns: lower limit and upper limit.
//' @references
//' 1. Rothman, K.J., 1978. Estimation of confidence limits for the cumulative
//' probability of survival in life table analysis. *Journal of Chronic Diseases*,
//' 31(8), pp. 557–560.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd WH_Rothman(const Eigen::VectorXd& surv,
                           const Eigen::VectorXi& risk,
                           const Eigen::VectorXi& event,
                           double alpha = 0.05) {
  int n = surv.size();
  double Z = WH_qstdnorm(1.0 - 0.5 * alpha);
  double Z2 = Z * Z;
  Eigen::VectorXd risk_d = risk.cast<double>();
  Eigen::VectorXd event_d = event.cast<double>();
  Eigen::VectorXd Greenwood = event_d.array() / (risk_d.array() * (risk_d.array() - event_d.array()));
  Eigen::VectorXd Greenwood_cumsum = Greenwood;
  for (int i = 1; i < n; ++i) Greenwood_cumsum(i) += Greenwood_cumsum(i - 1);
  Eigen::VectorXd variance = Greenwood_cumsum.array() * surv.array().square();
  Eigen::VectorXd n_null = surv.array() * (1.0 - surv.array()) / variance.array();
  Eigen::VectorXd term = Z2 / (4.0 * n_null.array().square());
  Eigen::VectorXd root = (variance.array() + term.array()).sqrt();
  Eigen::VectorXd scale = n_null.array() / (n_null.array() + Z2);
  Eigen::VectorXd upper = scale.array() * (surv.array() + Z2 / (2.0 * n_null.array()) + Z * root.array());
  Eigen::VectorXd lower = scale.array() * (surv.array() + Z2 / (2.0 * n_null.array()) - Z * root.array());
  for (int i = 0; i < n; ++i) {
    if (std::isnan(upper(i))) upper(i) = 1.0;
    if (std::isnan(lower(i))) lower(i) = 1.0;
  }
  Eigen::MatrixXd output(n, 2);
  output.col(0) = lower;
  output.col(1) = upper;
  return output;
}

//' Thomas–Grunkemeier confidence intervals
//'
//' Computes Thomas–Grunkemeier likelihood-ratio pointwise confidence intervals¹
//' for the Kaplan–Meier survival estimator at each unique event time.
//' @param time Numeric vector of unique event times.
//' @param risk Integer vector of numbers at risk at each `time`.
//' @param event Integer vector of numbers of events at each `time`.
//' @param alpha Significance level. Default = `0.05`.
//' @return A numeric matrix with 2 columns: lower limit and upper limit.
//' @references
//' 1. Thomas, D.R. and Grunkemeier, G.L., 1975. Confidence interval estimation
//' of survival probabilities for censored data. *Journal of the American
//' Statistical Association*, 70(352), pp. 865–871.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd WH_ThomasGrunkemeier(const Eigen::VectorXd& time,
                                     const Eigen::VectorXi& risk,
                                     const Eigen::VectorXi& event,
                                     double alpha = 0.05) {
  int n = time.size();
  Eigen::MatrixXd X = WH_KMexpand(time, risk, event);
  Eigen::VectorXd X_time = X.col(0);
  Eigen::VectorXi X_status = X.col(1).cast<int>();
  int m = 0;
  for (int i = 0; i < n; ++i) if (event(i) > 0) ++m;
  Eigen::MatrixXd output = Eigen::MatrixXd::Constant(m, 2, std::numeric_limits<double>::quiet_NaN());
  int j = 0;
  for (int i = 0; i < n; ++i) {
    if (event(i) == 0) continue;
    // output(j, 0) = time(i);
    if (event(i) != risk(i)) {
      Eigen::VectorXd CI = WH_ThomasGrunkemeier_CI(X_time, X_status, time(i), alpha);
      output(j, 0) = CI(0);
      output(j, 1) = CI(1);
    }
    ++j;
  }
  return output;
}

////////////////////////////////////////////////////////////////////////////////

//' Hollander–McKeague confidence bands
//'
//' Computes the Hollander–McKeague likelihood-ratio simultaneous confidence
//' band¹ for the Kaplan–Meier survival estimator at each unique event time.
//' @param time Numeric vector of unique event times.
//' @param risk Integer vector of numbers at risk at each `time`.
//' @param event Integer vector of numbers of events at each `time`.
//' @param alpha Optional significance level. Default = `0.05`.
//' @param verbose Optional logical; if `TRUE`, prints additional diagnostic
//' information. Default = `TRUE`.
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
//' @return A numeric matrix with 2 columns: lower limit and upper limit.
//' @details
//' The Hollander–McKeague procedure begins with a specified *simultaneous* \ifelse{latex}{\out{$\mathit{\alpha}$}}{\ifelse{html}{\out{<i>&alpha;</i>}}{*alpha*}} and
//' computes a smaller *pointwise* \ifelse{latex}{\out{$\mathit{\alpha}$}}{\ifelse{html}{\out{<i>&alpha;</i>}}{*alpha*}} such that the resulting *pointwise* confidence
//' intervals achieve the desired *simultaneous* coverage. Extremely small
//' *pointwise* \ifelse{latex}{\out{$\mathit{\alpha}$}}{\ifelse{html}{\out{<i>&alpha;</i>}}{*alpha*}} can cause 1 − \ifelse{latex}{\out{$\mathit{\alpha}$}}{\ifelse{html}{\out{<i>&alpha;</i>}}{*alpha*}} to round to 1 in double precision. To prevent
//' this, *pointwise* \ifelse{latex}{\out{$\mathit{\alpha}$}}{\ifelse{html}{\out{<i>&alpha;</i>}}{*alpha*}} is rounded up to at least half the machine epsilon.
//' @references
//' 1. Hollander, M. and McKeague, I.W., 1997. Likelihood ratio-based confidence
//' bands for survival functions. *Journal of the American Statistical
//' Association*, 92(437), pp. 215–226.
//' 2. Hall, W.J. and Wellner, J.A., 1980. Confidence bands for a survival curve
//' from censored data. *Biometrika*, 67(1), pp. 133–143.
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd WH_HollanderMcKeague(const Eigen::VectorXd& time,
                                     const Eigen::VectorXi& risk,
                                     const Eigen::VectorXi& event,
                                     double alpha = 0.05, bool verbose = true,
                                     double tol_G = 1e-10, int maxit_G = 10000,
                                     double tol_K = 1e-10, int maxit_K = 10000) {
  Eigen::MatrixXd s = WH_HollanderMcKeague_summary(time, risk, event);
  double alpha_new = WH_HollanderMcKeague_alpha(risk, s.col(0), s.col(1), s.col(2),
                                                alpha, tol_G, maxit_G, tol_K, maxit_K);
  if (alpha_new < std::numeric_limits<double>::epsilon() / 2.0) {
    alpha_new = std::numeric_limits<double>::epsilon() / 2.0;
    if (verbose) Rcpp::Rcout << "[WH_HollanderMcKeague] pointwise \u03b1 rounded up to " << alpha_new << std::endl;
  }
  return WH_ThomasGrunkemeier(time, risk, event, alpha_new);
}

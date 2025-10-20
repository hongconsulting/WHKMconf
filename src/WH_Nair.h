// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_NAIR_H
#define WH_NAIR_H

#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace WH
{
constexpr double _SQRT8PI = 5.013256549262000483225; // sqrt(8 * pi)
}

double WH_Ax(double a, double b, double x) {
  return x * std::exp(-x * x / 2.0) *
    (std::log((1.0 - a) * b) - std::log(a * (1.0 - b))) /
      WH::_SQRT8PI;
}

//' Critical value for Nair's equal-precision confidence bands (Borokov–Sycheva)
//'
//' Computes the critical value *e* for Nair's equal-precision confidence bands¹
//' using Borokov–Sycheva approximation². Note: to reproduce Table 2 of Nair
//' (1982), use `alpha/2`.
//' @param lower Lower limit of supremum evaluation.
//' @param upper Upper limit of supremum evaluation.
//' @param alpha Significance level.
//' @param tol Convergence tolerance. Default = `1e-10`.
//' @param maxit Maximum number of iterations. Default = `10000`.
//' @return The critical value *e*.
//' @references
//' 1. Nair, V.N., 1984. Conﬁdence bands for survival functions with censored
//' data: a comparative study. *Technometrics*, 26, pp. 265–275.
//' 2. Borokov, A.A. and Sycheva, N.M., 1968. On asymptotically optimal
//' non-parametric criteria. *Theory of Probability & Its Applications*, 13(3),
//' pp. 359–393.
//' @export
// [[Rcpp::export]]
double WH_e_alpha_BS(double lower, double upper, double alpha, double tol = 1e-10, int maxit = 10000) {
  double x_lower = 0.5, x_upper = 5.0;
  double f_lower = WH_Ax(lower, upper, x_lower) - alpha;
  for (int i = 0; i < maxit; ++i) {
    double mid = 0.5 * (x_lower + x_upper);
    double f_mid = WH_Ax(lower, upper, mid) - alpha;
    if (std::fabs(f_mid) < tol) return mid;
    if (f_mid * f_lower < 0.0) {
      x_upper = mid;
    } else {
      x_lower = mid;
      f_lower = f_mid;
    }
  }
  std::cout << "[WH_e_alpha_BS] lower = " << lower << ", upper = " << upper << ", alpha = " << alpha << std::endl;
  throw std::runtime_error("[WH_e_alpha_BS] non-convergence");
}

//' Critical value for Nair's equal-precision confidence bands (Monte Carlo)
//'
//' Computes the critical value *e* for Nair's equal-precision confidence bands¹
//' using Monte Carlo simulation. Note: to reproduce Table 2 of Nair (1982), use
//' `alpha/2`.
//' @param lower Lower limit of supremum evaluation.
//' @param upper Upper limit of supremum evaluation.
//' @param alpha Significance level.
//' @param n_step Number of discretization steps for each Brownian bridge.
//' Default = `1e5`.
//' @param n_rep Number of Monte Carlo replicates. Default = `1e5`.
//' @param seed Random seed. Default = `24601`.
//' @details
//' Parallelized using OpenMP if available.
//' @return The critical value *e*.
//' @references
//' 1. Nair, V.N., 1984. Conﬁdence bands for survival functions with censored
//' data: a comparative study. *Technometrics*, 26, pp. 265–275.
//' @export
// [[Rcpp::export]]
double WH_e_alpha_MC(double lower, double upper, double alpha, int n_step = 1e5, int n_rep = 1e5, int seed = 24601) {
  Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(n_step, lower, upper);
  Eigen::VectorXd U(n_rep);
  #ifdef _OPENMP
  #pragma omp parallel
  {
  int thread_id = omp_get_thread_num();
  std::mt19937_64 RNG(seed + thread_id);
  std::normal_distribution<double> norm(0.0, 1.0);
  #pragma omp for
    for (int r = 0; r < n_rep; ++r) {
      Eigen::VectorXd B(n_step);
      B(0) = 0.0;
      for (int i = 1; i < n_step; ++i)
        B(i) = B(i - 1) + std::sqrt(x(i) - x(i - 1)) * norm(RNG);
      double B1 = B(n_step - 1);
      double max_value = 0.0;
      for (int i = 0; i < n_step; ++i) {
        double current_value = std::fabs(B(i) - x(i) * B1) / std::sqrt(x(i) * (1.0 - x(i)));
        if (current_value > max_value) max_value = current_value;
      }
      U(r) = max_value;
    }
  }
  #else
  std::mt19937_64 RNG(seed);
  std::normal_distribution<double> norm(0.0, 1.0);
  for (int r = 0; r < n_rep; ++r) {
    Eigen::VectorXd B(n_step);
    B(0) = 0.0;
    for (int i = 1; i < n_step; ++i)
      B(i) = B(i - 1) + std::sqrt(x(i) - x(i - 1)) * norm(RNG);
    double B1 = B(n_step - 1);
    double max_value = 0.0;
    for (int i = 0; i < n_step; ++i) {
      double current_value = std::fabs(B(i) - x(i) * B1) / std::sqrt(x(i) * (1.0 - x(i)));
      if (current_value > max_value) max_value = current_value;
    }
    U(r) = max_value;
  }
  #endif
  std::sort(U.data(), U.data() + n_rep);
  int index = static_cast<int>(std::ceil((1.0 - alpha) * n_rep)) - 1;
  return U(index);
}

#endif

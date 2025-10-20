// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_THOMASGRUNKEMEIER_LRT_H
#define WH_THOMASGRUNKEMEIER_LRT_H

double WH_ThomasGrunkemeier_LRT(double lambda, const Eigen::VectorXd& N, const Eigen::VectorXd& D, double theta) {
  int n = N.size();
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    double term1 = N(i) * std::log(1.0 + lambda / N(i));
    double term2 = (N(i) - D(i)) * std::log(1.0 + lambda / (N(i) - D(i)));
    sum += term1 - term2;
  }
  return theta - 2.0 * sum;
}

double WH_ThomasGrunkemeier_LRT_bisect(const Eigen::VectorXd& N, const Eigen::VectorXd& D,
                                       double theta, double a, double b,
                                       int maxit = 500) {
  double fa = WH_ThomasGrunkemeier_LRT(a, N, D, theta);
  double fb = WH_ThomasGrunkemeier_LRT(b, N, D, theta);
  if (fa * fb > 0.0) {
    // std::cout << "N = " << N.transpose() << std::endl;
    // std::cout << "D = " << D.transpose() << std::endl;
    // std::cout << "theta = " << theta << ", a = " << a << ", b = " << b
    //           << ", f(a) = " << fa << ", f(b) = " << fb << std::endl;
    throw std::runtime_error("[WH_ThomasGrunkemeier_LRT_bisect] f(a) and f(b) must have different signs");
  }
  double x1 = std::min(a, b);
  double x2 = std::max(a, b);
  double xm = 0.5 * (x1 + x2);
  int n = 1;
  while (x1 < xm && xm < x2 && n < maxit) { // while ((x2 - x1) > tol && n < maxit) {
    n++;
    if ((std::signbit(x1) != std::signbit(x2)) && x1 != 0.0 && x2 != 0.0) {
      xm = 0.0;
      if (WH_ThomasGrunkemeier_LRT(xm, N, D, theta) == 0.0) {
        x1 = 0.0;
        x2 = 0.0;
        break;
      }
    }
    if (std::signbit(WH_ThomasGrunkemeier_LRT(x1, N, D, theta)) !=
        std::signbit(WH_ThomasGrunkemeier_LRT(xm, N, D, theta))) {
      x2 = xm;
    } else {
      x1 = xm;
    }
    xm = 0.5 * (x1 + x2);
  }
  if (!(x1 < xm && xm < x2)) return xm; // if ((x2 - x1) <= tol) return xm;
  throw std::runtime_error("[WH_ThomasGrunkemaqeier_LRT_bisect] non-convergence");
}

#endif

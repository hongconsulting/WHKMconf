// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_DISTR_NORM_H
#define WH_DISTR_NORM_H

#include "WH_Cephes_SciPy_ndtri.h"
#include "WH_constexpr.h"

double WH_cpstdnorm(double x) { // standard normal distribution CCDF
  return 0.5 * std::erfc(x * WH::_SQRT1_2);
}

double WH_qstdnorm(double p) {
  return Cephes_SciPy::ndtri(p);
}

#endif

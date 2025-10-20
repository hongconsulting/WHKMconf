// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_DISTR_CHISQ_H
#define WH_DISTR_CHISQ_H

#include "WH_Cephes_SciPy_igam.h"
#include "WH_Cephes_SciPy_igami.h"

double WH_pchisq(double x, double df) {
  if( (x < 0.0) || (df < 0.0) ) {
    throw std::invalid_argument("[WH_pchisq] df < 0 or x < 0");
  }
  return(Cephes_SciPy::igam(df/2.0, x/2.0));
}

double WH_qchisq(double p, double df) {
  if (df <= 0.0) throw std::invalid_argument("[WH_qchisq] df < 0");
  if (p <= 0.0) return 0.0;
  if (p >= 1.0) return std::numeric_limits<double>::infinity();
  return 2.0 * Cephes_SciPy::igami(0.5 * df, 1.0 - p);
}

#endif
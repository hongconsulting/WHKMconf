/*
 * Relative error approximations for function arguments near
 * unity.
 *    log1p(x) = log(1+x)
 *    expm1(x) = exp(x) - 1
 *    cosm1(x) = cos(x) - 1
 *    lgam1p(x) = lgam(1+x)
 */

/* Scipy changes:
 * - 06-10-2016: added lgam1p
 */

#ifndef WH_CEPHES_SCIPY_UNITY_H
#define WH_CEPHES_SCIPY_UNITY_H

#include "WH_Cephes_SciPy_const.h"
#include "WH_Cephes_SciPy_mconf.h"
#include "WH_Cephes_SciPy_zeta.h"

namespace Cephes_SciPy
{

// #define MAXITER 500

/* log(1 + x) - x */
double log1pmx(double x) {
  if (fabs(x) < 0.5) {
    int n;
    double xfac = x;
    double term;
    double res = 0;
    for(n = 2; n < MAXITER; n++) {
      xfac *= -x;
      term = xfac / n;
      res += term;
      if (fabs(term) < MACHEP * fabs(res)) {
        break;
      }
    }
    return res;
  }
  else {
    return log1p(x) - x;
  }
}

/* Compute lgam(x + 1) around x = 0 using its Taylor series. */
static double lgam1p_taylor(double x)
{
  int n;
  double xfac, coeff, res;

  if (x == 0) {
    return 0;
  }
  res = -EULER * x;
  xfac = -x;
  for (n = 2; n < 42; n++) {
    xfac *= -x;
    coeff = zeta(n, 1) * xfac / n;
    res += coeff;
    if (fabs(coeff) < MACHEP * fabs(res)) {
      break;
    }
  }
  return res;
}

/* Compute lgam(x + 1). */
double lgam1p(double x) {
  if (fabs(x) <= 0.5) {
    return lgam1p_taylor(x);
  } else if (fabs(x - 1) < 0.5) {
    return log(x) + lgam1p_taylor(x - 1);
  } else {
    return std::lgamma(x + 1); //return lgam(x + 1);
  }
}

} // namespace

#endif
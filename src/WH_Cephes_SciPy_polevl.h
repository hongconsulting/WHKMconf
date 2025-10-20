/*
 * Cephes Math Library Release 2.1:  December, 1988
 * Copyright 1984, 1987, 1988 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 */

/* Sources:
 * [1] Holin et. al., "Polynomial and Rational Function Evaluation",
 *     https://www.boost.org/doc/libs/1_61_0/libs/math/doc/html/math_toolkit/roots/rational.html
 */

/* Scipy changes:
 * - 06-23-2016: add code for evaluating rational functions
 */

#ifndef WH_CEPHES_SCIPY_POLEVL_H
#define WH_CEPHES_SCIPY_POLEVL_H

namespace Cephes_SciPy
{

static double polevl(double x, const double coef[], int N) {
  double ans;
  int i;
  const double *p;
  p = coef;
  ans = *p++;
  i = N;
  do
    ans = ans * x + *p++;
  while (--i);
  return (ans);
}

/*                                                     p1evl() */
/*                                          N
 * Evaluate polynomial when coefficient of x  is 1.0.
 * Otherwise same as polevl.
 */

static double p1evl(double x, const double coef[], int N) {
  double ans;
  const double *p;
  int i;
  p = coef;
  ans = x + *p++;
  i = N - 1;
  do
    ans = ans * x + *p++;
  while (--i);
  return (ans);
}

static double ratevl(double x, const double num[], int M, const double denom[], int N) {
  int i, dir;
  double y, num_ans, denom_ans;
  double absx = fabs(x);
  const double *p;
  if (absx > 1) {
    /* Evaluate as a polynomial in 1/x. */
    dir = -1;
    p = num + M;
    y = 1 / x;
  } else {
    dir = 1;
    p = num;
    y = x;
  }
  /* Evaluate the numerator */
  num_ans = *p;
  p += dir;
  for (i = 1; i <= M; i++) {
    num_ans = num_ans * y + *p;
    p += dir;
  }
  /* Evaluate the denominator */
  if (absx > 1) {
    p = denom + N;
  } else {
    p = denom;
  }
  denom_ans = *p;
  p += dir;
  for (i = 1; i <= N; i++) {
    denom_ans = denom_ans * y + *p;
    p += dir;
  }
  if (absx > 1) {
    i = N - M;
    return pow(x, i) * num_ans / denom_ans;
  } else {
    return num_ans / denom_ans;
  }
}

} // namespace

#endif
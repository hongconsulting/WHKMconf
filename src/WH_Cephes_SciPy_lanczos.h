/*  (C) Copyright John Maddock 2006.
 *  Use, modification and distribution are subject to the
 *  Boost Software License, Version 1.0. (See accompanying file
 *  LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)
 */

/* Scipy changes:
 * - 06-22-2016: Removed all code not related to double precision and
 *   ported to c for use in Cephes
 */

#ifndef WH_CEPHES_SCIPY_LANCZOS_H
#define WH_CEPHES_SCIPY_LANCZOS_H

#include "WH_Cephes_SciPy_polevl.h"

namespace Cephes_SciPy
{

static const double lanczos_sum_expg_scaled_num[13] = {
  0.006061842346248906525783753964555936883222,
  0.5098416655656676188125178644804694509993,
  19.51992788247617482847860966235652136208,
  449.9445569063168119446858607650988409623,
  6955.999602515376140356310115515198987526,
  75999.29304014542649875303443598909137092,
  601859.6171681098786670226533699352302507,
  3481712.15498064590882071018964774556468,
  14605578.08768506808414169982791359218571,
  43338889.32467613834773723740590533316085,
  86363131.28813859145546927288977868422342,
  103794043.1163445451906271053616070238554,
  56906521.91347156388090791033559122686859
};

static const double lanczos_sum_expg_scaled_denom[13] = {
  1,
  66,
  1925,
  32670,
  357423,
  2637558,
  13339535,
  45995730,
  105258076,
  150917976,
  120543840,
  39916800,
  0
};

static const double lanczos_g = 6.024680040776729583740234375;

double lanczos_sum_expg_scaled(double x) {
  return ratevl(x, lanczos_sum_expg_scaled_num,
                sizeof(lanczos_sum_expg_scaled_num) / sizeof(lanczos_sum_expg_scaled_num[0]) - 1,
                lanczos_sum_expg_scaled_denom,
                sizeof(lanczos_sum_expg_scaled_denom) / sizeof(lanczos_sum_expg_scaled_denom[0]) - 1);
}

} // namespace

#endif
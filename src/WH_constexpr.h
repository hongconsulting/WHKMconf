// [[Rcpp::depends(RcppEigen)]]
#include <RcppEigen.h>

#ifndef WH_CONSTEXPR_H
#define WH_CONSTEXPR_H

namespace WH
{
constexpr double _0632     = 0.6321205588285576659757; // 1-1/e
constexpr double _1_PHI    = 0.6180339887498947915034; // 1/phi
constexpr double _1_PI     = 0.3183098861837906912164; // 1/pi
constexpr double _DAYSPERMONTH = 30.436875;
//constexpr double _LN_2_PI  = -0.4515827052894548221396; // log(2/pi)
constexpr double _PI       = 3.1415926535897932;
//constexpr double _PI_2     = 1.570796326794896557999; // pi/2
constexpr double _QNORM975 = 1.959963984540053383299; 
constexpr double _QNORM95  = 1.644853626951471525786;
constexpr double _SQRT1_2  = 0.7071067811865475727373; // sqrt(1/2)
constexpr double _SQRT2    = 1.414213562373095145475;
constexpr double _SQRT2_PI  = 0.7978845608028654057264; // sqrt(2/pi)
constexpr int _AIC            = 1;
constexpr int _AICc           = 2;
constexpr int _AICr           = 3;
constexpr int _BIC            = 4;
constexpr int _CAIC           = 5;
constexpr int _CAICF          = 6;
constexpr int _ICOMP_IFIM_C1  = 7;
}

#endif
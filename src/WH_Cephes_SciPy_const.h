/*
Cephes Math Library Release 2.3:  March, 1995
Copyright 1984, 1995 by Stephen L. Moshier
*/

#ifndef WH_CEPHES_SCIPY_CONST_H
#define WH_CEPHES_SCIPY_CONST_H

namespace Cephes_SciPy
{

constexpr double EULER     = 0.577215664901532860606512090082402431;
constexpr double MACHEP    = 1.11022302462515654042e-16;
constexpr double MAXLOG    = 7.08396418532264106224e2;
constexpr double MAXNUM    = 1.79769313486231570815e308; 
constexpr double MINLOG    = -7.08396418532264106224e2;
constexpr double PI        = 3.14159265358979323846;

constexpr double big       = 4.503599627370496e15; // igam, incbet
constexpr double biginv    = 2.22044604925031308085e-16; // igam, incbet
constexpr double LOGPI     = 1.14472988584940017414; // gamma
constexpr double LS2PI     = 0.91893853320467274178; // gamma: log(sqrt(2*pi))
constexpr double MAXGAM    = 171.624376956302725; // beta, gamma, incbet
constexpr double MAXLGM    = 2.556348e305; // gamma
constexpr double MAXSTIR   = 143.01608; // gamma
constexpr double s2pi      = 2.50662827463100050242e0; // ndtri: sqrt(2*pi)
constexpr double SQTPI     = 2.50662827463100050242E0; // gamma

}

#endif
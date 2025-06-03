/*
 * macac_math.h -
 *  MACA device functions for long double type
 */

#ifndef __MXCC_MACAC_MATH_H__
#define __MXCC_MACAC_MATH_H__

#if defined(EIGEN_MACA_ARCH)

__device__ inline long double sqrt(long double x) {
  return ::sqrtl(x);
}

__device__ inline long double abs(long double x) {
  return ::fabsl(x);
}

__device__ inline long double round(long double x) {
  return ::roundl(x);
}

__device__ inline long double floor(long double x) {
  return ::floorl(x);
}

__device__ inline long double ceil(long double x) {
  return ::ceill(x);
}

__device__ inline long double rint(long double x) {
  return ::rintl(x);
}

__device__ inline long double exp(long double x) {
  return ::expl(x);
}

__device__ inline long double log(long double x) {
  return ::log(x);
}

__device__ inline long double pow(long double x, long double y) {
  return ::powl(x, y);
}

__device__ inline long double cos(long double x) {
  return ::cosl(x);
}

__device__ inline long double sin(long double x) {
  return ::sinl(x);
}

__device__ inline long double tan(long double x) {
  return ::tanl(x);
}

__device__ inline long double acos(long double x) {
  return ::acosl(x);
}

__device__ inline long double acosh(long double x) {
  return ::acoshl(x);
}

__device__ inline long double asinh(long double x) {
  return ::asinhl(x);
}

__device__ inline long double atan(long double x) {
  return ::atanl(x);
}

__device__ inline long double atanh(long double x) {
  return ::atanhl(x);
}

__device__ inline long double cosh(long double x) {
  return ::coshl(x);
}

__device__ inline long double sinh(long double x) {
  return ::sinhl(x);
}

__device__ inline long double tanh(long double x) {
  return ::tanhl(x);
}

__device__ inline long double fmod(long double x, long double y) {
  return ::fmodl(x, y);
}

#endif  // EIGEN_MACA_ARCH

#endif  // __MXCC_MACAC_MATH_H__

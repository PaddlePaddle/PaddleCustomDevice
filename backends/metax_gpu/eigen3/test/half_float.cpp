// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>

#include "main.h"

#include <Eigen/src/Core/arch/Default/Half.h>

#define VERIFY_HALF_BITS_EQUAL(h, bits) \
  VERIFY_IS_EQUAL((numext::bit_cast<numext::uint16_t>(h)), (static_cast<numext::uint16_t>(bits)))

// Make sure it's possible to forward declare Eigen::half
namespace Eigen {
struct half;
}

using Eigen::half;

void test_conversion()
{
  #if defined(EIGEN_HAS_CUDA_FP16)
  using Eigen::half_impl::__half_raw;
  #endif

  // Round-trip bit-cast with uint16.
  VERIFY_IS_EQUAL(
    numext::bit_cast<Eigen::half>(numext::bit_cast<numext::uint16_t>(Eigen::half(1.0f))),
    Eigen::half(1.0f));
  VERIFY_IS_EQUAL(
    numext::bit_cast<Eigen::half>(numext::bit_cast<numext::uint16_t>(Eigen::half(0.5f))),
    Eigen::half(0.5f));
  VERIFY_IS_EQUAL(
    numext::bit_cast<Eigen::half>(numext::bit_cast<numext::uint16_t>(Eigen::half(-0.33333f))),
    Eigen::half(-0.33333f));
   VERIFY_IS_EQUAL(
    numext::bit_cast<Eigen::half>(numext::bit_cast<numext::uint16_t>(Eigen::half(0.0f))),
    Eigen::half(0.0f));

  // Conversion from float.
  VERIFY_HALF_BITS_EQUAL(Eigen::half(1.0f), 0x3c00);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(0.5f), 0x3800);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(0.33333f), 0x3555);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(0.0f), 0x0000);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(-0.0f), 0x8000);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(65504.0f), 0x7bff);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(65536.0f), 0x7c00);  // Becomes infinity.

  // Denormals.
  VERIFY_HALF_BITS_EQUAL(Eigen::half(-5.96046e-08f), 0x8001);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(5.96046e-08f), 0x0001);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(1.19209e-07f), 0x0002);

  // Verify round-to-nearest-even behavior.
  #if defined(EIGEN_HAS_CUDA_FP16)
  float val1 = float(Eigen::half(__half_raw(0x3c00)));
  float val2 = float(Eigen::half(__half_raw(0x3c01)));
  float val3 = float(Eigen::half(__half_raw(0x3c02)));
  #else
  float val1 = float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x3c00)));
  float val2 = float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x3c01)));
  float val3 = float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x3c02)));
  #endif
  VERIFY_HALF_BITS_EQUAL(Eigen::half(0.5f * (val1 + val2)), 0x3c00);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(0.5f * (val2 + val3)), 0x3c02);

  // Conversion from int.
  VERIFY_HALF_BITS_EQUAL(Eigen::half(-1), 0xbc00);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(0), 0x0000);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(1), 0x3c00);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(2), 0x4000);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(3), 0x4200);

  // Conversion from bool.
  VERIFY_HALF_BITS_EQUAL(Eigen::half(false), 0x0000);
  VERIFY_HALF_BITS_EQUAL(Eigen::half(true), 0x3c00);

  // Conversion to float.
  #if defined(EIGEN_HAS_CUDA_FP16)
  VERIFY_IS_EQUAL(float(Eigen::half(__half_raw(0x0000))), 0.0f);
  VERIFY_IS_EQUAL(float(Eigen::half(__half_raw(0x3c00))), 1.0f);
  #else
  VERIFY_IS_EQUAL(float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x0000))), 0.0f);
  VERIFY_IS_EQUAL(float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x3c00))), 1.0f);
  #endif

  // Denormals.
  #if defined(EIGEN_HAS_CUDA_FP16)
  VERIFY_IS_APPROX(float(Eigen::half(__half_raw(0x8001))), -5.96046e-08f);
  VERIFY_IS_APPROX(float(Eigen::half(__half_raw(0x0001))), 5.96046e-08f);
  VERIFY_IS_APPROX(float(Eigen::half(__half_raw(0x0002))), 1.19209e-07f);
  #else
  VERIFY_IS_APPROX(float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x8001))), -5.96046e-08f);
  VERIFY_IS_APPROX(float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x0001))), 5.96046e-08f);
  VERIFY_IS_APPROX(float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x0002))), 1.19209e-07f);
  #endif

  // NaNs and infinities.
  VERIFY(!(numext::isinf)(float(Eigen::half(65504.0f))));  // Largest finite number.
  VERIFY(!(numext::isnan)(float(Eigen::half(0.0f))));
  #if defined(EIGEN_HAS_CUDA_FP16)
  VERIFY((numext::isinf)(float(Eigen::half(__half_raw(0xfc00)))));
  VERIFY((numext::isnan)(float(Eigen::half(__half_raw(0xfc01)))));
  VERIFY((numext::isinf)(float(Eigen::half(__half_raw(0x7c00)))));
  VERIFY((numext::isnan)(float(Eigen::half(__half_raw(0x7c01)))));
  #else
  VERIFY((numext::isinf)(float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0xfc00)))));
  VERIFY((numext::isnan)(float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0xfc01)))));
  VERIFY((numext::isinf)(float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x7c00)))));
  VERIFY((numext::isnan)(float(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x7c01)))));
  #endif

#if !EIGEN_COMP_MSVC
  // Visual Studio errors out on divisions by 0
  VERIFY((numext::isnan)(float(Eigen::half(0.0 / 0.0))));
  VERIFY((numext::isinf)(float(Eigen::half(1.0 / 0.0))));
  VERIFY((numext::isinf)(float(Eigen::half(-1.0 / 0.0))));
#endif

  // Exactly same checks as above, just directly on the Eigen::half representation.
  #if defined(EIGEN_HAS_CUDA_FP16)
  VERIFY(!(numext::isinf)(Eigen::half(__half_raw(0x7bff))));
  VERIFY(!(numext::isnan)(Eigen::half(__half_raw(0x0000))));
  VERIFY((numext::isinf)(Eigen::half(__half_raw(0xfc00))));
  VERIFY((numext::isnan)(Eigen::half(__half_raw(0xfc01))));
  VERIFY((numext::isinf)(Eigen::half(__half_raw(0x7c00))));
  VERIFY((numext::isnan)(Eigen::half(__half_raw(0x7c01))));
  #else
  VERIFY(!(numext::isinf)(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x7bff))));
  VERIFY(!(numext::isnan)(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x0000))));
  VERIFY((numext::isinf)(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0xfc00))));
  VERIFY((numext::isnan)(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0xfc01))));
  VERIFY((numext::isinf)(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x7c00))));
  VERIFY((numext::isnan)(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x7c01))));
  #endif

#if !EIGEN_COMP_MSVC
  // Visual Studio errors out on divisions by 0
  VERIFY((numext::isnan)(Eigen::half(0.0 / 0.0)));
  VERIFY((numext::isinf)(Eigen::half(1.0 / 0.0)));
  VERIFY((numext::isinf)(Eigen::half(-1.0 / 0.0)));
#endif

  // Conversion to bool
  VERIFY(!static_cast<bool>(Eigen::half(0.0)));
  VERIFY(!static_cast<bool>(Eigen::half(-0.0)));
  #if defined(EIGEN_HAS_CUDA_FP16)
  VERIFY(static_cast<bool>(Eigen::half(__half_raw(0x7bff))));
  #else
  VERIFY(static_cast<bool>(Eigen::half(Eigen::half_impl::raw_uint16_to_half(0x7bff))));
  #endif
  VERIFY(static_cast<bool>(Eigen::half(-0.33333)));
  VERIFY(static_cast<bool>(Eigen::half(1.0)));
  VERIFY(static_cast<bool>(Eigen::half(-1.0)));
  VERIFY(static_cast<bool>(Eigen::half(-5.96046e-08f)));
}

void test_numtraits()
{
  std::cout << "epsilon       = " << NumTraits<Eigen::half>::epsilon() << "  (0x" << std::hex << numext::bit_cast<numext::uint16_t>(NumTraits<Eigen::half>::epsilon()) << ")" << std::endl;
  std::cout << "highest       = " << NumTraits<Eigen::half>::highest() << "  (0x" << std::hex << numext::bit_cast<numext::uint16_t>(NumTraits<Eigen::half>::highest()) << ")" << std::endl;
  std::cout << "lowest        = " << NumTraits<Eigen::half>::lowest() << "  (0x" << std::hex << numext::bit_cast<numext::uint16_t>(NumTraits<Eigen::half>::lowest()) << ")" << std::endl;
  std::cout << "min           = " << (std::numeric_limits<Eigen::half>::min)() << "  (0x" << std::hex << numext::bit_cast<numext::uint16_t>(Eigen::half((std::numeric_limits<Eigen::half>::min)())) << ")" << std::endl;
  std::cout << "denorm min    = " << (std::numeric_limits<Eigen::half>::denorm_min)() << "  (0x" << std::hex << numext::bit_cast<numext::uint16_t>(Eigen::half((std::numeric_limits<Eigen::half>::denorm_min)())) << ")" << std::endl;
  std::cout << "infinity      = " << NumTraits<Eigen::half>::infinity() << "  (0x" << std::hex << numext::bit_cast<numext::uint16_t>(NumTraits<Eigen::half>::infinity()) << ")" << std::endl;
  std::cout << "quiet nan     = " << NumTraits<Eigen::half>::quiet_NaN() << "  (0x" << std::hex << numext::bit_cast<numext::uint16_t>(NumTraits<Eigen::half>::quiet_NaN()) << ")" << std::endl;
  std::cout << "signaling nan = " << std::numeric_limits<Eigen::half>::signaling_NaN() << "  (0x" << std::hex << numext::bit_cast<numext::uint16_t>(std::numeric_limits<Eigen::half>::signaling_NaN()) << ")" << std::endl;

  VERIFY(NumTraits<Eigen::half>::IsSigned);

  VERIFY_IS_EQUAL(
    numext::bit_cast<numext::uint16_t>(std::numeric_limits<Eigen::half>::infinity()),
    numext::bit_cast<numext::uint16_t>(Eigen::half(std::numeric_limits<float>::infinity())) );
  // There is no guarantee that casting a 32-bit NaN to 16-bit has a precise
  // bit pattern.  We test that it is in fact a NaN, then test the signaling
  // bit (msb of significand is 1 for quiet, 0 for signaling).
  const numext::uint16_t HALF_QUIET_BIT = 0x0200;
  VERIFY(
    (numext::isnan)(std::numeric_limits<Eigen::half>::quiet_NaN())
    && (numext::isnan)(Eigen::half(std::numeric_limits<float>::quiet_NaN()))
    && ((numext::bit_cast<numext::uint16_t>(std::numeric_limits<Eigen::half>::quiet_NaN()) & HALF_QUIET_BIT) > 0)
    && ((numext::bit_cast<numext::uint16_t>(Eigen::half(std::numeric_limits<float>::quiet_NaN())) & HALF_QUIET_BIT) > 0) );
  // After a cast to Eigen::half, a signaling NaN may become non-signaling
  // (e.g. in the case of casting float to native __fp16). Thus, we check that
  // both are NaN, and that only the `numeric_limits` version is signaling.
  VERIFY(
    (numext::isnan)(std::numeric_limits<Eigen::half>::signaling_NaN())
    && (numext::isnan)(Eigen::half(std::numeric_limits<float>::signaling_NaN()))
    && ((numext::bit_cast<numext::uint16_t>(std::numeric_limits<Eigen::half>::signaling_NaN()) & HALF_QUIET_BIT) == 0) );

  VERIFY( (std::numeric_limits<Eigen::half>::min)() > Eigen::half(0.f) );
  VERIFY( (std::numeric_limits<Eigen::half>::denorm_min)() > Eigen::half(0.f) );
  VERIFY( (std::numeric_limits<Eigen::half>::min)()/Eigen::half(2) > Eigen::half(0.f) );
  VERIFY_IS_EQUAL( (std::numeric_limits<Eigen::half>::denorm_min)()/Eigen::half(2), Eigen::half(0.f) );
}

void test_arithmetic()
{
  VERIFY_IS_EQUAL(float(Eigen::half(2) + Eigen::half(2)), 4);
  VERIFY_IS_EQUAL(float(Eigen::half(2) + Eigen::half(-2)), 0);
  VERIFY_IS_APPROX(float(Eigen::half(0.33333f) + Eigen::half(0.66667f)), 1.0f);
  VERIFY_IS_EQUAL(float(Eigen::half(2.0f) * Eigen::half(-5.5f)), -11.0f);
  VERIFY_IS_APPROX(float(Eigen::half(1.0f) / Eigen::half(3.0f)), 0.33333f);
  VERIFY_IS_EQUAL(float(-Eigen::half(4096.0f)), -4096.0f);
  VERIFY_IS_EQUAL(float(-Eigen::half(-4096.0f)), 4096.0f);
  
  Eigen::half x(3);
  Eigen::half y = ++x;
  VERIFY_IS_EQUAL(x, Eigen::half(4));
  VERIFY_IS_EQUAL(y, Eigen::half(4));
  y = --x;
  VERIFY_IS_EQUAL(x, Eigen::half(3));
  VERIFY_IS_EQUAL(y, Eigen::half(3));
  y = x++;
  VERIFY_IS_EQUAL(x, Eigen::half(4));
  VERIFY_IS_EQUAL(y, Eigen::half(3));
  y = x--;
  VERIFY_IS_EQUAL(x, Eigen::half(3));
  VERIFY_IS_EQUAL(y, Eigen::half(4));
}

void test_comparison()
{
  VERIFY(Eigen::half(1.0f) > Eigen::half(0.5f));
  VERIFY(Eigen::half(0.5f) < Eigen::half(1.0f));
  VERIFY(!(Eigen::half(1.0f) < Eigen::half(0.5f)));
  VERIFY(!(Eigen::half(0.5f) > Eigen::half(1.0f)));

  VERIFY(!(Eigen::half(4.0f) > Eigen::half(4.0f)));
  VERIFY(!(Eigen::half(4.0f) < Eigen::half(4.0f)));

  VERIFY(!(Eigen::half(0.0f) < Eigen::half(-0.0f)));
  VERIFY(!(Eigen::half(-0.0f) < Eigen::half(0.0f)));
  VERIFY(!(Eigen::half(0.0f) > Eigen::half(-0.0f)));
  VERIFY(!(Eigen::half(-0.0f) > Eigen::half(0.0f)));

  VERIFY(Eigen::half(0.2f) > Eigen::half(-1.0f));
  VERIFY(Eigen::half(-1.0f) < Eigen::half(0.2f));
  VERIFY(Eigen::half(-16.0f) < Eigen::half(-15.0f));

  VERIFY(Eigen::half(1.0f) == Eigen::half(1.0f));
  VERIFY(Eigen::half(1.0f) != Eigen::half(2.0f));

  // Comparisons with NaNs and infinities.
#if !EIGEN_COMP_MSVC
  // Visual Studio errors out on divisions by 0
  VERIFY(!(Eigen::half(0.0 / 0.0) == Eigen::half(0.0 / 0.0)));
  VERIFY(Eigen::half(0.0 / 0.0) != Eigen::half(0.0 / 0.0));

  VERIFY(!(Eigen::half(1.0) == Eigen::half(0.0 / 0.0)));
  VERIFY(!(Eigen::half(1.0) < Eigen::half(0.0 / 0.0)));
  VERIFY(!(Eigen::half(1.0) > Eigen::half(0.0 / 0.0)));
  VERIFY(Eigen::half(1.0) != Eigen::half(0.0 / 0.0));

  VERIFY(Eigen::half(1.0) < Eigen::half(1.0 / 0.0));
  VERIFY(Eigen::half(1.0) > Eigen::half(-1.0 / 0.0));
#endif
}

void test_basic_functions()
{
  VERIFY_IS_EQUAL(float(numext::abs(Eigen::half(3.5f))), 3.5f);
  VERIFY_IS_EQUAL(float(abs(Eigen::half(3.5f))), 3.5f);
  VERIFY_IS_EQUAL(float(numext::abs(Eigen::half(-3.5f))), 3.5f);
  VERIFY_IS_EQUAL(float(abs(Eigen::half(-3.5f))), 3.5f);

  VERIFY_IS_EQUAL(float(numext::floor(Eigen::half(3.5f))), 3.0f);
  VERIFY_IS_EQUAL(float(floor(Eigen::half(3.5f))), 3.0f);
  VERIFY_IS_EQUAL(float(numext::floor(Eigen::half(-3.5f))), -4.0f);
  VERIFY_IS_EQUAL(float(floor(Eigen::half(-3.5f))), -4.0f);

  VERIFY_IS_EQUAL(float(numext::ceil(Eigen::half(3.5f))), 4.0f);
  VERIFY_IS_EQUAL(float(ceil(Eigen::half(3.5f))), 4.0f);
  VERIFY_IS_EQUAL(float(numext::ceil(Eigen::half(-3.5f))), -3.0f);
  VERIFY_IS_EQUAL(float(ceil(Eigen::half(-3.5f))), -3.0f);

  VERIFY_IS_APPROX(float(numext::sqrt(Eigen::half(0.0f))), 0.0f);
  VERIFY_IS_APPROX(float(sqrt(Eigen::half(0.0f))), 0.0f);
  VERIFY_IS_APPROX(float(numext::sqrt(Eigen::half(4.0f))), 2.0f);
  VERIFY_IS_APPROX(float(sqrt(Eigen::half(4.0f))), 2.0f);

  #if !defined(EIGEN_USE_MACA)
  // NOTE(xbei): this subtest compile failed, because there is no suitable device builtin function pow() for Eigen::half type.
  //   when EIGEN_USE_MACA is defined, the Eigen::half type definition is different from __half defined in maca_fp16.h
  VERIFY_IS_APPROX(float(numext::pow(Eigen::half(0.0f), Eigen::half(1.0f))), 0.0f);
  VERIFY_IS_APPROX(float(pow(Eigen::half(0.0f), Eigen::half(1.0f))), 0.0f);
  VERIFY_IS_APPROX(float(numext::pow(Eigen::half(2.0f), Eigen::half(2.0f))), 4.0f);
  VERIFY_IS_APPROX(float(pow(Eigen::half(2.0f), Eigen::half(2.0f))), 4.0f);
  #endif

  VERIFY_IS_EQUAL(float(numext::exp(Eigen::half(0.0f))), 1.0f);
  VERIFY_IS_EQUAL(float(exp(Eigen::half(0.0f))), 1.0f);
  VERIFY_IS_APPROX(float(numext::exp(Eigen::half(EIGEN_PI))), 20.f + float(EIGEN_PI));
  VERIFY_IS_APPROX(float(exp(Eigen::half(EIGEN_PI))), 20.f + float(EIGEN_PI));

  VERIFY_IS_EQUAL(float(numext::expm1(Eigen::half(0.0f))), 0.0f);
  VERIFY_IS_EQUAL(float(expm1(Eigen::half(0.0f))), 0.0f);
  VERIFY_IS_APPROX(float(numext::expm1(Eigen::half(2.0f))), 6.3890561f);
  VERIFY_IS_APPROX(float(expm1(Eigen::half(2.0f))), 6.3890561f);

  VERIFY_IS_EQUAL(float(numext::log(Eigen::half(1.0f))), 0.0f);
  VERIFY_IS_EQUAL(float(log(Eigen::half(1.0f))), 0.0f);
  VERIFY_IS_APPROX(float(numext::log(Eigen::half(10.0f))), 2.30273f);
  VERIFY_IS_APPROX(float(log(Eigen::half(10.0f))), 2.30273f);

  VERIFY_IS_EQUAL(float(numext::log1p(Eigen::half(0.0f))), 0.0f);
  VERIFY_IS_EQUAL(float(log1p(Eigen::half(0.0f))), 0.0f);
  VERIFY_IS_APPROX(float(numext::log1p(Eigen::half(10.0f))), 2.3978953f);
  VERIFY_IS_APPROX(float(log1p(Eigen::half(10.0f))), 2.3978953f);

  #if !defined(EIGEN_USE_MACA)
  // NOTE(xbei): this subtest compile failed, because there is no suitable device builtin function fmod() for Eigen::half type.
  //   when EIGEN_USE_MACA is defined, the Eigen::half type definition is different from __half defined in maca_fp16.h
  VERIFY_IS_APPROX(numext::fmod(Eigen::half(5.3f), Eigen::half(2.0f)), Eigen::half(1.3f));
  VERIFY_IS_APPROX(fmod(Eigen::half(5.3f), Eigen::half(2.0f)), Eigen::half(1.3f));
  VERIFY_IS_APPROX(numext::fmod(Eigen::half(-18.5f), Eigen::half(-4.2f)), Eigen::half(-1.7f));
  VERIFY_IS_APPROX(fmod(Eigen::half(-18.5f), Eigen::half(-4.2f)), Eigen::half(-1.7f));
  #endif
}

void test_trigonometric_functions()
{
  VERIFY_IS_APPROX(numext::cos(Eigen::half(0.0f)), Eigen::half(cosf(0.0f)));
  VERIFY_IS_APPROX(cos(Eigen::half(0.0f)), Eigen::half(cosf(0.0f)));
  VERIFY_IS_APPROX(numext::cos(Eigen::half(EIGEN_PI)), Eigen::half(cosf(EIGEN_PI)));
  // VERIFY_IS_APPROX(numext::cos(Eigen::half(EIGEN_PI/2)), Eigen::half(cosf(EIGEN_PI/2)));
  // VERIFY_IS_APPROX(numext::cos(Eigen::half(3*EIGEN_PI/2)), Eigen::half(cosf(3*EIGEN_PI/2)));
  VERIFY_IS_APPROX(numext::cos(Eigen::half(3.5f)), Eigen::half(cosf(3.5f)));

  VERIFY_IS_APPROX(numext::sin(Eigen::half(0.0f)), Eigen::half(sinf(0.0f)));
  VERIFY_IS_APPROX(sin(Eigen::half(0.0f)), Eigen::half(sinf(0.0f)));
  //  VERIFY_IS_APPROX(numext::sin(Eigen::half(EIGEN_PI)), Eigen::half(sinf(EIGEN_PI)));
  VERIFY_IS_APPROX(numext::sin(Eigen::half(EIGEN_PI/2)), Eigen::half(sinf(EIGEN_PI/2)));
  VERIFY_IS_APPROX(numext::sin(Eigen::half(3*EIGEN_PI/2)), Eigen::half(sinf(3*EIGEN_PI/2)));
  VERIFY_IS_APPROX(numext::sin(Eigen::half(3.5f)), Eigen::half(sinf(3.5f)));

  VERIFY_IS_APPROX(numext::tan(Eigen::half(0.0f)), Eigen::half(tanf(0.0f)));
  VERIFY_IS_APPROX(tan(Eigen::half(0.0f)), Eigen::half(tanf(0.0f)));
  //  VERIFY_IS_APPROX(numext::tan(Eigen::half(EIGEN_PI)), Eigen::half(tanf(EIGEN_PI)));
  //  VERIFY_IS_APPROX(numext::tan(Eigen::half(EIGEN_PI/2)), Eigen::half(tanf(EIGEN_PI/2)));
  //VERIFY_IS_APPROX(numext::tan(Eigen::half(3*EIGEN_PI/2)), Eigen::half(tanf(3*EIGEN_PI/2)));
  VERIFY_IS_APPROX(numext::tan(Eigen::half(3.5f)), Eigen::half(tanf(3.5f)));
}

void test_array()
{
  typedef Array<Eigen::half,1,Dynamic> ArrayXh;
  Index size = internal::random<Index>(1,10);
  Index i = internal::random<Index>(0,size-1);
  ArrayXh a1 = ArrayXh::Random(size), a2 = ArrayXh::Random(size);
  VERIFY_IS_APPROX( a1+a1, Eigen::half(2)*a1 );
  VERIFY( (a1.abs() >= Eigen::half(0)).all() );
  VERIFY_IS_APPROX( (a1*a1).sqrt(), a1.abs() );

  VERIFY( ((a1.min)(a2) <= (a1.max)(a2)).all() );
  a1(i) = Eigen::half(-10.);
  VERIFY_IS_EQUAL( a1.minCoeff(), Eigen::half(-10.) );
  a1(i) = Eigen::half(10.);
  VERIFY_IS_EQUAL( a1.maxCoeff(), Eigen::half(10.) );

  std::stringstream ss;
  ss << a1;
}

void test_product()
{
  typedef Matrix<Eigen::half,Dynamic,Dynamic> MatrixXh;
  Index rows  = internal::random<Index>(1,EIGEN_TEST_MAX_SIZE);
  Index cols  = internal::random<Index>(1,EIGEN_TEST_MAX_SIZE);
  Index depth = internal::random<Index>(1,EIGEN_TEST_MAX_SIZE);
  MatrixXh Ah = MatrixXh::Random(rows,depth);
  MatrixXh Bh = MatrixXh::Random(depth,cols);
  MatrixXh Ch = MatrixXh::Random(rows,cols);
  MatrixXf Af = Ah.cast<float>();
  MatrixXf Bf = Bh.cast<float>();
  MatrixXf Cf = Ch.cast<float>();
  VERIFY_IS_APPROX(Ch.noalias()+=Ah*Bh, (Cf.noalias()+=Af*Bf).cast<Eigen::half>());
}

EIGEN_DECLARE_TEST(half_float)
{
  CALL_SUBTEST(test_numtraits());
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST(test_conversion());
    CALL_SUBTEST(test_arithmetic());
    CALL_SUBTEST(test_comparison());
    CALL_SUBTEST(test_basic_functions());
    CALL_SUBTEST(test_trigonometric_functions());
    CALL_SUBTEST(test_array());
    CALL_SUBTEST(test_product());
  }
}

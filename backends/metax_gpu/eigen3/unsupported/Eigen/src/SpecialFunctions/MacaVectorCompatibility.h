#ifndef MACA_VECTOR_COMPATIBILITY_H
#define MACA_VECTOR_COMPATIBILITY_H

namespace maca_impl {
  template <typename, typename, unsigned int> struct Scalar_accessor;
}   // end namespace maca_impl

namespace Eigen {
namespace internal {

#define MACA_SCALAR_ACCESSOR_BUILDER(NAME) \
template <typename T, typename U, unsigned int n> \
struct NAME <maca_impl::Scalar_accessor<T, U, n>> : NAME <T> {};

#define MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(NAME) \
template <typename T, typename U, unsigned int n> \
struct NAME##_impl <maca_impl::Scalar_accessor<T, U, n>> : NAME##_impl <T> {}; \
template <typename T, typename U, unsigned int n> \
struct NAME##_retval <maca_impl::Scalar_accessor<T, U, n>> : NAME##_retval <T> {};

#define MACA_SCALAR_ACCESSOR_BUILDER_IGAMMA(NAME) \
template <typename T, typename U, unsigned int n, IgammaComputationMode mode> \
struct NAME <maca_impl::Scalar_accessor<T, U, n>, mode> : NAME <T, mode> {};

#if EIGEN_HAS_C99_MATH
MACA_SCALAR_ACCESSOR_BUILDER(betainc_helper)
MACA_SCALAR_ACCESSOR_BUILDER(incbeta_cfe)

MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(erf)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(erfc)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(igammac)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(lgamma)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(ndtri)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(polygamma)

MACA_SCALAR_ACCESSOR_BUILDER_IGAMMA(igamma_generic_impl)
#endif

MACA_SCALAR_ACCESSOR_BUILDER(digamma_impl_maybe_poly)
MACA_SCALAR_ACCESSOR_BUILDER(zeta_impl_series)

MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_i0)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_i0e)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_i1)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_i1e)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_j0)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_j1)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_k0)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_k0e)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_k1)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_k1e)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_y0)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(bessel_y1)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(betainc)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(digamma)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(gamma_sample_der_alpha)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(igamma_der_a)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(igamma)
MACA_SCALAR_ACCESSOR_BUILDER_RETVAL(zeta)

MACA_SCALAR_ACCESSOR_BUILDER_IGAMMA(igamma_series_impl)
MACA_SCALAR_ACCESSOR_BUILDER_IGAMMA(igammac_cf_impl)

}  // end namespace internal
}  // end namespace Eigen

#endif  // MACA_VECTOR_COMPATIBILITY_H

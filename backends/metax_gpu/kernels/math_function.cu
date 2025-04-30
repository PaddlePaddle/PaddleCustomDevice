#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/math_function_impl.h"

namespace phi {
namespace funcs {
template struct SetConstant<phi::CustomContext, float8_e4m3fn>;
template struct SetConstant<phi::CustomContext, float8_e5m2>;
template struct SetConstant<phi::CustomContext, float16>;
template struct SetConstant<phi::CustomContext, bfloat16>;
template struct SetConstant<phi::CustomContext, float>;
template struct SetConstant<phi::CustomContext, double>;
template struct SetConstant<phi::CustomContext, uint8_t>;
template struct SetConstant<phi::CustomContext, int8_t>;
template struct SetConstant<phi::CustomContext, int>;
template struct SetConstant<phi::CustomContext, int16_t>;
template struct SetConstant<phi::CustomContext, int64_t>;
template struct SetConstant<phi::CustomContext, bool>;
template struct SetConstant<phi::CustomContext, phi::dtype::complex<float>>;
template struct SetConstant<phi::CustomContext, phi::dtype::complex<double>>;

#define DEFINE_GPU_TRANS(RANK)                                        \
  template struct Transpose<phi::CustomContext, bool, RANK>;          \
  template struct Transpose<phi::CustomContext, unsigned char, RANK>; \
  template struct Transpose<phi::CustomContext, float, RANK>;         \
  template struct Transpose<phi::CustomContext, double, RANK>;        \
  template struct Transpose<phi::CustomContext, float8_e4m3fn, RANK>; \
  template struct Transpose<phi::CustomContext, float8_e5m2, RANK>;   \
  template struct Transpose<phi::CustomContext, float16, RANK>;       \
  template struct Transpose<phi::CustomContext, bfloat16, RANK>;      \
  template struct Transpose<phi::CustomContext, int8_t, RANK>;        \
  template struct Transpose<phi::CustomContext, int16_t, RANK>;       \
  template struct Transpose<phi::CustomContext, int32_t, RANK>;       \
  template struct Transpose<phi::CustomContext, int64_t, RANK>;       \
  template struct Transpose<phi::CustomContext,                       \
                            phi::dtype::complex<float>,               \
                            RANK>;                                    \
  template struct Transpose<phi::CustomContext,                       \
                            phi::dtype::complex<double>,              \
                            RANK>;

DEFINE_GPU_TRANS(1);
DEFINE_GPU_TRANS(2);
DEFINE_GPU_TRANS(3);
DEFINE_GPU_TRANS(4);
DEFINE_GPU_TRANS(5);
DEFINE_GPU_TRANS(6);
}  // namespace funcs
}  // namespace phi
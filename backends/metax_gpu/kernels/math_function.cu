#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/kernels/funcs/math_function_impl.h"
#include "paddle/phi/kernels/funcs/math_function.h"


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
}
}
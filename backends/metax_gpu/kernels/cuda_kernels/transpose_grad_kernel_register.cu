#include "paddle/phi/kernels/transpose_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(transpose_grad,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::TransposeGradKernel,
                   bool,
                   float,
                   double,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(trans_layout_grad,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::TransLayoutGradKernel,
                   bool,
                   float,
                   double,
                   int32_t,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
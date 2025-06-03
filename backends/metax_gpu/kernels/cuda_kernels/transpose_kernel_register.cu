#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(transpose,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::TransposeKernel,
                   bool,
                   float,
                   double,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

#include "paddle/phi/kernels/strided_slice_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"


PD_CUSTOM_KERNEL_REGISTER(strided_slice_raw_grad,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::StridedSliceRawGradKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(strided_slice_array_grad,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::StridedSliceArrayGradKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
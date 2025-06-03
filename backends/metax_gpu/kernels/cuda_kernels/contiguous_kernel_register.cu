
#include "paddle/phi/kernels/contiguous_kernel.h"

// #include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
// #include "paddle/phi/kernels/transpose_kernel.h"
PD_CUSTOM_KERNEL_REGISTER(contiguous,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::ContiguousKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int32_t,
                   int64_t,
                   float,
                   double,
                   ::phi::dtype::float16,
                   ::phi::dtype::bfloat16,
                   ::phi::dtype::complex<float>,
                   ::phi::dtype::complex<double>,
                   ::phi::dtype::float8_e4m3fn,
                   ::phi::dtype::float8_e5m2) {}

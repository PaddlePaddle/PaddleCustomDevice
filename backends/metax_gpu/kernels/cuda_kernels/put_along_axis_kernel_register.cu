#include "paddle/phi/kernels/put_along_axis_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(put_along_axis,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::PutAlongAxisKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

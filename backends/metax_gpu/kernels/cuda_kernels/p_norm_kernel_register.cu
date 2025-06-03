#include "paddle/phi/kernels/p_norm_kernel.h"


#include "paddle/phi/core/kernel_registry.h"


PD_CUSTOM_KERNEL_REGISTER(p_norm,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::PNormKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

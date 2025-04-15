#include "paddle/phi/kernels/index_add_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(index_add,
                          GPGPU,
                          ALL_LAYOUT,
                          phi::IndexAddKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int,
                          int64_t) {}
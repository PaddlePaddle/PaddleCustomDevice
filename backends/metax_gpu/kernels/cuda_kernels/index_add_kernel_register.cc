#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/index_add_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(index_add,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::IndexAddKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int,
                          int64_t) {}
#include "paddle/phi/kernels/eye_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(eye,
                          GPGPU,
                          ALL_LAYOUT,
                          phi::EyeKernel,
                          float,
                          double,
                          int64_t,
                          int,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
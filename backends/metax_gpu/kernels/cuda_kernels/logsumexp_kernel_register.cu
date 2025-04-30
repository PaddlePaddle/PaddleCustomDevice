#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/logsumexp_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(logsumexp,
                          metax_gpu,
                          ALL_LAYOUT,
                          phi::LogsumexpKernel,
                          float,
                          double,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

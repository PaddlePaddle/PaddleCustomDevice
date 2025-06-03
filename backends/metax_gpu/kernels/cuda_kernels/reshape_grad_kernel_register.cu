
#include "paddle/phi/kernels/reshape_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"






PD_CUSTOM_KERNEL_REGISTER_FOR_ALL_DTYPE(reshape_grad,
                                        metax_gpu,
                                         ALL_LAYOUT,
                                         phi::ReshapeGradKernel) {}

PD_CUSTOM_KERNEL_REGISTER_FOR_ALL_DTYPE(reshape_double_grad,
                                        metax_gpu,
                                         ALL_LAYOUT,
                                         phi::ReshapeDoubleGradKernel) {}

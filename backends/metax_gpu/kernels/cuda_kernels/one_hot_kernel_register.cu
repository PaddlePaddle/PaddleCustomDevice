#include "paddle/phi/kernels/one_hot_kernel.h"
#include "paddle/phi/core/kernel_registry.h"


PD_CUSTOM_KERNEL_REGISTER(one_hot, metax_gpu, ALL_LAYOUT, phi::OneHotKernel, int, int64_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
}


#include "paddle/phi/kernels/array_kernel.h"


#include "paddle/phi/core/kernel_registry.h"


PD_CUSTOM_KERNEL_REGISTER(create_array,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::CreateArrayKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(create_array_like,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::CreateArrayLikeKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(array_read,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::ArrayReadKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(array_write,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::ArrayWriteKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(array_to_tensor,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::ArrayToTensorKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}

PD_CUSTOM_KERNEL_REGISTER(array_pop,
                   metax_gpu,
                   ALL_LAYOUT,
                   phi::ArrayPopKernel,
                   bool,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
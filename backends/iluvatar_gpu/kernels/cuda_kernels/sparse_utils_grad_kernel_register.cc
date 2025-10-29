/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/sparse_utils_grad_kernel.h"

PD_CUSTOM_KERNEL_REGISTER(sparse_coo_tensor_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::SparseCooTensorGradKernel,
                          float,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          phi::dtype::complex<float>) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_CUSTOM_KERNEL_REGISTER(values_coo_grad,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::sparse::ValuesCooGradKernel,
                          float,
                          phi::float16,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::complex<float>) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

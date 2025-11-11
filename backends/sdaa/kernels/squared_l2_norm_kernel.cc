// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void SquaredL2NormKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SquaredL2NormKernel";
  dev_ctx.template Alloc<T>(out);

  tecodnnHandle_t tecodnn_handle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(x.dims()), x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t y_desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(out->dims()), out->dtype(), TensorFormat::NHWC);
  TECODNN_CHECK(tecodnnSquaredL2Norm(
      tecodnn_handle, x_desc, x.data(), y_desc, out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_desc));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(squared_l2_norm,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SquaredL2NormKernel,
                          float,
                          phi::dtype::float16) {}

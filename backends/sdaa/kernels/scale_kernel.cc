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

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::Scalar& in_scale,
                 float bias,
                 bool bias_after_scale,
                 phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA ScaleKernel";
  using MT = typename sdaa_ops::MPTypeTrait<T>::Type;
  auto scale = in_scale.to<MT>();
  auto bias_MT = static_cast<MT>(bias);

  if (isEnvEnable("HIGH_PERFORMANCE_CONV") && (&x != out) &&
      (x.storage_properties_initialized())) {
    SDAAStorageProperties x_properties =
        x.storage_properties<SDAAStorageProperties>();
    sdaa_ops::doAddStorageProperties(dev_ctx, out, x_properties);
  }

  dev_ctx.template Alloc<T>(out);

  if (x.numel() <= 0 || (!x.initialized())) {
    return;
  }

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  TECODNN_CHECK(tecodnnCustomScaleWithBias(tecodnnHandle,
                                           &scale,
                                           &bias_MT,
                                           bias_after_scale,
                                           x_Desc,
                                           x.data(),
                                           x_Desc,
                                           out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scale,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ScaleKernel,
                          double,
                          float,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int32_t,
                          int64_t,
                          phi::dtype::float16) {}

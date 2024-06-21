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
void ClipKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::Scalar& min,
                const phi::Scalar& max,
                phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA ClipKernel";
  dev_ctx.template Alloc<T>(out);

  auto max_ = max.to<T>();
  auto min_ = min.to<T>();

  PADDLE_ENFORCE_LE(min_,
                    max_,
                    phi::errors::InvalidArgument(
                        "max should be greater than or equal to min. "
                        "But received min = %lf, max = %lf",
                        static_cast<double>(min_),
                        static_cast<double>(max_)));

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::Undefined);

  phi::DenseTensor x_temp(x);

  TECODNN_CHECK(tecodnnClampTensor(tecodnnHandle,
                                   &min_,
                                   &max_,
                                   x_Desc,
                                   x_temp.data(),
                                   out_Desc,
                                   out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

template <typename T, typename Context>
void ClipGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    const phi::Scalar& min,
                    const phi::Scalar& max,
                    phi::DenseTensor* x_grad) {
  VLOG(4) << "Call SDAA ClipGradKernel";
  dev_ctx.template Alloc<T>(x_grad);

  double min_val = min.to<double>();
  double max_val = max.to<double>();

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_grad_dims = phi::vectorize<int>(out_grad.dims());
  std::vector<int> x_grad_dims = phi::vectorize<int>(x_grad->dims());
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_grad_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_grad_dims, out_grad.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t x_grad_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_grad_dims, x_grad->dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnClipBackward(tecodnnHandle,
                                    &min_val,
                                    &max_val,
                                    x_Desc,
                                    out_grad.data(),
                                    x_Desc,
                                    x.data(),
                                    x_Desc,
                                    x_grad->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_grad_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_grad_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(clip,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ClipKernel,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(clip_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ClipGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::dtype::float16) {}

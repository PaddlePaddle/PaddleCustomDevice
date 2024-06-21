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

#include "funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void doWhereTensor(const Context& dev_ctx,
                   const phi::DenseTensor& condition,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  VLOG(4) << "tecodnn where tensor called";

  std::vector<int> condition_dims = phi::vectorize<int>(condition.dims());
  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> y_dims = phi::vectorize<int>(y.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t condition_Desc = sdaa_ops::GetTecodnnTensorDesc(
      condition_dims, condition.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t x_Desc =
      sdaa_ops::GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t y_Desc =
      sdaa_ops::GetTecodnnTensorDesc(y_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t out_Desc =
      sdaa_ops::GetTecodnnTensorDesc(out_dims, x.dtype(), TensorFormat::NHWC);

  TECODNN_CHECK(tecodnnWhereTensor(tecodnnHandle,
                                   condition_Desc,
                                   condition.data(),
                                   x_Desc,
                                   x.data(),
                                   y_Desc,
                                   y.data(),
                                   out_Desc,
                                   out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(condition_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

template <typename T, typename Context>
void WhereKernel(const Context& dev_ctx,
                 const phi::DenseTensor& condition,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA WhereKernel";

  dev_ctx.template Alloc<T>(out);

  doWhereTensor<T, Context>(dev_ctx, condition, x, y, out);
}

template <typename T, typename Context>
void WhereGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& condition,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     const phi::DenseTensor& dout,
                     phi::DenseTensor* dx,
                     phi::DenseTensor* dy) {
  VLOG(4) << "CALL SDAA WhereGradKernel";

  if (dx) dev_ctx.template Alloc<T>(dx);
  if (dy) dev_ctx.template Alloc<T>(dy);

  phi::DenseTensor zero_tensor;
  phi::DenseTensorMeta zero_tensor_meta = {dout.dtype(), dout.dims()};
  zero_tensor.set_meta(zero_tensor_meta);
  dev_ctx.template Alloc<T>(&zero_tensor);

  sdaa_ops::doFillTensor<T>(
      dev_ctx, static_cast<T>(0.f), dout.dtype(), &zero_tensor);

  if (dy) {
    doWhereTensor<T, Context>(dev_ctx, condition, zero_tensor, dout, dy);
  }
  if (dx) {
    doWhereTensor<T, Context>(dev_ctx, condition, dout, zero_tensor, dx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(where,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::WhereKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(where_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::WhereGradKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int,
                          int64_t) {}

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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void LogSoftmaxKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      int axis,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  const int rank = x.dims().size();
  axis = CanonicalAxis(axis, rank);

  if (rank == 0) {
    auto out_dim = out->dims();
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(1), out);
    out->Resize(out_dim);
    return;
  }
  // cnnl softmax only support 3-dims, regard all shape as [d1, d2, d3]
  const int cnnl_softmax_dims = 3;
  const int d1 = custom_kernel::SizeToAxis(axis, x.dims());
  const int d2 = x.dims()[axis];
  const int d3 = custom_kernel::SizeOutAxis(axis, x.dims());

  // CNNL_SOFTMAX_MODE_LOW_DIMENSION has better perfermence, use it as much as
  // possible.
  cnnlSoftmaxMode_t mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
  std::vector<int> regard_in_shape{d1, 1, d2};
  if (d3 != 1) {
    mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
    regard_in_shape = {d1, d2, d3};
  }

  static const cnnlSoftmaxAlgorithm_t algo = CNNL_SOFTMAX_LOG;
  MLUCnnlTensorDesc in_desc(
      cnnl_softmax_dims, regard_in_shape.data(), ToCnnlDataType<T>());

  MLUCnnl::SoftmaxForward(dev_ctx,
                          algo,
                          mode,
                          NULL,
                          in_desc.get(),
                          GetBasePtr(&x),
                          NULL,
                          in_desc.get(),
                          GetBasePtr(out));
}

template <typename T, typename Context>
void LogSoftmaxGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& out,
                          const phi::DenseTensor& dout,
                          int axis,
                          phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  const int rank = dout.dims().size();
  axis = CanonicalAxis(axis, rank);

  if (rank == 0) {
    auto dx_dim = dx->dims();
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0), dx);
    dx->Resize(dx_dim);
    return;
  }

  // cnnl softmax only support 3-dims, regard all shape as [d1, d2, d3]
  const int cnnl_softmax_dims = 3;
  const int d1 = custom_kernel::SizeToAxis(axis, out.dims());
  const int d2 = out.dims()[axis];
  const int d3 = custom_kernel::SizeOutAxis(axis, out.dims());

  // CNNL_SOFTMAX_MODE_LOW_DIMENSION has better perfermence, use it as much as
  // possible.
  cnnlSoftmaxMode_t mode = CNNL_SOFTMAX_MODE_LOW_DIMENSION;
  std::vector<int> regard_out_shape{d1, 1, d2};
  if (d3 != 1) {
    mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
    regard_out_shape = {d1, d2, d3};
  }

  static const cnnlSoftmaxAlgorithm_t algo = CNNL_SOFTMAX_LOG;
  MLUCnnlTensorDesc out_desc(
      cnnl_softmax_dims, regard_out_shape.data(), ToCnnlDataType<T>());
  MLUCnnl::SoftmaxBackward(dev_ctx,
                           algo,
                           mode,
                           out_desc.get(),
                           GetBasePtr(&out),
                           out_desc.get(),
                           GetBasePtr(&dout),
                           out_desc.get(),
                           GetBasePtr(dx));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(log_softmax,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LogSoftmaxKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(log_softmax_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LogSoftmaxGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

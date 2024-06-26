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
void cumsum(const Context& dev_ctx,
            const phi::DenseTensor& x,
            const std::vector<int>& x_dims,
            int axis,
            phi::DenseTensor* out) {
  VLOG(4) << "tecodnn cumsum tensor called";

  int x_size = x_dims.size();
  axis += 4 - x_size;

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc =
      sdaa_ops::GetTecodnnTensorDesc(x_dims, x.dtype(), TensorFormat::NCHW);
  TECODNN_CHECK(tecodnnCumSum(
      tecodnnHandle, axis, x_Desc, x.data(), x_Desc, out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

template <typename T, typename Context>
void CumsumKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis_scalar,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA CumsumKernel";
  dev_ctx.template Alloc<T>(out);

  if (x.dtype() == phi::DataType::INT64 && (exclusive || reverse)) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "tecodnn not support exclusive=true or reverse=true "
        "when data type is int64 for cumsum op."));
  }

  auto axis = axis_scalar.to<int>();
  int xDims = x.dims().size();
  if (axis < 0) {
    axis += xDims;
  }
  std::vector<int> xdims;
  for (int i = 0; i < x.dims().size(); i++) {
    xdims.push_back(x.dims()[i]);
  }
  if (flatten) {
    PADDLE_ENFORCE_EQ(
        axis,
        xDims - 1,
        phi::errors::InvalidArgument(
            "when flatten is true, attr axis must be equal to (dims-1)"));
    xdims = {static_cast<int>(x.numel())};
    axis = 0;
  }

  custom_kernel::cumsum<T, Context>(dev_ctx, x, xdims, axis, out);

  if (reverse) {
    std::vector<int> reduce_dims = xdims;
    reduce_dims[axis] = 1;
    phi::DenseTensor Sn;
    phi::DDim Sn_dims = phi::make_ddim(reduce_dims);
    phi::DenseTensorMeta Sn_meta = {x.dtype(), Sn_dims};
    Sn.set_meta(Sn_meta);
    dev_ctx.template Alloc<T>(&Sn);
    std::vector<int64_t> reduce_axis = {static_cast<int64_t>(axis)};
    sdaa_ops::doSumTensor(dev_ctx, x, reduce_axis, &Sn);

    phi::DenseTensor Sm;
    phi::DenseTensorMeta Sm_meta = {x.dtype(), x.dims()};
    Sm.set_meta(Sm_meta);
    dev_ctx.template Alloc<T>(&Sm);
    sdaa_ops::doElementSub(dev_ctx, Sn, *out, -1, &Sm);
    if (exclusive) {
      phi::Copy(dev_ctx, Sm, dev_ctx.GetPlace(), false, out);
    } else {
      sdaa_ops::doElementAdd(dev_ctx, x, Sm, -1, out);
    }
  } else if (exclusive) {
    float alpha = -1.0;
    float beta = 1.0;
    sdaa_ops::doAddTensor(dev_ctx, x, alpha, beta, out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cumsum,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::CumsumKernel,
                          float,
                          int64_t,
                          phi::dtype::float16) {}

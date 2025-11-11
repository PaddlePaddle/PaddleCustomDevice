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

template <typename Context>
void bce_loss(const Context& dev_ctx,
              const void* x,
              const void* y,
              void* w,
              void* r,
              const std::vector<int>& x_dims,
              tecodnnDataType_t dt) {
  VLOG(4) << "tecodnn bce_loss forward tensor called";
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc;

  TECODNN_CHECK(tecodnnCreateTensorDescriptor(&x_Desc));

  int x_size = x_dims.size();

  int dim[4] = {1, 1, 1, 1};

  if (x_size > 4) {
    for (int i = 0; i < 3; ++i) {
      dim[i] = x_dims[i];
    }
    for (int i = 3; i < x_size; ++i) {
      dim[3] *= x_dims[i];
    }
  } else {
    int index = 3;
    for (int i = x_size - 1; i >= 0; i--) {
      dim[index--] = x_dims[i];
    }
  }

  const int N = dim[0];
  const int C = dim[1];
  const int H = dim[2];
  const int W = dim[3];

  tecodnnTensorFormat_t tf = TECODNN_TENSOR_NHWC;
  tecodnnLossReductionMode_t mode = TECODNN_LOSS_REDUCTION_NONE;
  TECODNN_CHECK(tecodnnSetTensor4dDescriptor(x_Desc, tf, dt, N, C, H, W));

  TECODNN_CHECK(tecodnnBCELossForward(
      tecodnnHandle, mode, x_Desc, x, x_Desc, y, x_Desc, w, x_Desc, r));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

template <typename Context>
void bce_loss_grad(const Context& dev_ctx,
                   const void* x,
                   const void* y,
                   void* w,
                   const void* dy,
                   void* dx,
                   const std::vector<int>& x_dims,
                   tecodnnDataType_t dt) {
  VLOG(4) << "tecodnn bce_loss backward tensor called";
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc;

  TECODNN_CHECK(tecodnnCreateTensorDescriptor(&x_Desc));

  int x_size = x_dims.size();

  int dim[4] = {1, 1, 1, 1};

  if (x_size > 4) {
    for (int i = 0; i < 3; ++i) {
      dim[i] = x_dims[i];
    }
    for (int i = 3; i < x_size; ++i) {
      dim[3] *= x_dims[i];
    }
  } else {
    int index = 3;
    for (int i = x_size - 1; i >= 0; i--) {
      dim[index--] = x_dims[i];
    }
  }

  const int N = dim[0];
  const int C = dim[1];
  const int H = dim[2];
  const int W = dim[3];

  tecodnnTensorFormat_t tf = TECODNN_TENSOR_NHWC;
  tecodnnLossReductionMode_t mode = TECODNN_LOSS_REDUCTION_NONE;
  TECODNN_CHECK(tecodnnSetTensor4dDescriptor(x_Desc, tf, dt, N, C, H, W));

  TECODNN_CHECK(tecodnnBCELossBackward(tecodnnHandle,
                                       mode,
                                       x_Desc,
                                       x,
                                       x_Desc,
                                       y,
                                       x_Desc,
                                       w,
                                       x_Desc,
                                       dy,
                                       x_Desc,
                                       dx));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

template <typename T, typename Context>
void BCELossKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& labels,
                   phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA BCELossKernel";
  dev_ctx.template Alloc<T>(out);
  std::vector<int> xdims;
  for (int i = 0; i < x.dims().size(); i++) {
    xdims.push_back(x.dims()[i]);
  }
  phi::DenseTensor w;
  phi::DenseTensorMeta w_meta = {x.dtype(), x.dims()};
  w.set_meta(w_meta);
  dev_ctx.template Alloc<T>(&w);
  sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(1), x.dtype(), &w);

  bce_loss(dev_ctx,
           x.data(),
           labels.data(),
           w.data(),
           out->data(),
           xdims,
           sdaa_ops::ToTecodnnDataType(phi::CppTypeToDataType<T>::Type()));
}

template <typename T, typename Context>
void BCELossGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& labels,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA BCELossGradKernel";
  dev_ctx.template Alloc<T>(dx);
  std::vector<int> xdims;
  for (int i = 0; i < x.dims().size(); i++) {
    xdims.push_back(x.dims()[i]);
  }
  phi::DenseTensor w;
  phi::DenseTensorMeta w_meta = {x.dtype(), x.dims()};
  w.set_meta(w_meta);
  dev_ctx.template Alloc<T>(&w);
  sdaa_ops::doFillTensor<T>(dev_ctx, static_cast<T>(1), x.dtype(), &w);
  bce_loss_grad(dev_ctx,
                x.data(),
                labels.data(),
                w.data(),
                dout.data(),
                dx->data(),
                xdims,
                sdaa_ops::ToTecodnnDataType(phi::CppTypeToDataType<T>::Type()));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(bce_loss,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::BCELossKernel,
                          float,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(bce_loss_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::BCELossGradKernel,
                          float,
                          phi::dtype::float16) {}

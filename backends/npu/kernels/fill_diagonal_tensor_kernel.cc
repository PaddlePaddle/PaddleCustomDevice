// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/string_helper.h"

namespace custom_kernel {

template <typename T, typename Context>
void FillDiagonalTensorKernel(const Context &dev_ctx,
                              const phi::DenseTensor &x,
                              const phi::DenseTensor &y,
                              int64_t offset,
                              int dim1,
                              int dim2,
                              phi::DenseTensor *out) {
  phi::DenseTensor output_cpu_t;
  output_cpu_t.Resize(out->dims());
  auto out_data = dev_ctx.template HostAlloc<T>(&output_cpu_t);
  TensorCopy(dev_ctx, x, false, &output_cpu_t, phi::CPUPlace());

  std::vector<T> fill_data;
  TensorToVector(dev_ctx, y, dev_ctx, &fill_data);
  auto out_dims = out->dims();
  auto matdims = y.dims();
  auto fill_dims = phi::flatten_to_2d(matdims, matdims.size() - 1);

  int64_t new_dims[2], strides[2];
  std::vector<int64_t> matdim;
  matdim.resize(fill_dims[0]);
  CalMatDims(out_dims, dim1, dim2, &offset, new_dims, strides, matdim.data());
  PADDLE_ENFORCE_EQ(
      new_dims[0],
      fill_dims[0],
      phi::errors::InvalidArgument("The dims should be %d x %d, but get "
                                   "%d x %d in fill tensor Y",
                                   new_dims[0],
                                   new_dims[1],
                                   fill_dims[0],
                                   fill_dims[1]));
  PADDLE_ENFORCE_EQ(
      new_dims[1],
      fill_dims[1],
      phi::errors::InvalidArgument("The dims should be %d x %d, but get "
                                   "%d x %d in fill tensor Y",
                                   new_dims[0],
                                   new_dims[1],
                                   fill_dims[0],
                                   fill_dims[1]));
  auto size = out->numel();
  for (int64_t i = 0; i < fill_dims[0]; i += 1) {
    auto sumoff = matdim[i] + offset;
    for (int64_t j = 0; j < fill_dims[1]; j += 1) {
      auto fill_index = j * (strides[1] + strides[0]) + sumoff;
      if (fill_index < size) {
        out_data[fill_index] = fill_data[i * fill_dims[1] + j];
      }
    }
  }
  dev_ctx.template Alloc<T>(out);
  TensorCopy(dev_ctx, output_cpu_t, true, out);

  /*
   * CANN Implementation with Error 507018

  std::vector<bool> mask(x.numel(), 0);
  for (int64_t i = 0; i < fill_dims[0]; i += 1) {
      auto sumoff = matdim[i] + offset;
      for (int64_t j = 0; j < fill_dims[1]; j += 1) {
          auto fill_index = j * (strides[1] + strides[0]) + sumoff;
          if (fill_index < size) {
              mask[fill_index] = true;
          }
      }
  }
  phi::DenseTensor mask_t;
  mask_t.Resize(x.dims());
  dev_ctx.template Alloc<bool>(&mask_t);
  TensorFromVector<bool>(dev_ctx, mask, dev_ctx, &mask_t);

  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner("MaskedScatter",
                                      {x, mask_t, y},
                                      {x},
                                      {});
  runner.Run(stream);
  TensorCopy(dev_ctx, x, true, out);
  }
  */
}

template <typename T, typename Context>
void FillDiagonalTensorGradKernel(const Context &dev_ctx,
                                  const phi::DenseTensor &out_grad,
                                  int64_t offset,
                                  int dim1,
                                  int dim2,
                                  phi::DenseTensor *x_grad) {
  auto matrows = 1;
  phi::DenseTensor x_grad_cpu;
  x_grad_cpu.Resize(x_grad->dims());
  auto data = dev_ctx.template HostAlloc<T>(&x_grad_cpu);
  TensorCopy(dev_ctx, out_grad, true, &x_grad_cpu, phi::CPUPlace());

  if (x_grad) {
    auto dx_dims = x_grad->dims();
    for (int i = 0; i < dx_dims.size(); i++) {
      if (i != dim1 && i != dim2) {
        matrows *= dx_dims[i];
      }
    }

    int64_t new_dims[2], strides[2];
    std::vector<int64_t> matdim;
    matdim.resize(matrows);
    CalMatDims(dx_dims, dim1, dim2, &offset, new_dims, strides, matdim.data());

    auto size = x_grad->numel();

    for (int64_t i = 0; i < new_dims[0]; i += 1) {
      auto sumoff = matdim[i] + offset;
      for (int64_t j = 0; j < new_dims[1]; j += 1) {
        auto fill_index = j * (strides[1] + strides[0]) + sumoff;
        if (fill_index < size) {
          data[fill_index] = 0;
        }
      }
    }
  }
  dev_ctx.template Alloc<T>(x_grad);
  TensorCopy(dev_ctx, x_grad_cpu, true, x_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(fill_diagonal_tensor,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FillDiagonalTensorKernel,
                          float,
                          double,
                          int64_t,
                          int,
                          int8_t,
                          uint8_t,
                          bool,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(fill_diagonal_tensor_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::FillDiagonalTensorGradKernel,
                          float,
                          double,
                          int64_t,
                          int,
                          int8_t,
                          uint8_t,
                          bool,
                          phi::dtype::float16) {}

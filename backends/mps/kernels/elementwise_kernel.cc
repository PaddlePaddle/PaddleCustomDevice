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

#include "kernels/elementwise_impl.h"
#include "kernels/phi_funcs.h"
#include "paddle/phi/capi/all.h"

namespace custom_kernel {

template <typename T>
void BinaryElementwiseKernel(const phi::Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& y,
                             int axis,
                             phi::DenseTensor* out,
                             mps_kernel::MPSElementwiseOP op) {
  dev_ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto dst_dims = phi::BroadcastDims(axis, x_dims, y_dims);
  phi::DenseTensor tmp_x, tmp_y;
  phi::BroadcastTo<T>(dev_ctx, x, dst_dims, axis, &tmp_x);
  phi::BroadcastTo<T>(dev_ctx, y, dst_dims, axis, &tmp_y);

  auto x_data = tmp_x.data<T>();
  auto y_data = tmp_y.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);
  mps_kernel::Elementwise(x_data, y_data, out_data, dst_dims, op);
}

template <typename T>
void AddRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  BinaryElementwiseKernel<T>(
      dev_ctx, x, y, axis, out, mps_kernel::MPSElementwiseOP::ADD);
}

template <typename T>
void AddKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::AddRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T>
void DivRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  BinaryElementwiseKernel<T>(
      dev_ctx, x, y, axis, out, mps_kernel::MPSElementwiseOP::DIV);
}

template <typename T>
void DivKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::DivRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T>
void MulRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  BinaryElementwiseKernel<T>(
      dev_ctx, x, y, axis, out, mps_kernel::MPSElementwiseOP::MUL);
}

template <typename T>
void MulKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::MulRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T>
void SubRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  BinaryElementwiseKernel<T>(
      dev_ctx, x, y, axis, out, mps_kernel::MPSElementwiseOP::SUB);
}

template <typename T>
void SubKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::SubRawKernel<T>(dev_ctx, x, y, axis, out);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(
    add_raw, mps, ALL_LAYOUT, custom_kernel::AddRawKernel, float) {}

PD_BUILD_PHI_KERNEL(add, mps, ALL_LAYOUT, custom_kernel::AddKernel, float) {}

PD_BUILD_PHI_KERNEL(
    divide_raw, mps, ALL_LAYOUT, custom_kernel::DivRawKernel, float) {}

PD_BUILD_PHI_KERNEL(divide, mps, ALL_LAYOUT, custom_kernel::DivKernel, float) {}

PD_BUILD_PHI_KERNEL(
    multiply_raw, mps, ALL_LAYOUT, custom_kernel::MulRawKernel, float) {}

PD_BUILD_PHI_KERNEL(
    multiply, mps, ALL_LAYOUT, custom_kernel::MulKernel, float) {}

PD_BUILD_PHI_KERNEL(
    subtract_raw, mps, ALL_LAYOUT, custom_kernel::SubRawKernel, float) {}

PD_BUILD_PHI_KERNEL(
    subtract, mps, ALL_LAYOUT, custom_kernel::SubKernel, float) {}

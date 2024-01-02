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

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"  //NOLINT

namespace custom_kernel {

template <typename T>
void MultiplyRawKernel(const phi::Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  auto dst_dims = phi::BroadcastDims(axis, x_dims, y_dims);
  phi::DenseTensor tmp_x, tmp_y;
  phi::BroadcastTo<T>(dev_ctx, x, dst_dims, axis, &tmp_x);
  phi::BroadcastTo<T>(dev_ctx, y, dst_dims, axis, &tmp_y);

  auto x_data = tmp_x.data<T>();
  auto y_data = tmp_y.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] * y_data[i];
  }
}

template <typename T>
void MultiplyKernel(const phi::Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  int axis = -1;
  MultiplyRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T>
void AddRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
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
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] + y_data[i];
  }
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
void MaxRawKernel(const phi::Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
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
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = std::max(x_data[i], y_data[i]);
  }
}

template <typename T>
void MaxKernel(const phi::Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::MaxRawKernel<T>(dev_ctx, x, y, axis, out);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(multiply_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::MultiplyRawKernel,
                    int32_t,
                    int64_t,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(multiply,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::MultiplyKernel,
                    int32_t,
                    int64_t,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(add_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::AddRawKernel,
                    int32_t,
                    int64_t,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(add,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::AddKernel,
                    int32_t,
                    int64_t,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(maximum_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::MaxRawKernel,
                    int32_t,
                    int64_t,
                    float,
                    double) {}

PD_BUILD_PHI_KERNEL(maximum,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::MaxKernel,
                    int32_t,
                    int64_t,
                    float,
                    double) {}

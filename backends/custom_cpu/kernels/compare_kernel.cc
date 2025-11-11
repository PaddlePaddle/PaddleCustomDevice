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

#include <cmath>

#include "paddle/phi/capi/all.h"
#include "phi_funcs.h"  //NOLINT

namespace custom_kernel {

template <typename T>
void NotEqualRawKernel(const phi::Context& dev_ctx,
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
  auto out_data = dev_ctx.template Alloc<bool>(out);
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    if (std::is_floating_point<T>::value) {
      out_data[i] = static_cast<bool>(
          fabs(static_cast<double>(x_data[i] - y_data[i])) >= 1e-8);
    } else {
      out_data[i] = x_data[i] != y_data[i];
    }
  }
}

template <typename T>
void NotEqualKernel(const phi::Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  custom_kernel::NotEqualRawKernel<T>(dev_ctx, x, y, -1, out);
}

template <typename T>
void EqualRawKernel(const phi::Context& dev_ctx,
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
  auto out_data = dev_ctx.template Alloc<bool>(out);
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    if (std::is_floating_point<T>::value) {
      out_data[i] = static_cast<bool>(
          fabs(static_cast<double>(x_data[i] - y_data[i])) < 1e-8);
    } else {
      out_data[i] = x_data[i] == y_data[i];
    }
  }
}

template <typename T>
void EqualKernel(const phi::Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  custom_kernel::EqualRawKernel<T>(dev_ctx, x, y, -1, out);
}

template <typename T>
void LessThanRawKernel(const phi::Context& dev_ctx,
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
  auto out_data = dev_ctx.template Alloc<bool>(out);
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] < y_data[i];
  }
}

template <typename T>
void LessThanKernel(const phi::Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  custom_kernel::LessThanRawKernel<T>(dev_ctx, x, y, -1, out);
}

template <typename T>
void LessEqualRawKernel(const phi::Context& dev_ctx,
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
  auto out_data = dev_ctx.template Alloc<bool>(out);
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] <= y_data[i];
  }
}

template <typename T>
void LessEqualKernel(const phi::Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  custom_kernel::LessEqualRawKernel<T>(dev_ctx, x, y, -1, out);
}

template <typename T>
void GreaterThanRawKernel(const phi::Context& dev_ctx,
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
  auto out_data = dev_ctx.template Alloc<bool>(out);
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] > y_data[i];
  }
}

template <typename T>
void GreaterThanKernel(const phi::Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       phi::DenseTensor* out) {
  custom_kernel::GreaterThanRawKernel<T>(dev_ctx, x, y, -1, out);
}

template <typename T>
void GreaterEqualRawKernel(const phi::Context& dev_ctx,
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
  auto out_data = dev_ctx.template Alloc<bool>(out);
  auto numel = out->numel();
  for (auto i = 0; i < numel; ++i) {
    out_data[i] = x_data[i] >= y_data[i];
  }
}

template <typename T>
void GreaterEqualKernel(const phi::Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        phi::DenseTensor* out) {
  custom_kernel::GreaterEqualRawKernel<T>(dev_ctx, x, y, -1, out);
}

}  // namespace custom_kernel

PD_BUILD_PHI_KERNEL(not_equal,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::NotEqualKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(not_equal_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::NotEqualRawKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(equal,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::EqualKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(equal_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::EqualRawKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(less_than,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::LessThanKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(less_than_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::LessThanRawKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(less_equal,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::LessEqualKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(less_equal_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::LessEqualRawKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(greater_than,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::GreaterThanKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(greater_than_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::GreaterThanRawKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(greater_equal,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::GreaterEqualKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

PD_BUILD_PHI_KERNEL(greater_equal_raw,
                    custom_cpu,
                    ALL_LAYOUT,
                    custom_kernel::GreaterEqualRawKernel,
                    float,
                    double,
                    uint8_t,
                    int16_t,
                    int32_t,
                    int64_t,
                    bool) {}

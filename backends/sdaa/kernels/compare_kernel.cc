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
void EqualRawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    int axis,
                    phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA EqualRawKernel";
  dev_ctx.template Alloc<bool>(out);
  sdaa_ops::doCompareTensor(dev_ctx, x, y, CompareType::Equal, out);
}

template <typename T, typename Context>
void EqualKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {
  custom_kernel::EqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void NotEqualRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA NotEqualRawKernel";
  dev_ctx.template Alloc<bool>(out);
  sdaa_ops::doCompareTensor(dev_ctx, x, y, CompareType::NotEqual, out);
}

template <typename T, typename Context>
void NotEqualKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  custom_kernel::NotEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void LessEqualRawKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        int axis,
                        phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA LessEqualRawKernel";
  dev_ctx.template Alloc<bool>(out);
  sdaa_ops::doCompareTensor(dev_ctx, x, y, CompareType::LessEqual, out);
}

template <typename T, typename Context>
void LessEqualKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  custom_kernel::LessEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void GreaterEqualRawKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& y,
                           int axis,
                           phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA GreaterEqualRawKernel";
  dev_ctx.template Alloc<bool>(out);
  sdaa_ops::doCompareTensor(dev_ctx, x, y, CompareType::GreaterEqual, out);
}

template <typename T, typename Context>
void GreaterEqualKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        phi::DenseTensor* out) {
  custom_kernel::GreaterEqualRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void LessThanRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA LessThanRawKernel";
  dev_ctx.template Alloc<bool>(out);
  sdaa_ops::doCompareTensor(dev_ctx, x, y, CompareType::LessThan, out);
}

template <typename T, typename Context>
void LessThanKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  custom_kernel::LessThanRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void GreaterThanRawKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          int axis,
                          phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA GreaterThanRawKernel";
  dev_ctx.template Alloc<bool>(out);
  sdaa_ops::doCompareTensor(dev_ctx, x, y, CompareType::GreaterThan, out);
}

template <typename T, typename Context>
void GreaterThanKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       phi::DenseTensor* out) {
  custom_kernel::GreaterThanRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(equal,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::EqualKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(equal_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::EqualRawKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(less_than,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LessThanKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(less_than_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LessThanRawKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(greater_than,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GreaterThanKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(greater_than_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GreaterThanRawKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(greater_equal,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GreaterEqualKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(greater_equal_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GreaterEqualRawKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(less_equal,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LessEqualKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(less_equal_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LessEqualRawKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(not_equal,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::NotEqualKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(not_equal_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::NotEqualRawKernel,
                          bool,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

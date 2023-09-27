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

#include "kernels/funcs/logic_op.h"

namespace custom_kernel {

template <typename T, typename Context>
void LogicalNotMLUKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         phi::DenseTensor* out) {
  // LogicalNot only has one input x, set y = x also for cnnl computation
  MLULogicOp(dev_ctx, x, x, "not", out);
}

template <typename T, typename Context>
void LogicalAndMLUKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         phi::DenseTensor* out) {
  MLULogicOp(dev_ctx, x, y, "and", out);
}

template <typename T, typename Context>
void LogicalOrMLUKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        phi::DenseTensor* out) {
  MLULogicOp(dev_ctx, x, y, "or", out);
}

template <typename T, typename Context>
void LogicalXorMLUKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         phi::DenseTensor* out) {
  MLULogicOp(dev_ctx, x, y, "xor", out);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(logical_not,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalNotMLUKernel,
                          bool,
                          int,
                          float,
                          phi::dtype::float16,
                          int16_t,
                          int64_t,
                          int8_t,
                          uint8_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(logical_and,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalAndMLUKernel,
                          bool,
                          int,
                          float,
                          phi::dtype::float16,
                          int16_t,
                          int64_t,
                          int8_t,
                          uint8_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(logical_or,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalOrMLUKernel,
                          bool,
                          int,
                          float,
                          phi::dtype::float16,
                          int16_t,
                          int64_t,
                          int8_t,
                          uint8_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

PD_REGISTER_PLUGIN_KERNEL(logical_xor,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::LogicalXorMLUKernel,
                          bool,
                          int,
                          float,
                          phi::dtype::float16,
                          int16_t,
                          int64_t,
                          int8_t,
                          uint8_t) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

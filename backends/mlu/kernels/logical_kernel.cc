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

#include "kernels/funcs/mlu_funcs.h"
#include "kernels/funcs/mlu_baseop.h"

namespace custom_kernel {
template <typename T, typename Context, cnnlLogicOp_t log_method>
void LogicalBaseMLUKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  
  MLUCnnlTensorDesc x_desc(x, CNNL_LAYOUT_ARRAY, ToCnnlDataType(x.dtype()));
  MLUCnnlTensorDesc y_desc(y, CNNL_LAYOUT_ARRAY, ToCnnlDataType(y.dtype()));
  MLUCnnlTensorDesc out_desc(*out);
  
  MLUCnnl::Logic(dev_ctx,
                 log_method,
                 x_desc.get(),
                 GetBasePtr(&x),
                 y_desc.get(),
                 GetBasePtr(&y),
                 out_desc.get(),
                 GetBasePtr(out));
}

template <typename T, typename Context>
void LogicalNotMLUKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         phi::DenseTensor* out) {
  // LogicalNot only has one input x, set y = x also for cnnl computation
  custom_kernel::LogicalBaseMLUKernel<T, Context, CNNL_LOGIC_OP_NOT>(
        dev_ctx,
        x,
        x,
        out);
}

template <typename T, typename Context>
void LogicalAndMLUKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         phi::DenseTensor* out) {
  custom_kernel::LogicalBaseMLUKernel<T, Context, CNNL_LOGIC_OP_AND>(
        dev_ctx,
        x,
        y,
        out);
}

template <typename T, typename Context>
void LogicalOrMLUKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         phi::DenseTensor* out) {
  custom_kernel::LogicalBaseMLUKernel<T, Context, CNNL_LOGIC_OP_OR>(
        dev_ctx,
        x,
        y,
        out);
}

template <typename T, typename Context>
void LogicalXorMLUKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& y,
                         phi::DenseTensor* out) {
  custom_kernel::LogicalBaseMLUKernel<T, Context, CNNL_LOGIC_OP_XOR>(
        dev_ctx,
        x,
        y,
        out);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(logical_not,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::LogicalNotMLUKernel,
                          bool,
                          int,
                          float,
                          phi::dtype::float16,
                          int16_t,
                          int8_t,
                          uint8_t) {}

PD_REGISTER_PLUGIN_KERNEL(logical_and,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::LogicalAndMLUKernel,
                          bool,
                          int,
                          float,
                          phi::dtype::float16,
                          int16_t,
                          int8_t,
                          uint8_t) {}

PD_REGISTER_PLUGIN_KERNEL(logical_or,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::LogicalOrMLUKernel,
                          bool,
                          int,
                          float,
                          phi::dtype::float16,
                          int16_t,
                          int8_t,
                          uint8_t) {}

PD_REGISTER_PLUGIN_KERNEL(logical_xor,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::LogicalXorMLUKernel,
                          bool,int,
                          float,
                          phi::dtype::float16,
                          int16_t,
                          int8_t,
                          uint8_t) {}
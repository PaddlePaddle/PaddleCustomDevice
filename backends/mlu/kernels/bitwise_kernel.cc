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

namespace custom_kernel {

#define DEFINE_BITWISE_KERNEL(op_type)                     \
  template <typename T, typename Context>                  \
  void Bitwise##op_type##Kernel(const Context& dev_ctx,    \
                                const phi::DenseTensor& x, \
                                const phi::DenseTensor& y, \
                                phi::DenseTensor* out) {   \
    dev_ctx.template Alloc<T>(out);                        \
    MLUCnnlTensorDesc x_desc(x);                           \
    MLUCnnlTensorDesc y_desc(y);                           \
    MLUCnnlTensorDesc out_desc(*out);                      \
    cnnlBitComputeOp_t type = CNNL_CYCLE_B##op_type##_OP;  \
    MLUCnnl::BitWise(dev_ctx,                              \
                     type,                                 \
                     x_desc.get(),                         \
                     GetBasePtr(&x),                       \
                     y_desc.get(),                         \
                     GetBasePtr(&y),                       \
                     out_desc.get(),                       \
                     GetBasePtr(out));                     \
  }

DEFINE_BITWISE_KERNEL(AND)
DEFINE_BITWISE_KERNEL(OR)
DEFINE_BITWISE_KERNEL(XOR)
#undef DEFINE_BITWISE_KERNEL

template <typename T, typename Context>
void BitwiseNOTKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  cnnlBitComputeOp_t type = CNNL_BNOT_OP;
  MLUCnnl::BitWise(dev_ctx,
                   type,
                   x_desc.get(),
                   GetBasePtr(&x),
                   x_desc.get(),
                   GetBasePtr(&x),
                   out_desc.get(),
                   GetBasePtr(out));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(bitwise_and,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseANDKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int) {}

PD_REGISTER_PLUGIN_KERNEL(bitwise_or,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseORKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int) {}

PD_REGISTER_PLUGIN_KERNEL(bitwise_xor,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseXORKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int) {}

PD_REGISTER_PLUGIN_KERNEL(bitwise_not,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseNOTKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int) {}

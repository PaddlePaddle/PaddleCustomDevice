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

#include "kernels/common_ops/unary_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_name_list.h"
#include "kernels/funcs/gcu_op_runner.h"
#include "paddle/phi/common/type_traits.h"

namespace custom_kernel {

// #define DEFINE_BITWISE_KERNEL(op_type)                     \
//   template <typename T, typename Context>                  \
//   void Bitwise##op_type##Kernel(const Context& dev_ctx,    \
//                                 const phi::DenseTensor& x, \
//                                 const phi::DenseTensor& y, \
//                                 phi::DenseTensor* out) {   \
//     dev_ctx.template Alloc<T>(out);                        \
//   }

// DEFINE_BITWISE_KERNEL(AND)
// DEFINE_BITWISE_KERNEL(OR)
// DEFINE_BITWISE_KERNEL(XOR)
// #undef DEFINE_BITWISE_KERNEL

template <typename T, typename Context>
void BitwiseNOTKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "bitwise_not", bitwise_not);
    bitwise_not_compute(
        static_cast<const phi::CustomContext&>(dev_ctx), x, out);
    PADDLE_GCU_KERNEL_END("bitwise_not", bitwise_not);
  } else {
    dev_ctx.template Alloc<T>(out);

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "bitwise_not",
              dev_ctx);
  }
}
}  // namespace custom_kernel

// PD_REGISTER_PLUGIN_KERNEL(bitwise_and,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::BitwiseANDKernel,
//                           bool,
//                           uint8_t,
//                           int8_t,
//                           int16_t,
//                           int) {}

// PD_REGISTER_PLUGIN_KERNEL(bitwise_or,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::BitwiseORKernel,
//                           bool,
//                           uint8_t,
//                           int8_t,
//                           int16_t,
//                           int) {}

// PD_REGISTER_PLUGIN_KERNEL(bitwise_xor,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::BitwiseXORKernel,
//                           bool,
//                           uint8_t,
//                           int8_t,
//                           int16_t,
//                           int) {}

PD_REGISTER_PLUGIN_KERNEL(bitwise_not,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseNOTKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int) {}

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

#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"
#include "paddle/phi/common/type_traits.h"

namespace custom_kernel {

template <typename T, typename Context>
void MaskedSelectKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& mask,
                        phi::DenseTensor* out) {
  //   dev_ctx.template Alloc<T>(out);

  //   const phi::DenseTensorMeta meta(phi::CppTypeToDataType<T>::Type(),
  //                                   out->dims());
  //   out->set_meta(meta);

  TensorNameMap input_names;
  input_names["X"] = {"x"};
  input_names["Mask"] = {"mask"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};
  inputs["Mask"] = {const_cast<DenseTensor*>(&mask)};

  TensorNameMap output_names;
  output_names["Y"] = {"out"};

  TensorValueMap outputs;
  outputs["Y"] = {out};

  GcuRunner(
      input_names, inputs, output_names, outputs, {}, "masked_select", dev_ctx);
}

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& mask,
                            const phi::DenseTensor& out_grad,
                            phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  TensorNameMap input_names;
  input_names["X"] = {"x"};
  input_names["Mask"] = {"mask"};
  input_names[GradVarName("Y")] = {"out_grad"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};
  inputs["Mask"] = {const_cast<DenseTensor*>(&mask)};
  inputs[GradVarName("Y")] = {const_cast<DenseTensor*>(&out_grad)};

  TensorNameMap output_names;
  output_names[GradVarName("X")] = {"x_grad"};

  TensorValueMap outputs;
  outputs[GradVarName("X")] = {x_grad};

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            {},
            "masked_select_grad",
            dev_ctx);
}

}  // namespace custom_kernel

// PD_REGISTER_PLUGIN_KERNEL(masked_select,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::MaskedSelectKernel,
//                           phi::dtype::float16,
//                           float,
//                           int,
//                           int64_t) {
//   kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
//   kernel->OutputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
// }

// PD_REGISTER_PLUGIN_KERNEL(masked_select_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::MaskedSelectGradKernel,
//                           phi::dtype::float16,
//                           float,
//                           int,
//                           int64_t) {
//   kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
// }

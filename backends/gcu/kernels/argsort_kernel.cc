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

#include "kernels/funcs/gcu_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"
#include "paddle/phi/backends/custom/custom_context.h"

namespace custom_kernel {

template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   bool descending,
                   phi::DenseTensor* output,
                   phi::DenseTensor* indices) {
  dev_ctx.template Alloc<T>(output);
  dev_ctx.template Alloc<int64_t>(indices);

  TensorNameMap input_names;
  input_names["X"] = {"x"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};

  TensorNameMap output_names;
  output_names["Out"] = {"output"};
  output_names["Indices"] = {"indices"};

  TensorValueMap outputs;
  outputs["Out"] = {output};
  outputs["Indices"] = {indices};

  GcuAttributeMap attrs;
  attrs["axis"] = axis;
  attrs["descending"] = descending;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, "argsort", dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(argsort,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ArgsortKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}

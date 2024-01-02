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
void AccuracyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out,
                       const phi::DenseTensor& indices,
                       const phi::DenseTensor& label,
                       phi::DenseTensor* accuracy,
                       phi::DenseTensor* correct,
                       phi::DenseTensor* total) {
  dev_ctx.template Alloc<float>(accuracy);
  dev_ctx.template Alloc<int>(correct);
  dev_ctx.template Alloc<int>(total);

  TensorNameMap input_names;
  input_names["Out"] = {"out"};
  input_names["Indices"] = {"indices"};
  input_names["Label"] = {"label"};

  TensorValueMap inputs;
  inputs["Out"] = {const_cast<DenseTensor*>(&out)};
  inputs["Indices"] = {const_cast<DenseTensor*>(&indices)};
  inputs["Label"] = {const_cast<DenseTensor*>(&label)};

  TensorNameMap output_names;
  output_names["Accuracy"] = {"accuracy"};
  output_names["Correct"] = {"correct"};
  output_names["Total"] = {"total"};

  TensorValueMap outputs;
  outputs["Accuracy"] = {accuracy};
  outputs["Correct"] = {correct};
  outputs["Total"] = {total};

  GcuRunner(
      input_names, inputs, output_names, outputs, {}, "accuracy", dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(accuracy,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::AccuracyRawKernel,
                          float,
                          phi::dtype::float16,
                          int,
                          int64_t) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
  kernel->InputAt(2).SetDataType(phi::DataType::INT64);
  kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::INT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::INT32);
}

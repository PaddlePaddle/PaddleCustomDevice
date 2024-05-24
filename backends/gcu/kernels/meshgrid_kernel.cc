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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
void MeshgridKernel(const Context& dev_ctx,
                    const std::vector<const phi::DenseTensor*>& ins,
                    std::vector<phi::DenseTensor*> outs) {
  PADDLE_GCU_KERNEL_TRACE("meshgrid");
  PADDLE_ENFORCE_EQ(
      (ins.size() > 1) && (ins.size() < 7),
      true,
      phi::errors::InvalidArgument(
          "Excepted Tensor numbers between 2 and 6, but received %zu.",
          ins.size()));

  PADDLE_ENFORCE_EQ(
      ins.size(),
      outs.size(),
      phi::errors::InvalidArgument(
          "Excepted input size equals to output size, but received %zu vs %zu.",
          ins.size(),
          outs.size()));

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    std::vector<std::string> in_names;
    in_names.reserve(ins.size());
    std::vector<phi::DenseTensor*> in_tensors;
    in_tensors.reserve(ins.size());

    std::vector<std::string> out_names;
    out_names.reserve(outs.size());
    std::vector<phi::DenseTensor*> out_tensors;
    out_tensors.reserve(outs.size());

    for (size_t i = 0; i < ins.size(); ++i) {
      in_names.emplace_back(std::string("x") + std::to_string(i));
      out_names.emplace_back(std::string("out") + std::to_string(i));

      in_tensors.emplace_back(const_cast<DenseTensor*>(ins[i]));
      dev_ctx.template Alloc<T>(outs[i]);
      out_tensors.emplace_back(outs[i]);
    }

    TensorNameMap input_names;
    input_names["X"] = in_names;

    TensorValueMap inputs;
    inputs["X"] = in_tensors;

    TensorNameMap output_names;
    output_names["Out"] = out_names;

    TensorValueMap outputs;
    outputs["Out"] = out_tensors;

    GcuAttributeMap attrs;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "meshgrid", dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(meshgrid,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MeshgridKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

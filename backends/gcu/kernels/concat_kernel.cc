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

#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& ins,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "concat", concat);

    std::vector<phi::DenseTensor> ins_tensor;
    for (auto in : ins) ins_tensor.emplace_back(*in);
    concat(static_cast<const phi::CustomContext&>(dev_ctx),
           ins_tensor,
           axis_scalar.to<int64_t>(),
           *out);

    PADDLE_GCU_KERNEL_END("concat", concat);
  } else {
    TensorNameMap input_names;
    TensorValueMap inputs;
    std::vector<std::string> names;
    names.reserve(ins.size());
    std::vector<phi::DenseTensor*> values;
    values.reserve(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
      names.emplace_back(std::string("x_") + std::to_string(i));
      values.emplace_back(const_cast<DenseTensor*>(ins[i]));
    }
    input_names["X"] = names;
    inputs["X"] = values;

    TensorNameMap output_names;
    TensorValueMap outputs;

    output_names["Out"] = {"out"};
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["axis"] = axis_scalar.to<int>();

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "concat", dev_ctx);
  }
}

template <typename T, typename Context>
void ConcatGradKernel(const Context& dev_ctx,
                      const std::vector<const phi::DenseTensor*>& ins,
                      const phi::DenseTensor& dout,
                      const phi::Scalar& axis_scalar,
                      std::vector<phi::DenseTensor*> outs) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "concat_grad", concat_grad);

    auto axis = axis_scalar.to<int>();
    const int64_t rank = dout.dims().size();
    if (axis < 0) axis += rank;
    std::vector<int64_t> sections;
    for (auto in : ins) sections.push_back(in->dims()[axis]);
    auto splits = split(dev_ctx, dout, axis, 0, sections);
    for (size_t i = 0; i < outs.size(); ++i) {
      if (outs[i] != nullptr) {
        dev_ctx.template Alloc<T>(outs[i]);
        TensorCopy(dev_ctx, splits[i], false, outs[i]);
      }
    }

    PADDLE_GCU_KERNEL_END("concat_grad", concat_grad);
  } else {
    TensorNameMap input_names;
    TensorValueMap inputs;
    {
      std::vector<std::string> names;
      names.reserve(ins.size());
      std::vector<phi::DenseTensor*> values;
      values.reserve(ins.size());
      for (size_t i = 0; i < ins.size(); ++i) {
        names.emplace_back(std::string("x_") + std::to_string(i));
        values.emplace_back(const_cast<DenseTensor*>(ins[i]));
      }
      input_names["X"] = names;
      inputs["X"] = values;
    }

    input_names[GradVarName("Out")] = {"dout"};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    TensorValueMap outputs;
    {
      std::vector<std::string> names;
      names.reserve(outs.size());
      std::vector<phi::DenseTensor*> values;
      values.reserve(outs.size());
      for (size_t i = 0; i < outs.size(); ++i) {
        if ((outs[i] != nullptr) && (outs[i]->numel() != 0UL)) {
          dev_ctx.template Alloc<T>(outs[i]);
          names.emplace_back(
              GradVarName(std::string("x_") + std::to_string(i)));
          values.emplace_back(outs[i]);
        }
      }
      output_names[GradVarName("X")] = names;
      outputs[GradVarName("X")] = values;
    }

    GcuAttributeMap attrs;
    attrs["axis"] = axis_scalar.to<int>();

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "concat_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(concat,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ConcatKernel,
                          bool,
                          phi::dtype::float16,
                          float,
                          int,
                          int64_t,
                          uint8_t) {}

PD_REGISTER_PLUGIN_KERNEL(concat_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ConcatGradKernel,
                          bool,
                          phi::dtype::float16,
                          float,
                          int,
                          int64_t,
                          uint8_t) {}

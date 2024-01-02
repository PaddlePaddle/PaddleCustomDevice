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

namespace custom_kernel {
template <typename Context>
void GetSeed(const Context& dev_ctx,
             const paddle::optional<phi::DenseTensor>& seed_tensor,
             int seed,
             bool fix_seed,
             int* seed_out) {
  if (seed_tensor) {
    phi::DenseTensor cpu_tensor;
    TensorCopy(dev_ctx, seed_tensor.get(), true, &cpu_tensor);
    std::memcpy(seed_out, cpu_tensor.data(), sizeof(int));
  } else if (!fix_seed) {
    // use cpu engine to generate a seed for npu.
    auto offset = 0;
    auto& engine = *dev_ctx.GetGenerator()->GetCPUEngine();
    *seed_out = static_cast<int>(engine());
  } else {
    *seed_out = seed;
  }
}

template <typename T, typename Context>
void DropoutKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const paddle::optional<phi::DenseTensor>& seed_tensor,
                   const phi::Scalar& p,
                   bool is_test,
                   const std::string& mode,
                   int seed,
                   bool fix_seed,
                   phi::DenseTensor* out,
                   phi::DenseTensor* mask) {
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<uint8_t>(mask);

  TensorNameMap input_names;
  input_names["X"] = {"x"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};
  output_names["Mask"] = {"mask"};

  TensorValueMap outputs;
  outputs["Out"] = {out};
  outputs["Mask"] = {mask};

  auto dropout_prob = p.to<float>();
  int seed_data = 0;
  GetSeed<Context>(dev_ctx, seed_tensor, seed, fix_seed, &seed_data);

  GcuAttributeMap attrs;
  attrs["dropout_prob"] = dropout_prob;
  attrs["dropout_implementation"] = mode;
  attrs["seed"] = seed_data;
  attrs["is_test"] = is_test;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, "dropout", dev_ctx);
}

template <typename T, typename Context>
void DropoutGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& mask,
                       const phi::DenseTensor& dout,
                       const phi::Scalar& p,
                       bool is_test,
                       const std::string& mode,
                       phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  TensorNameMap input_names;
  input_names["Mask"] = {"mask"};
  input_names[GradVarName("Out")] = {"dout"};

  TensorValueMap inputs;
  inputs["Mask"] = {const_cast<DenseTensor*>(&mask)};
  inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

  TensorNameMap output_names;
  output_names[GradVarName("X")] = {"dx"};

  TensorValueMap outputs;
  outputs[GradVarName("X")] = {dx};

  auto dropout_prob = p.to<float>();

  GcuAttributeMap attrs;
  attrs["dropout_prob"] = dropout_prob;
  attrs["is_test"] = is_test;
  attrs["dropout_implementation"] = mode;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            "dropout_grad",
            dev_ctx);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(dropout,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DropoutKernel,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UINT8);
}

PD_REGISTER_PLUGIN_KERNEL(dropout_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DropoutGradKernel,
                          float,
                          phi::dtype::float16) {}

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
void Pool2dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& in_x,
                  const phi::IntArray& kernel_size,
                  const std::vector<int>& strides_t,
                  const std::vector<int>& paddings_t,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("pool2d");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    std::vector<int> ksize(kernel_size.GetData().begin(),
                           kernel_size.GetData().end());

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&in_x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["adaptive"] = adaptive;
    attrs["global_pooling"] = global_pooling;
    attrs["padding_algorithm"] = padding_algorithm;
    attrs["pooling_type"] = pooling_type;
    attrs["data_format"] = data_format;
    attrs["ksize"] = ksize;
    attrs["strides"] = strides_t;
    attrs["paddings"] = paddings_t;
    attrs["ceil_mode"] = ceil_mode;
    attrs["exclusive"] = exclusive;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "pool2d", dev_ctx);
  }
}

template <typename T, typename Context>
void Pool2dGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& in_x,
                      const phi::DenseTensor& out,
                      const phi::DenseTensor& out_grad,
                      const phi::IntArray& kernel_size,
                      const std::vector<int>& strides_t,
                      const std::vector<int>& paddings_t,
                      bool ceil_mode,
                      bool exclusive,
                      const std::string& data_format,
                      const std::string& pooling_type,
                      bool global_pooling,
                      bool adaptive,
                      const std::string& padding_algorithm,
                      phi::DenseTensor* in_x_grad) {
  PADDLE_GCU_KERNEL_TRACE("pool2d_grad");
  dev_ctx.template Alloc<T>(in_x_grad);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    std::vector<int> ksize(kernel_size.GetData().begin(),
                           kernel_size.GetData().end());

    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Y"] = {"y"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&in_x)};
    inputs["Y"] = {const_cast<DenseTensor*>(&out)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"in_x_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {in_x_grad};

    GcuAttributeMap attrs;
    attrs["adaptive"] = adaptive;
    attrs["global_pooling"] = global_pooling;
    attrs["padding_algorithm"] = padding_algorithm;
    attrs["pooling_type"] = pooling_type;
    attrs["data_format"] = data_format;
    attrs["ksize"] = ksize;
    attrs["strides"] = strides_t;
    attrs["paddings"] = paddings_t;
    attrs["ceil_mode"] = ceil_mode;
    attrs["exclusive"] = exclusive;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "pool2d_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(pool2d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(pool2d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dGradKernel,
                          float,
                          phi::dtype::float16) {}

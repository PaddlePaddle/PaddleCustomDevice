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

#include "npu_funcs.h"
#include "npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::Scalar& in_scale,
                 float bias,
                 bool bias_after_scale,
                 phi::DenseTensor* out) {
  auto scale = in_scale.to<float>();
  auto stream = dev_ctx.stream();
  float power = 1.0;
  VLOG(4) << "scale:" << scale << ", bias:" << bias
          << " ,bias_after_scale:" << bias_after_scale;
  if (std::isinf(scale)) {
    if (std::signbit(scale)) {
      scale = -std::numeric_limits<float>::max();
    } else {
      scale = std::numeric_limits<float>::max();
    }
  }
  if (!bias_after_scale) {
    bias *= scale;
  }
  dev_ctx.template Alloc<T>(out);

  NPUAttributeMap attrs = {{"power", power}, {"scale", scale}, {"shift", bias}};

  auto op_func = [](const std::vector<phi::DenseTensor>& inputs,
                    const std::vector<phi::DenseTensor>& outputs,
                    const NPUAttributeMap& attrs,
                    const phi::CustomContext& dev_ctx) {
    const auto& muls_runner = NpuOpRunner(
        "Muls", {inputs[0]}, {outputs[0]}, {{"value", attrs.at("scale")}});
    muls_runner.Run(dev_ctx.stream());

    const auto& adds_runner = NpuOpRunner(
        "Adds", {outputs[0]}, {outputs[0]}, {{"value", attrs.at("shift")}});
    adds_runner.Run(dev_ctx.stream());
  };

  if (x.dtype() == phi::DataType::INT32) {
    NpuOpRunner::TypeAdapter({x},
                             {*out},
                             attrs,
                             dev_ctx,
                             op_func,
                             {phi::DataType::INT32},
                             {phi::DataType::INT32});
  } else if (x.dtype() == phi::DataType::INT64) {
    NpuOpRunner::TypeAdapter({x},
                             {*out},
                             attrs,
                             dev_ctx,
                             op_func,
                             {phi::DataType::INT32},
                             {phi::DataType::INT32});
  } else {
    const auto& runner = NpuOpRunner("Power", {x}, {*out}, attrs);
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scale,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::ScaleKernel,
                          phi::dtype::float16,
                          float,
                          int,
                          int64_t) {}

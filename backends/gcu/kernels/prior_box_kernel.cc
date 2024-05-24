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
void PriorBoxKernel(const Context& dev_ctx,
                    const phi::DenseTensor& input,
                    const phi::DenseTensor& image,
                    const std::vector<float>& min_sizes,
                    const std::vector<float>& max_sizes,
                    const std::vector<float>& aspect_ratios,
                    const std::vector<float>& variances,
                    bool flip,
                    bool clip,
                    float step_w,
                    float step_h,
                    float offset,
                    bool min_max_aspect_ratios_order,
                    phi::DenseTensor* out,
                    phi::DenseTensor* var) {
  PADDLE_GCU_KERNEL_TRACE("prior_box");
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(var);

  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    TensorNameMap input_names;
    input_names["Input"] = {"input"};
    input_names["Image"] = {"image"};

    TensorValueMap inputs;
    inputs["Input"] = {const_cast<DenseTensor*>(&input)};
    inputs["Image"] = {const_cast<DenseTensor*>(&image)};

    TensorNameMap output_names;
    output_names["Boxes"] = {"out"};
    output_names["Variances"] = {"var"};

    TensorValueMap outputs;
    outputs["Boxes"] = {out};
    outputs["Variances"] = {var};

    GcuAttributeMap attrs;
    attrs["min_sizes"] = min_sizes;
    attrs["max_sizes"] = max_sizes;
    attrs["aspect_ratios"] = aspect_ratios;
    attrs["variances"] = variances;
    attrs["flip"] = flip;
    attrs["clip"] = clip;
    attrs["step_w"] = step_w;
    attrs["step_h"] = step_h;
    attrs["offset"] = offset;
    attrs["min_max_aspect_ratios_order"] = min_max_aspect_ratios_order;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "prior_box",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(prior_box,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::PriorBoxKernel,
                          float,
                          phi::dtype::float16) {}

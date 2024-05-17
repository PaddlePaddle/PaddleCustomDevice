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
void ReshapeKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const phi::IntArray& shape,
                   DenseTensor* out,
                   DenseTensor* xshape) {
  PADDLE_GCU_KERNEL_TRACE("reshape");
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    dev_ctx.template Alloc<T>(out);
    dev_ctx.template Alloc<T>(xshape);

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};
    output_names["XShape"] = {"xshape"};

    TensorValueMap outputs;
    outputs["Out"] = {out};
    outputs["XShape"] = {xshape};

    std::vector<int> out_shape = GetIntList(shape.GetData());

    GcuAttributeMap attrs;
    attrs["shape"] = out_shape;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "reshape2", dev_ctx);
  }
}

template <typename T, typename Context>
void ReshapeGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& out_grad,
                       phi::DenseTensor* x_grad) {
  PADDLE_GCU_KERNEL_TRACE("reshape_grad");
  dev_ctx.template Alloc<T>(x_grad);

  phi::DenseTensor* tmp_tensor = nullptr;
  if (LaunchAOTKernel()) {
    THROW_AOT_UNIMPLEMENTED();
  } else {  // kernel impl base on JIT
    auto out_shape = phi::vectorize<int64_t>(x_grad->dims());
    std::vector<int64_t> xshape = {0};
    for (auto dim : out_shape) {
      xshape.emplace_back(dim);
    }
    phi::DenseTensor x_shape;
    x_shape.Resize(phi::make_ddim(xshape));
    dev_ctx.template Alloc<T>(&x_shape);

    TensorNameMap input_names;
    input_names[GradVarName("Out")] = {"out_grad"};
    input_names["XShape"] = {"xshape"};

    TensorValueMap inputs;
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};
    inputs["XShape"] = {&x_shape};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"x_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {(tmp_tensor == nullptr ? x_grad : tmp_tensor)};

    GcuAttributeMap attrs;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "reshape2_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

// PD_REGISTER_PLUGIN_KERNEL(reshape,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::ReshapeKernel,
//                           float,
//                           phi::dtype::float16,
//                           double,
//                           int8_t,
//                           int16_t,
//                           int32_t,
//                           int64_t,
//                           uint8_t,
//                           bool) {}

// PD_REGISTER_PLUGIN_KERNEL(reshape_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::ReshapeGradKernel,
//                           float,
//                           phi::dtype::float16,
//                           double,
//                           int8_t,
//                           int16_t,
//                           int32_t,
//                           int64_t,
//                           uint8_t,
//                           bool) {}

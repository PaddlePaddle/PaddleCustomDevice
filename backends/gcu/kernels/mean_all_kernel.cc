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
void MeanAllKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "mean_all", mean_all);
    auto input_shape = phi::vectorize(x.dims());
    bool keepdims = false;
    int64_t dim = static_cast<int64_t>(input_shape.size());
    std::vector<int64_t> axis;
    for (int64_t i = 0; i < dim; i++) {
      axis.emplace_back(i);
    }
    *out = reduce_mean_compute(dev_ctx, x, false, axis);
    PADDLE_GCU_KERNEL_END("mean_all", mean_all);
  } else {
    dev_ctx.template Alloc<T>(out);

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "mean", dev_ctx);
  }
}

template <typename T, typename Context>
void MeanAllGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& grad,
                       phi::DenseTensor* x_grad) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "mean_all_grad", mean_all_grad);
    int64_t dim = x.dims().size();
    std::vector<int64_t> axis;
    for (int64_t i = 0; i < dim; i++) {
      axis.emplace_back(i);
    }
    auto output_size = grad.numel();
    if (output_size == 0) {
      output_size = 1;
    }
    auto input_size = x.numel();
    if (input_size == 0) {
      input_size = 1;
    }
    float reduced_size = static_cast<float>(input_size / output_size);
    float reciprocal = 1.0 / reduced_size;
    auto derivative = full_like(dev_ctx, grad, reciprocal);
    auto tmp_grad = mul_compute(dev_ctx, grad, derivative);
    auto output_rank = grad.dims().size();
    std::vector<int64_t> broadcast_dims;
    int iter = 0;
    for (int64_t i = 0; i < output_rank; ++i) {
      if (i == axis[iter]) {
        ++iter;
      } else {
        broadcast_dims.emplace_back(i);
      }
    }
    *x_grad = broadcast_in_dim(
        dev_ctx, tmp_grad, phi::vectorize(x.dims()), broadcast_dims);
    PADDLE_GCU_KERNEL_END("mean_all_grad", mean_all_grad);
  } else {
    dev_ctx.template Alloc<T>(x_grad);

    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names[GradVarName("Out")] = {"grad"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&grad)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"x_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {x_grad};

    GcuAttributeMap attrs;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "mean_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(mean_all,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(mean_all_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

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

#include "common/common.h"
#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {
template <typename T, typename Context>
void FlattenKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int start_axis,
                   int stop_axis,
                   phi::DenseTensor* out,
                   phi::DenseTensor* xshape) {
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(xshape);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "flatten", flatten);
    const int64_t input_rank = x.dims().size();
    PADDLE_ENFORCE(
        -input_rank <= start_axis && start_axis < input_rank,
        phi::errors::InvalidArgument("the range of start_axis is "
                                     "[-input_rank, "
                                     "input_rank), where got start_axis: %d",
                                     start_axis));
    PADDLE_ENFORCE(
        -input_rank <= stop_axis && stop_axis < input_rank,
        phi::errors::InvalidArgument("the range of stop_axis is "
                                     "[-input_rank, "
                                     "input_rank), where got stop_axis: %d",
                                     stop_axis));
    if (start_axis < 0) {
      start_axis += input_rank;
    }
    if (stop_axis < 0) {
      stop_axis += input_rank;
    }
    PADDLE_ENFORCE(
        start_axis <= stop_axis,
        phi::errors::InvalidArgument("the start_axis must <= stop_axis, "
                                     "where got start_axis: "
                                     "%d, stop_axis: %d",
                                     start_axis,
                                     stop_axis));

    auto input_shape = phi::vectorize(x.dims());

    std::vector<int64_t> result_shape;
    for (int64_t i = 0; i < start_axis; ++i) {
      result_shape.emplace_back(input_shape[i]);
    }

    int64_t outer = 1;
    for (int64_t i = start_axis; i <= stop_axis; ++i) {
      outer *= input_shape[i];
    }
    result_shape.emplace_back(outer);

    for (int64_t i = stop_axis + 1; i < input_rank; ++i) {
      result_shape.emplace_back(input_shape[i]);
    }

    *out = reshape(dev_ctx, x, result_shape);
    PADDLE_GCU_KERNEL_END("flatten", flatten);
  } else {
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

    GcuAttributeMap attrs;
    attrs["start_axis"] = start_axis;
    attrs["stop_axis"] = stop_axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "flatten_contiguous_range",
              dev_ctx);
  }
}

template <typename T, typename Context>
void FlattenGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& xshape,
                       const phi::DenseTensor& out_grad,
                       DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  phi::DenseTensor* tmp_tensor = nullptr;
  if (UseScatterMemory()) {
    if (x_grad->data() == out_grad.data()) {
      tmp_tensor = new phi::DenseTensor();
      tmp_tensor->Resize(x_grad->dims());
      dev_ctx.template Alloc(tmp_tensor, x_grad->dtype());
    }
  }

  TensorNameMap input_names;
  input_names["XShape"] = {"xshape"};
  input_names[GradVarName("Out")] = {"out_grad"};

  TensorValueMap inputs;
  inputs["XShape"] = {const_cast<DenseTensor*>(&xshape)};
  inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

  TensorNameMap output_names;
  output_names[GradVarName("X")] = {"x_grad"};

  TensorValueMap outputs;
  outputs[GradVarName("X")] = {((tmp_tensor == nullptr ? x_grad : tmp_tensor))};

  GcuAttributeMap attrs;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            "flatten_contiguous_range_grad",
            dev_ctx);

  if (tmp_tensor != nullptr) {
    *x_grad = *tmp_tensor;
    delete tmp_tensor;
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(flatten,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FlattenKernel,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          uint8_t,
                          int8_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(flatten_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::FlattenGradKernel,
                          phi::dtype::bfloat16,
                          float,
                          double,
                          uint8_t,
                          int8_t,
                          int,
                          int64_t) {}

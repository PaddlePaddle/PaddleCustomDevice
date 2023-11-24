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
#include "common/utils.h"
#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_name_list.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {
template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsKernel(const Context& dev_ctx,
                                         const phi::DenseTensor& x,
                                         const phi::DenseTensor& label,
                                         bool normalize,
                                         int ignore_index,
                                         phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  if (UseScatterMemory()) {
    /// loss = max(x, 0) − x ∗ Labels + log(1 + exp(−|x|))
    PADDLE_GCU_KERNEL_START(dev_ctx,
                            "sigmoid_cross_entropy_with_logits",
                            sigmoid_cross_entropy_with_logits);

    auto zero_tensor = zeros_like(dev_ctx, x);
    auto one_tensor = ones_like(dev_ctx, x);
    auto ignore_tensor =
        full_like(dev_ctx, x, static_cast<int32_t>(ignore_index));

    auto is_ignore_out = equal_compute(dev_ctx, label, ignore_tensor);

    // max(x, 0)
    auto max_out = maximum_compute(dev_ctx, x, zero_tensor);
    // x * labels
    auto mul_out = mul_compute(dev_ctx, x, label);

    /// log(1 + exp(−|X|))
    // |x|
    auto abs_out = abs_compute(dev_ctx, x);
    // -|x|
    auto sub_out = sub_compute(dev_ctx, zero_tensor, abs_out);
    // exp(-|x|)
    auto exp_out = exp_compute(dev_ctx, sub_out);
    // 1 + exp(-|x|)
    auto add_out = add_compute(dev_ctx, one_tensor, exp_out);
    // log(1 + exp(-|x|))
    auto log_out = log_compute(dev_ctx, add_out);

    // max(x, 0) - x * labels
    auto sub2_out = sub_compute(dev_ctx, max_out, mul_out);

    // (max(X, 0) − X ∗ Labels) + log(1 + exp(−|X|))
    auto add2_out = add_compute(dev_ctx, sub2_out, log_out);

    // Specifies a target value that is ignored and
    // does not contribute to the input gradient.
    auto zero_add2_out = zeros_like(dev_ctx, add2_out);
    auto after_ignore_out =
        select(dev_ctx, is_ignore_out, zero_add2_out, add2_out);

    if (normalize) {
      // get (value != ignore_index) number
      auto no_ignore_one_out =
          select(dev_ctx, is_ignore_out, one_tensor, zero_tensor);
      std::vector<int64_t> contiguous_layout(no_ignore_one_out.dims().size(),
                                             0);
      std::iota(contiguous_layout.begin(), contiguous_layout.end(), 0);
      auto no_ignore_count_out = reduce_sum_compute(
          dev_ctx, no_ignore_one_out, false, contiguous_layout);

      // Make sure the denominator is not zero
      auto min_tensor =
          full_like(dev_ctx, no_ignore_count_out, static_cast<T>(1e-5));
      auto no_ignore_count_nonzero_out =
          maximum_compute(dev_ctx, no_ignore_count_out, min_tensor);

      // out = after_ignore_out / no_ignore_count_nonzero
      div_compute(dev_ctx, after_ignore_out, no_ignore_count_nonzero_out, out);
    } else {
      *out = after_ignore_out;
    }

    PADDLE_GCU_KERNEL_END("sigmoid_cross_entropy_with_logits",
                          sigmoid_cross_entropy_with_logits);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Label"] = {"label"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Label"] = {const_cast<DenseTensor*>(&label)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["normalize"] = normalize;
    attrs["ignore_index"] = ignore_index;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "sigmoid_cross_entropy_with_logits",
              dev_ctx);
  }
}

template <typename T, typename Context>
void SigmoidCrossEntropyWithLogitsGradKernel(const Context& dev_ctx,
                                             const phi::DenseTensor& x,
                                             const phi::DenseTensor& label,
                                             const phi::DenseTensor& dout,
                                             bool normalize,
                                             int ignore_index,
                                             phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx,
                            "sigmoid_cross_entropy_with_logits_grad",
                            sigmoid_cross_entropy_with_logits_grad);

    auto ignore_tensor =
        full_like(dev_ctx, x, static_cast<int32_t>(ignore_index));
    auto is_ignore_out = equal_compute(dev_ctx, label, ignore_tensor);

    // dout * (sigmoid(x) - label)
    auto sigmoid_x_out = sigmoid_compute(dev_ctx, x);
    auto sub_out = sub_compute(dev_ctx, sigmoid_x_out, label);
    auto mul_out = mul_compute(dev_ctx, dout, sub_out);

    // Specifies a target value that is ignored and
    // does not contribute to the input gradient.
    auto zero_mul_out = zeros_like(dev_ctx, mul_out);
    auto after_ignore_out =
        select(dev_ctx, is_ignore_out, zero_mul_out, mul_out);

    if (normalize) {
      // get (value != ignore_index) number
      auto zero_tensor = zeros_like(dev_ctx, after_ignore_out);
      auto one_tensor = ones_like(dev_ctx, after_ignore_out);
      auto no_ignore_one_out =
          select(dev_ctx, is_ignore_out, one_tensor, zero_tensor);
      std::vector<int64_t> contiguous_layout(no_ignore_one_out.dims().size(),
                                             0);
      std::iota(contiguous_layout.begin(), contiguous_layout.end(), 0);
      auto no_ignore_count_out = reduce_sum_compute(
          dev_ctx, no_ignore_one_out, false, contiguous_layout);

      // Make sure the denominator is not zero
      auto min_tensor =
          full_like(dev_ctx, no_ignore_count_out, static_cast<T>(1e-5));
      auto no_ignore_count_nonzero_out =
          maximum_compute(dev_ctx, no_ignore_count_out, min_tensor);

      // out = after_ignore_out / no_ignore_count_nonzero
      div_compute(dev_ctx, after_ignore_out, no_ignore_count_nonzero_out, dx);
    } else {
      *dx = after_ignore_out;
    }

    PADDLE_GCU_KERNEL_END("sigmoid_cross_entropy_with_logits_grad",
                          sigmoid_cross_entropy_with_logits_grad);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Label"] = {"label"};
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Label"] = {const_cast<DenseTensor*>(&label)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&dout)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"dx"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {dx};

    GcuAttributeMap attrs;
    attrs["normalize"] = normalize;
    attrs["ignore_index"] = ignore_index;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "sigmoid_cross_entropy_with_logits_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sigmoid_cross_entropy_with_logits,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SigmoidCrossEntropyWithLogitsKernel,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(
    sigmoid_cross_entropy_with_logits_grad,
    gcu,
    ALL_LAYOUT,
    custom_kernel::SigmoidCrossEntropyWithLogitsGradKernel,
    float,
    double) {}

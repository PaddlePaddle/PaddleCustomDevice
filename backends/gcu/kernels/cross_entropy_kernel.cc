/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "common/common.h"
#include "common/utils.h"
#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_name_list.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void CrossEntropyWithSoftmaxKernel(const Context& dev_ctx,
                                   const phi::DenseTensor& logits,
                                   const phi::DenseTensor& labels,
                                   bool soft_label,
                                   bool use_softmax,
                                   bool numeric_stable_mode,
                                   int ignore_index,
                                   int axis,
                                   phi::DenseTensor* softmax,
                                   phi::DenseTensor* loss) {
  dev_ctx.template Alloc<T>(loss);
  dev_ctx.template Alloc<T>(softmax);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(
        dev_ctx, "cross_entropy_with_softmax", cross_entropy_with_softmax);

    auto logits_shape = logits.dims();
    auto logits_rank = logits_shape.size();
    auto label_shape = labels.dims();

    PADDLE_ENFORCE_EQ(
        logits_shape.size(),
        label_shape.size(),
        phi::errors::InvalidArgument("Labels Rank and Logits Rank mismatch."));

    int64_t axis_ = axis;
    if (axis_ < 0) axis_ += logits_rank;

    auto label_out = labels;
    auto logit_out = EmptyTensor(dev_ctx, softmax->meta());

    if (soft_label) {
      // soft_label
      if (use_softmax) {
        softmax_compute(dev_ctx, logits, axis_, /*with_log*/ false, *softmax);
        softmax_compute(dev_ctx, logits, axis_, /*with_log*/ true, logit_out);
      } else {
        TensorCopy(dev_ctx, logits, false, softmax);
        log_compute(dev_ctx, logits, &logit_out);
      }
    } else {
      // hard_label
      if (use_softmax) {
        if (numeric_stable_mode) {
          auto max_logits = reduce_max_compute(dev_ctx, logits, true, {axis_});
          auto logits_sub = sub_compute(dev_ctx, logits, max_logits);
          softmax_compute(dev_ctx, logits_sub, axis_, false, *softmax);
          softmax_compute(dev_ctx, logits_sub, axis_, true, logit_out);
        } else {
          softmax_compute(dev_ctx, logits, axis_, false, *softmax);
          softmax_compute(dev_ctx, logits, axis_, true, logit_out);
        }
      } else {
        TensorCopy(dev_ctx, logits, false, softmax);
        log_compute(dev_ctx, logits, &logit_out);
      }

      auto labels_int32 = labels;
      if (labels.dtype() == phi::DataType::INT64) {
        labels_int32 = cast(dev_ctx, labels, phi::DataType::INT32);
      }

      label_out = EmptyTensor(dev_ctx, logits.meta());
      one_hot(dev_ctx, labels_int32, axis_, logits_shape[axis_], label_out);
    }

    auto neg_labels = neg_compute(dev_ctx, label_out);
    auto losses = mul_compute(dev_ctx, neg_labels, logit_out);
    reduce_sum_compute(dev_ctx, losses, true, {axis_}, *loss);

    PADDLE_GCU_KERNEL_END("cross_entropy_with_softmax",
                          cross_entropy_with_softmax);
  } else {
    TensorNameMap input_names;
    input_names["Logits"] = {"logits"};
    input_names["Label"] = {"labels"};

    TensorValueMap inputs;
    inputs["Logits"] = {const_cast<DenseTensor*>(&logits)};
    inputs["Label"] = {const_cast<DenseTensor*>(&labels)};

    TensorNameMap output_names;
    output_names["Softmax"] = {"softmax"};
    output_names["Loss"] = {"loss"};

    TensorValueMap outputs;
    outputs["Softmax"] = {softmax};
    outputs["Loss"] = {loss};

    GcuAttributeMap attrs;
    attrs["soft_label"] = soft_label;
    attrs["use_softmax"] = use_softmax;
    attrs["numeric_stable_mode"] = numeric_stable_mode;
    attrs["ignore_index"] = ignore_index;
    attrs["axis"] = axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "softmax_with_cross_entropy",
              dev_ctx);
  }
}

template <typename T, typename Context>
void CrossEntropyWithSoftmaxGradKernel(const Context& dev_ctx,
                                       const phi::DenseTensor& labels,
                                       const phi::DenseTensor& softmax,
                                       const phi::DenseTensor& loss_grad,
                                       bool soft_label,
                                       bool use_softmax,
                                       bool numeric_stable_mode,
                                       int ignore_index,
                                       int axis,
                                       phi::DenseTensor* logits_grad) {
  dev_ctx.template Alloc<T>(logits_grad);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx,
                            "cross_entropy_with_softmax_grad",
                            cross_entropy_with_softmax_grad);

    phi::DenseTensor grad_mid;
    auto label_out = labels;

    // hard_label
    if (!soft_label) {
      auto logits_shape = softmax.dims();
      int64_t axis_ = axis;
      if (axis_ < 0) axis_ += logits_shape.size();

      auto labels_int32 = labels;
      if (labels.dtype() == phi::DataType::INT64) {
        labels_int32 = cast(dev_ctx, labels, phi::DataType::INT32);
      }

      label_out = EmptyTensor(dev_ctx, softmax.meta());
      one_hot(dev_ctx, labels_int32, axis_, logits_shape[axis_], label_out);
    }

    if (use_softmax) {
      grad_mid = sub_compute(dev_ctx, softmax, label_out);
    } else {
      auto neg_labels = neg_compute(dev_ctx, label_out);
      grad_mid = div_compute(dev_ctx, neg_labels, softmax);
    }
    mul_compute(dev_ctx, grad_mid, loss_grad, logits_grad);

    PADDLE_GCU_KERNEL_END("cross_entropy_with_softmax_grad",
                          cross_entropy_with_softmax_grad);
  } else {
    TensorNameMap input_names;
    input_names["Label"] = {"labels"};
    input_names["Softmax"] = {"softmax"};
    input_names[GradVarName("Loss")] = {"loss_grad"};

    TensorValueMap inputs;
    inputs["Label"] = {const_cast<DenseTensor*>(&labels)};
    inputs["Softmax"] = {const_cast<DenseTensor*>(&softmax)};
    inputs[GradVarName("Loss")] = {const_cast<DenseTensor*>(&loss_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("Logits")] = {"logits_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("Logits")] = {logits_grad};

    GcuAttributeMap attrs;
    attrs["soft_label"] = soft_label;
    attrs["use_softmax"] = use_softmax;
    attrs["numeric_stable_mode"] = numeric_stable_mode;
    attrs["ignore_index"] = ignore_index;
    attrs["axis"] = axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "softmax_with_cross_entropy_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxKernel,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(cross_entropy_with_softmax_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::CrossEntropyWithSoftmaxGradKernel,
                          phi::dtype::float16,
                          float) {}

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

#include "graph/graph_executor.h"

namespace custom_graph {

class SoftmaxWithCrossEntropyAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& logits = ctx.Input("Logits");
    auto& labels = ctx.Input("Label");
    auto& softmax = ctx.Output("Softmax");
    auto& loss = ctx.Output("Loss");

    auto soft_label = ctx.Attr<bool>("soft_label");
    // PADDLE_ENFORCE_EQ(soft_label,
    //                   false,
    //                   platform::errors::Unimplemented(
    //                       "soft_label=True is not supported in "
    //                       "the npu kernel of softmax_with_cross_entropy."));

    auto logits_dims = logits.Shape();
    auto loss_dims = loss.Shape();
    int axis = ctx.Attr<int>("axis");
    axis = axis < 0 ? axis + logits_dims.size() : axis;
    int n = std::accumulate(logits_dims.begin(),
                            logits_dims.begin() + axis,
                            1,
                            std::multiplies<int>());
    int d = std::accumulate(logits_dims.begin() + axis,
                            logits_dims.end(),
                            1,
                            std::multiplies<int>());
    // PADDLE_ENFORCE_EQ(
    //     labels.Numel(),
    //     n,
    //     platform::errors::Unimplemented(
    //         "The size of labels should be equal to phi::funcs::SizeToAxis of
    //         " "logits," "but got size of labels is %d and
    //         phi::funcs::SizeToAxis is %d.", labels.Numel(), n));
    std::vector<int64_t> axes;
    for (auto i = axis; i < logits_dims.size(); ++i) {
      axes.push_back(i);
    }

    OpCommand("SoftmaxV2").Input(logits).Output(softmax).Attr("axes", axes);

    Tensor logits_2d, labels_1d, loss_tmp;
    auto& backprop = Tensor::Get(softmax.Name() + "_backprop");

    OpCommand::Reshape(logits, logits_2d, {n, d});
    OpCommand::Reshape(labels, labels_1d, {n});
    OpCommand("SparseSoftmaxCrossEntropyWithLogits")
        .Input(logits_2d)
        .Input(labels_1d)
        .Output(loss_tmp)
        .Output(backprop);
    OpCommand::Reshape(loss_tmp, loss, loss_dims);
  }
};

class SoftmaxWithCrossEntropyGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& softmax = ctx.Input("Softmax");
    auto& loss_grad = ctx.Input(paddle::framework::GradVarName("Loss"));
    auto& logits_grad = ctx.Output(paddle::framework::GradVarName("Logits"));
    auto& backprop = Tensor::Get(softmax.Name() + "_backprop");

    // PADDLE_ENFORCE_NOT_NULL(backprop,
    //                         platform::errors::PreconditionNotMet(
    //                             "backprop should not be null in NPU kernel of
    //                             " "softmax_with_cross_entropy_grad."));

    OpCommand("Mul").Input(loss_grad).Input(backprop).Output(logits_grad);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(softmax_with_cross_entropy,
               custom_graph::SoftmaxWithCrossEntropyAdapter);
REG_OP_ADAPTER(softmax_with_cross_entropy_grad,
               custom_graph::SoftmaxWithCrossEntropyGradAdapter);

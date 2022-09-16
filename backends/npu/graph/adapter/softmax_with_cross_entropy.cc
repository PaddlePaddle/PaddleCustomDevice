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

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto* logits = ctx.Input("Logits");
    auto* labels = ctx.Input("Label");
    auto* softmax = ctx.Output("Softmax");
    auto* loss = ctx.Output("Loss");
    auto* backprop = ctx.Output("Backprop");
    auto soft_label = ctx.Attr<bool>("soft_label");
    // PADDLE_ENFORCE_EQ(soft_label,
    //                   false,
    //                   platform::errors::Unimplemented(
    //                       "soft_label=True is not supported in "
    //                       "the npu kernel of softmax_with_cross_entropy."));

    auto logits_dims = logits->dims();
    auto loss_dims = loss->dims();
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
    //     labels->numel(),
    //     n,
    //     platform::errors::Unimplemented(
    //         "The size of labels should be equal to phi::funcs::SizeToAxis of
    //         " "logits," "but got size of labels is %d and
    //         phi::funcs::SizeToAxis is %d.", labels->numel(), n));
    std::vector<int64_t> axes;
    for (auto i = axis; i < logits_dims.size(); ++i) {
      axes.push_back(i);
    }

    auto softmax_node = ge::op::SoftmaxV2()
                            .set_input_x(graph->GetOp(logits->Name()))
                            .set_attr_axes(axes);
    graph->AddOp(softmax->Name(), softmax_node);

    auto logits_2d_shape =
        graph::funcs::constant({2}, std::vector<int>({n, d}));
    auto logits_2d = ge::op::Reshape()
                         .set_input_x(graph->GetOp(logits->Name()))
                         .set_input_shape(logits_2d_shape);

    auto labels_1d_shape = graph::funcs::constant({1}, std::vector<int>({n}));
    auto labels_1d = ge::op::Reshape()
                         .set_input_x(graph->GetOp(logits->Name()))
                         .set_input_shape(labels_1d_shape);

    auto softmax_cross_entropy_with_logits =
        ge::op::SparseSoftmaxCrossEntropyWithLogits()
            .set_input_features(logits_2d)
            .set_input_labels(labels_1d);

    auto loss_node = ge::op::Variable();
    loss_node.update_output_desc_y(ge::TensorDesc(
        ge::Shape(std::vector<int64_t>(loss_dims.begin(), loss_dims.end())),
        ge::Format::FORMAT_NCHW,
        graph::traits::pd_dtype_to_ge_dtype(loss->dtype())));
    auto assign_op1 = ge::op::Assign().set_input_ref(loss_node).set_input_value(
        softmax_cross_entropy_with_logits, "loss");

    graph->AddOp(loss->Name(), loss_node);

    auto backprop_node = ge::op::Variable();
    backprop_node.update_output_desc_y(
        ge::TensorDesc(ge::Shape(std::vector<int64_t>(n, d)),
                       ge::Format::FORMAT_NCHW,
                       graph::traits::pd_dtype_to_ge_dtype(loss->dtype())));
    auto assign_op2 =
        ge::op::Assign()
            .set_input_ref(backprop_node)
            .set_input_value(softmax_cross_entropy_with_logits, "backprop");

    graph->AddOp(loss->Name() + "_backprop", loss_node);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(softmax_with_cross_entropy,
               custom_graph::SoftmaxWithCrossEntropyAdapter);

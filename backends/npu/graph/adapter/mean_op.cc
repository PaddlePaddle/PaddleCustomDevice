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

class MeanAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto* x = ctx.Input("X");
    auto* out = ctx.Output("Out");
    auto reduce_mean = ge::op::ReduceMeanD()
                           .set_input_x(graph->GetOp(x->Name()))
                           .set_attr_axes(std::vector<int64_t>())
                           .set_attr_keep_dims(false);
    graph->AddOp(out->Name(), reduce_mean);
  }
};

class MeanGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto grad = ctx.Input(paddle::framework::GradVarName("Out"));
    auto dx = ctx.Output(paddle::framework::GradVarName("X"));

    auto dx_dim = dx->dims();
    // auto fillv2 = ge::op::FillV2D()
    //               .set_attr_value(1.0f / static_cast<float>(dx->numel()))
    //               .set_attr_dims(
    //                   std::vector<int64_t>(dx_dim.begin(), dx_dim.end()));
    auto fillv2 = graph::funcs::constant(
        dx_dim,
        std::vector<float>(dx->numel(),
                           1.0f / static_cast<float>(dx->numel())));
    auto reduce_mean_grad = ge::op::Mul().set_input_x1(fillv2).set_input_x2(
        graph->GetOp(grad->Name()));

    graph->AddOp(dx->Name(), reduce_mean_grad);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(mean, custom_graph::MeanAdapter);
REG_OP_ADAPTER(mean_grad, custom_graph::MeanGradAdapter);

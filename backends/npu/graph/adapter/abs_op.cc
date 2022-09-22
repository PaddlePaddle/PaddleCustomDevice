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

class AbsAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto* x = ctx.Input("X");
    auto* out = ctx.Output("Out");
    auto abs = ge::op::Abs().set_input_x(graph->GetOp(x->Name()));
    graph->AddOp(out->Name(), abs);
  }
};

class AbsGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto* x = ctx.Input("X");
    auto* dout = ctx.Input(paddle::framework::GradVarName("Out"));
    auto* dx = ctx.Output(paddle::framework::GradVarName("X"));

    auto abs_grad = ge::op::AbsGrad()
                        .set_input_y(graph->GetOp(x->Name()))
                        .set_input_dy(graph->GetOp(dout->Name()));
    graph->AddOp(dx->Name(), abs_grad);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(abs, custom_graph::AbsAdapter);
REG_OP_ADAPTER(abs_grad, custom_graph::AbsGradAdapter);

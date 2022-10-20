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

class SumAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto out_var = ctx.Output("Out");

    if (out_var->Type() == "lod_tensor") {
      auto x = ctx.MultiInput("X");
      int n = x.size();
      if (n == 1) {
        graph->RecordNode(out_var->Name(), graph->GetOp(x[0]->Name()));
      } else {
        auto add_n = ge::op::AddN().create_dynamic_input_x(n).set_attr_N(n);
        for (auto i = 0; i < n; ++i) {
          add_n = add_n.set_dynamic_input_x(i, graph->GetOp(x[i]->Name()));
        }
        graph->AddOp(out_var->Name(), add_n);
      }
    } else {
      // error
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(sum, custom_graph::SumAdapter);

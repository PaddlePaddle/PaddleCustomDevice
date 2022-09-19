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

class TopKV2Adapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto* input = ctx.Input("X");
    auto* k_tensor = ctx.Input("K");
    auto* out = ctx.Output("Out");
    auto out_dim = out->dims();
    auto* indices = ctx.Output("Indices");  // type: INT64
    auto indices_dim = indices->dims();

    auto k = ctx.Attr<int>("k");
    auto axis = ctx.Attr<int>("axis");
    auto sorted = ctx.Attr<bool>("sorted");
    auto largest = ctx.Attr<bool>("largest");

    if (axis < 0) {
      axis += input->dims().size();
    }

    if (k_tensor != nullptr) {
      auto node = ge::op::TopKV2()
                      .set_input_x(graph->GetOp(input->Name()))
                      .set_input_k(graph->GetOp(k_tensor->Name()))
                      .set_attr_sorted(sorted)
                      .set_attr_dim(axis)
                      .set_attr_largest(largest);

      auto indices_int32 = graph::funcs::get_output_by_name<int32_t>(
          node, indices_dim, "indices");
      auto indices_node = graph::funcs::cast(indices_int32, indices->dtype());
      auto out_node = graph::funcs::get_output_by_name(
          node, out_dim, out->dtype(), "values");

      graph->AddOp(out->Name(), out_node);
      graph->AddOp(indices->Name(), indices_node);
    } else {
      auto k_tensor_node = graph::funcs::constant({1}, std::vector<int>({k}));

      auto node = ge::op::TopKV2()
                      .set_input_x(graph->GetOp(input->Name()))
                      .set_input_k(k_tensor_node)
                      .set_attr_sorted(sorted)
                      .set_attr_dim(axis)
                      .set_attr_largest(largest);

      auto indices_int32 = graph::funcs::get_output_by_name<int32_t>(
          node, indices_dim, "indices");
      auto indices_node = graph::funcs::cast(indices_int32, indices->dtype());
      auto out_node = graph::funcs::get_output_by_name(
          node, out_dim, out->dtype(), "values");

      graph->AddOp(out->Name(), out_node);
      graph->AddOp(indices->Name(), indices_node);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(top_k_v2, custom_graph::TopKV2Adapter);

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

  void run(const Context& ctx) override {
    auto& input = ctx.Input("X");
    auto& out = ctx.Output("Out");
    auto& indices = ctx.Output("Indices");  // type: INT64

    auto out_dim = out.Shape();
    auto indices_dim = indices.Shape();

    auto k = ctx.Attr<int>("k");
    auto axis = ctx.Attr<int>("axis");
    auto sorted = ctx.Attr<bool>("sorted");
    auto largest = ctx.Attr<bool>("largest");

    if (axis < 0) {
      axis += input.Shape().size();
    }
    Tensor k_tensor, indices_int32;
    if (ctx.HasInput("K")) {
      k_tensor = ctx.Input("K");
    } else {
      OpCommand::FillConstant(k_tensor, {1}, std::vector<int>({k}));
    }
    OpCommand("TopKV2")
        .Input(input)
        .Input(k_tensor)
        .Output(indices_int32)
        .Output(out)
        .Attr("sorted", sorted)
        .Attr("dim", axis)
        .Attr("largest", largest);
    if (indices.DType() == paddle::framework::proto::VarType::INT32) {
      indices = indices_int32;
    } else {
      OpCommand::Cast(indices_int32, indices, indices.DType());
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(top_k_v2, custom_graph::TopKV2Adapter);

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

class TransposeAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& out = ctx.Output("Out");
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");

    Tensor perm;
    OpCommand::FillConstant(perm, {axis.size()}, axis);
    OpCommand("Transpose").Input(x).Input(perm).Output(out);
  }
};

class TransposeGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& out_grad = ctx.Input(paddle::framework::GradVarName("Out"));
    auto& x_grad = ctx.Output(paddle::framework::GradVarName("X"));
    std::vector<int> axis = ctx.Attr<std::vector<int>>("axis");
    std::vector<int> reversed_axis(axis);
    for (size_t i = 0; i < axis.size(); i++) {
      reversed_axis[axis[i]] = i;
    }

    Tensor perm;
    OpCommand::FillConstant(perm, {reversed_axis.size()}, reversed_axis);
    OpCommand("Transpose").Input(out_grad).Input(perm).Output(x_grad);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(transpose2, custom_graph::TransposeAdapter);
REG_OP_ADAPTER(transpose2_grad, custom_graph::TransposeGradAdapter);

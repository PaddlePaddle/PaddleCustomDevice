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

class ReluAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& out = ctx.Output("Out");

    OpCommand("Relu").Input(x).Output(out);
  }
};

class ReluGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& out_grad = ctx.Input("Out@GRAD");
    auto& out = ctx.Input("Out");
    auto& x_grad = ctx.Output("X@GRAD");

    OpCommand("ReluGrad").Input(out_grad).Input(out).Output(x_grad);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(relu, custom_graph::ReluAdapter);
REG_OP_ADAPTER(relu_grad, custom_graph::ReluGradAdapter);

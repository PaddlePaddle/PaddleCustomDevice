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

  void run(const Context& ctx) override {
    auto& out = ctx.Output("Out");

    // if (out_var->Type() == "lod_tensor") {
    auto x = ctx.MultiInput("X");
    int n = x.size();
    if (n == 1) {
      out = *x[0];
    } else {
      Tensor prev = *x[0];
      Tensor cur;
      for (auto i = 0; i < x.size(); ++i) {
        cur = *x[i];
        Tensor sum;
        OpCommand("Add").Input(prev).Input(cur).Output(sum);
        prev = sum;
      }
    }
    // } else {
    //   // error
    // }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(sum, custom_graph::SumAdapter);

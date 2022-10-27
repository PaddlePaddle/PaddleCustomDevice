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

class CastAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& out = ctx.Output("Out");
    auto dtype = static_cast<paddle::framework::proto::VarType::Type>(
        ctx.Attr<int>("out_dtype"));

    if (x.DType() == dtype) {
      out = x;
    } else {
      OpCommand::Cast(x, out, dtype);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(cast, custom_graph::CastAdapter);

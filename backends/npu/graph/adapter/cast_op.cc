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

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto* x = ctx.Input("X");
    auto* out = ctx.Output("Out");
    auto dtype = static_cast<paddle::framework::proto::VarType::Type>(
        ctx.Attr<int>("out_dtype"));

    if (x->dtype() == dtype) {
      graph->RecordNode(out->Name(), graph->GetOp(x->Name()));
    } else {
      auto cast = graph::funcs::cast(graph->GetOp(x->Name()), dtype);
      graph->AddOp(out->Name(), cast);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(cast, custom_graph::CastAdapter);

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

class FeedAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto out = ctx.Output("Out");
    auto out_dims = out->dims();
    auto col = ctx.Attr<int>("col");
    std::cout << "feed var " << out->Name()
              << ", dims: " << paddle::framework::ir::to_string(out_dims)
              << std::endl;
    auto ge_op =
        ge::op::Data(ge::AscendString(out->Name().c_str())).set_attr_index(col);
    ge::TensorDesc desc = ge_op.GetOutputDescByName("y");
    desc.SetShape(
        ge::Shape(std::vector<int64_t>(out_dims.begin(), out_dims.end())));
    desc.SetRealDimCnt(desc.GetShape().GetDimNum());
    ge_op.UpdateOutputDesc("y", desc);

    graph->AddOp(out->Name(), ge_op);
    graph->AddFeedInput(out->Name(), ge_op, col);
  }
};

class FetchV2Adapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto x = ctx.Input("X");
    std::cout << "fetch var " << x->Name() << std::endl;
    graph->AddFetchOutput(x->Name(), graph->GetOp(x->Name()));
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(feed, custom_graph::FeedAdapter);
REG_OP_ADAPTER(fetch_v2, custom_graph::FetchV2Adapter);

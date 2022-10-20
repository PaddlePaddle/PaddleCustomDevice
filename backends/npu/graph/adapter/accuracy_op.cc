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

class AccuracyAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto* inference = ctx.Input("Out");
    auto* label = ctx.Input("Label");
    auto* indices = ctx.Input("Indices");

    auto* accuracy = ctx.Output("Accuracy");
    auto* correct = ctx.Output("Correct");
    auto* total = ctx.Output("Total");

    int num_samples = inference->dims()[0];
    if (num_samples == 0) {
      return;
    }

    auto label_same_dtype_with_indices = graph->GetOp(label->Name());
    if (label->dtype() != indices->dtype()) {
      label_same_dtype_with_indices =
          graph::funcs::cast(label_same_dtype_with_indices, indices->dtype());
    }

    auto equal = ge::op::Equal()
                     .set_input_x1(graph->GetOp(indices->Name()))
                     .set_input_x2(label_same_dtype_with_indices);

    auto equal_float = graph::funcs::cast<float>(equal);

    auto reduce_max = ge::op::ReduceMaxD()
                          .set_input_x(equal_float)
                          .set_attr_axes(std::vector<int64_t>({1}))
                          .set_attr_keep_dims(false);

    auto reduce_sum =
        ge::op::ReduceSumD(ge::AscendString(correct->Name().c_str()))
            .set_input_x(reduce_max)
            .set_attr_axes(std::vector<int64_t>({0}))
            .set_attr_keep_dims(false);

    auto total_float = graph::funcs::constant(
        {1},
        std::vector<float>({static_cast<float>(num_samples)}),
        ge::Format::FORMAT_NCHW,
        total->Name());

    auto div = ge::op::Div(ge::AscendString(accuracy->Name().c_str()))
                   .set_input_x1(reduce_sum)
                   .set_input_x2(total_float);

    graph->AddOp(correct->Name(), reduce_sum);
    graph->AddOp(total->Name(), total_float);
    graph->AddInput(total_float);
    graph->AddOp(accuracy->Name(), div);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(accuracy, custom_graph::AccuracyAdapter);

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

class ScaleAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto* x = ctx.Input("X");
    auto* out = ctx.Output("Out");
    auto scale = ctx.Attr<float>("scale");
    auto bias = ctx.Attr<float>("bias");
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");

    float power = 1.0;

    // graph::utils::log() << "[INFO] scale:" << scale << ", bias:" << bias
    //         << " , bias_after_scale:" << bias_after_scale;
    if (ctx.HasInput("ScaleTensor")) {
      graph::utils::log() << "[ERROR] unsupport ScaleTensor.\n";
      exit(-1);
    }

    if (!bias_after_scale) {
      bias *= scale;
    }

    if (std::isinf(scale)) {
      if (std::signbit(scale)) {
        scale = -std::numeric_limits<float>::max();
      } else {
        scale = std::numeric_limits<float>::max();
      }
      auto op = ge::op::Power()
                    .set_input_x(graph->GetOp(x->Name()))
                    .set_attr_power(power)
                    .set_attr_scale(scale)
                    .set_attr_shift(bias);
      graph->AddOp(out->Name(), op);
    } else {
      auto muls = ge::op::Muls()
                      .set_input_x(graph->GetOp(x->Name()))
                      .set_attr_value(scale);
      auto op = ge::op::Adds().set_input_x(muls).set_attr_value(bias);
      graph->AddOp(out->Name(), op);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(scale, custom_graph::ScaleAdapter);

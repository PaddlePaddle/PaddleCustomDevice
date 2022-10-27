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

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& out = ctx.Output("Out");

    auto scale = ctx.Attr<float>("scale");
    auto bias = ctx.Attr<float>("bias");
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");

    if (ctx.HasInput("ScaleTensor")) {
      graph::utils::log() << "[ERROR] unsupport ScaleTensor.\n";
      exit(-1);
    }

    if (!bias_after_scale) {
      bias *= scale;
    }

    if (std::isinf(scale)) {
      float power = 1.0;
      scale = std::signbit(scale) ? -std::numeric_limits<float>::max()
                                  : std::numeric_limits<float>::max();
      OpCommand("Power")
          .Input(x)
          .Output(out)
          .Attr("power", power)
          .Attr("scale", scale)
          .Attr("shift", bias);
    } else {
      if (x.DType() == paddle::framework::proto::VarType::INT64) {
        OpCommandPipe()
            .Op("Cast")
            .Attr("dst_type",
                  static_cast<int32_t>(
                      graph::utils::cpp_type_to_ge_dtype<int>::value()))
            .Op("Muls")
            .Attr("value", scale)
            .Op("Adds")
            .Attr("value", bias)
            .Op("Cast")
            .Attr("dst_type",
                  static_cast<int>(
                      graph::utils::cpp_type_to_ge_dtype<int64_t>::value()))
            .Input(x)
            .Output(out)
            .End();
      } else {
        OpCommandPipe()
            .Op("Muls")
            .Attr("value", scale)
            .Op("Adds")
            .Attr("value", bias)
            .Input(x)
            .Output(out)
            .End();
      }
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(scale, custom_graph::ScaleAdapter);

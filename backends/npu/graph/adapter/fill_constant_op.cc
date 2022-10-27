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

class FillConstantAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& out = ctx.Output("Out");
    auto out_dims = out.Shape();
    auto value = ctx.Attr<float>("value");
    graph::utils::log() << "[INFO] fill constant tensor: " << out.Name()
                        << ", dims: "
                        << paddle::framework::ir::to_string(out.Shape())
                        << std::endl;

    Tensor constant_tensor;
    if (out.DType() == paddle::framework::proto::VarType::INT32) {
      OpCommand::FillConstant(
          constant_tensor,
          out_dims,
          std::vector<int32_t>(out.Numel(), static_cast<int32_t>(value)));
    } else if (out.DType() == paddle::framework::proto::VarType::FP16) {
      OpCommand::FillConstant(
          constant_tensor,
          out_dims,
          std::vector<phi::dtype::float16>(
              out.Numel(), static_cast<phi::dtype::float16>(value)));
    } else if (out.DType() == paddle::framework::proto::VarType::FP32) {
      OpCommand::FillConstant(
          constant_tensor,
          out_dims,
          std::vector<float>(out.Numel(), static_cast<float>(value)));
    } else if (out.DType() == paddle::framework::proto::VarType::FP64) {
      OpCommand::FillConstant(
          constant_tensor,
          out_dims,
          std::vector<double>(out.Numel(), static_cast<double>(value)));
    } else {
      graph::utils::log() << "[ERROR] fill_constant unsupported datatype "
                          << out.DType() << std::endl;
      exit(-1);
    }

    if (out.op() == nullptr) {
      out = constant_tensor;
    } else {
      OpCommand("Assign").Input(out).Input(constant_tensor);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(fill_constant, custom_graph::FillConstantAdapter);

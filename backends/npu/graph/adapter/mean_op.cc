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

class MeanAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& x = ctx.Input("X");
    auto& out = ctx.Output("Out");
    Tensor tmp_x;
    if (x.DType() == paddle::framework::proto::VarType::FP64) {
      OpCommand::Cast<float>(x, tmp_x);
    } else {
      tmp_x = x;
    }
    OpCommand("ReduceMeanD")
        .Input(tmp_x)
        .Output(out)
        .Attr("axes", std::vector<int64_t>())
        .Attr("keep_dims", false);
  }
};

class MeanGradAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto& grad = ctx.Input(paddle::framework::GradVarName("Out"));
    auto& dx = ctx.Output(paddle::framework::GradVarName("X"));

    Tensor dx_numel;
    OpCommand::FillConstant(
        dx_numel,
        dx.Shape(),
        std::vector<float>(dx.Numel(), 1.0f / static_cast<float>(dx.Numel())));
    Tensor dx_numel_same_dtype_with_grad;
    if (grad.DType() != paddle::framework::proto::VarType::FP32) {
      OpCommand::Cast(dx_numel, dx_numel_same_dtype_with_grad, grad.DType());
    } else {
      dx_numel_same_dtype_with_grad = dx_numel;
    }

    OpCommand("Mul")
        .Input(grad)
        .Input(dx_numel_same_dtype_with_grad)
        .Output(dx);
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(mean, custom_graph::MeanAdapter);
REG_OP_ADAPTER(mean_grad, custom_graph::MeanGradAdapter);

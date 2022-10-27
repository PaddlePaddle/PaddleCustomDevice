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

class MergedMomentumAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context& ctx) override {
    auto params = ctx.MultiInput("Param");
    auto grads = ctx.MultiInput("Grad");
    auto velocitys = ctx.MultiInput("Velocity");
    auto lrs = ctx.MultiInput("LearningRate");

    auto params_out = ctx.MultiOutput("ParamOut");
    auto velocitys_out = ctx.MultiOutput("VelocityOut");

    float mu = ctx.Attr<float>("mu");
    auto use_nesterov = ctx.Attr<bool>("use_nesterov");
    auto regularization_methods =
        ctx.Attr<std::vector<std::string>>("regularization_method");
    auto regularization_coeffs =
        ctx.Attr<std::vector<float>>("regularization_coeff");

    size_t n = params.size();

    for (size_t idx = 0; idx < n; ++idx) {
      auto regularization_flag = regularization_methods.size() > 0
                                     ? regularization_methods[idx]
                                     : "none";
      float regularization_coeff = 0.0;
      if (regularization_coeffs.size() != 0) {
        regularization_coeff = regularization_coeffs[idx];
      }

      auto& learning_rate = lrs.size() > 1 ? *lrs[idx] : *lrs[0];

      auto& param = *params[idx];
      auto& param_out = *params_out[idx];
      auto& velocity = *velocitys[idx];
      auto& velocity_out = *velocitys_out[idx];
      auto& grad = *grads[idx];

      Tensor regularized_grad;
      if (regularization_flag == "l2_decay") {
        Tensor param_mul_regularization_coeff;
        OpCommand("Muls")
            .Input(param)
            .Output(param_mul_regularization_coeff)
            .Attr("value", regularization_coeff);
        OpCommand("Add")
            .Input(param_mul_regularization_coeff)
            .Input(grad)
            .Output(regularized_grad);
      } else {
        regularized_grad = grad;
      }

      Tensor mu_tensor;
      if (param_out.DType() == paddle::framework::proto::VarType::FP16) {
        OpCommand::FillConstant(mu_tensor,
                                {1},
                                std::vector<phi::dtype::float16>(
                                    {static_cast<phi::dtype::float16>(mu)}));
      } else if (param_out.DType() == paddle::framework::proto::VarType::FP32) {
        OpCommand::FillConstant(
            mu_tensor, {1}, std::vector<float>({static_cast<float>(mu)}));
      } else if (param_out.DType() == paddle::framework::proto::VarType::FP64) {
        OpCommand::FillConstant(
            mu_tensor, {1}, std::vector<double>({static_cast<double>(mu)}));
      }

      OpCommand("ApplyMomentum")
          .Input(param_out)
          .Input(velocity_out)
          .Input(learning_rate)
          .Input(regularized_grad)
          .Input(mu_tensor)
          .Attr("use_nesterov", use_nesterov);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(merged_momentum, custom_graph::MergedMomentumAdapter);

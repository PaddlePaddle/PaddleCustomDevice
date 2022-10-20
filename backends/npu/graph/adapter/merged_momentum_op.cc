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

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto params = ctx.MultiInput("Param");
    auto params_out = ctx.MultiOutput("ParamOut");
    size_t n = params.size();
    // PADDLE_ENFORCE_EQ(n,
    //                   params_out.size(),
    //                   platform::errors::InvalidArgument(
    //                       "The size of Output(ParamOut) must be equal to "
    //                       "Input(Param), but got the size of Output(ParamOut)
    //                       " "is %d, the size of Input(Param) is %d.",
    //                       params_out.size(),
    //                       n));
    // for (size_t i = 0; i < n; ++i) {
    //   PADDLE_ENFORCE_EQ(params[i],
    //                     params_out[i],
    //                     platform::errors::InvalidArgument(
    //                         "The size of Input(Param) and Output(ParamOut) "
    //                         "must be the same Tensors."));
    // }

    auto grads = ctx.MultiInput("Grad");
    // PADDLE_ENFORCE_EQ(
    //     n,
    //     grads.size(),
    //     platform::errors::InvalidArgument(
    //         "The size of Input(Grad) must be equal to Input(Param), but got "
    //         "the size of Input(Grad) is %d, the size of Input(Param) is %d.",
    //         grads.size(),
    //         n));

    auto velocitys = ctx.MultiInput("Velocity");
    // PADDLE_ENFORCE_EQ(n,
    //                   velocitys.size(),
    //                   platform::errors::InvalidArgument(
    //                       "The size of Input(Velocity) must be equal to "
    //                       "Input(Param), but got the size of Input(Velocity)
    //                       " "is %d, the size of Input(Param) is %d.",
    //                       velocitys.size(),
    //                       n));

    auto velocitys_out = ctx.MultiOutput("VelocityOut");
    // PADDLE_ENFORCE_EQ(
    //     n,
    //     velocitys_out.size(),
    //     platform::errors::InvalidArgument(
    //         "The size of Output(VelocityOut) must be "
    //         "equal to Input(Param), but got the size of Output(VelocityOut)
    //         is "
    //         "%d, the size of Input(Param) is %d.",
    //         velocitys_out.size(),
    //         n));
    // for (size_t i = 0; i < n; ++i) {
    //   PADDLE_ENFORCE_EQ(velocitys[i],
    //                     velocitys_out[i],
    //                     platform::errors::InvalidArgument(
    //                         "Input(Velocity) and Output(VelocityOut) must be
    //                         " "the same Tensors."));
    // }

    float mu = ctx.Attr<float>("mu");
    auto lrs = ctx.MultiInput("LearningRate");
    // if (lrs.size() != 1) {
    //   PADDLE_ENFORCE_EQ(
    //       n,
    //       lrs.size(),
    //       platform::errors::InvalidArgument(
    //           "If the size of Input(LearningRate) is not 1, the size of "
    //           "Input(LearningRate) must be "
    //           "equal to Input(Param), but got the size of Input(LearningRate)
    //           " "is %d, the size of Input(Param) is %d.", lrs.size(), n));
    // }
    auto use_nesterov = ctx.Attr<bool>("use_nesterov");
    auto regularization_methods =
        ctx.Attr<std::vector<std::string>>("regularization_method");
    auto regularization_coeffs =
        ctx.Attr<std::vector<float>>("regularization_coeff");
    // if (regularization_methods.size() != 0) {
    //   PADDLE_ENFORCE_EQ(
    //       n,
    //       regularization_methods.size(),
    //       platform::errors::InvalidArgument(
    //           "The size of Attr(regularization_method) must be equal "
    //           "to Input(Param), but got the size of "
    //           "Attr(regularization_method) is %d, the size of Input(Param) is
    //           "
    //           "%d.",
    //           regularization_methods.size(),
    //           n));
    //   PADDLE_ENFORCE_EQ(
    //       n,
    //       regularization_coeffs.size(),
    //       platform::errors::InvalidArgument(
    //           "The size of Attr(regularization_coeff) must be equal "
    //           "to Input(Param), but got the size of
    //           Attr(regularization_coeff) " "is %d, the size of Input(Param)
    //           is %d.", regularization_coeffs.size(), n));
    // }

    // VLOG(5) << "use_nesterov: " << use_nesterov
    //         << ",  regularization_methods.size(): "
    //         << regularization_methods.size()
    //         << ",  regularization_coeffs.size(): "
    //         << regularization_coeffs.size();

    for (size_t idx = 0; idx < n; ++idx) {
      auto regularization_flag = regularization_methods.size() > 0
                                     ? regularization_methods[idx]
                                     : "none";
      float regularization_coeff = 0.0;
      if (regularization_coeffs.size() != 0) {
        regularization_coeff = regularization_coeffs[idx];
      }

      auto learning_rate = lrs.size() > 1 ? lrs[idx] : lrs[0];
      auto param = params[idx];
      auto param_out = params_out[idx];
      auto velocity = velocitys[idx];
      auto velocity_out = velocitys_out[idx];
      auto grad = grads[idx];

      if (regularization_flag == "l2_decay") {
        auto muls = ge::op::Muls()
                        .set_input_x(graph->GetOp(param->Name()))
                        .set_attr_value(regularization_coeff);
        auto regularized_grad = ge::op::Add().set_input_x1(muls).set_input_x2(
            graph->GetOp(grad->Name()));
        graph->AddOp(grad->Name() + "_regularized_grad", regularized_grad);
      } else {
        graph->RecordNode(grad->Name() + "_regularized_grad",
                          graph->GetOp(grad->Name()));
      }

      ge::Operator mu_tensor;
      if (param_out->dtype() == paddle::framework::proto::VarType::FP16) {
        mu_tensor =
            graph::funcs::constant({1},
                                   std::vector<phi::dtype::float16>(
                                       {static_cast<phi::dtype::float16>(mu)}));
      } else if (param_out->dtype() ==
                 paddle::framework::proto::VarType::FP32) {
        mu_tensor = graph::funcs::constant(
            {1}, std::vector<float>({static_cast<float>(mu)}));
      } else if (param_out->dtype() ==
                 paddle::framework::proto::VarType::FP64) {
        mu_tensor = graph::funcs::constant(
            {1}, std::vector<double>({static_cast<double>(mu)}));
      }
      auto op =
          ge::op::ApplyMomentum()
              .set_input_var(graph->GetOp(param_out->Name()))
              .set_input_accum(graph->GetOp(velocity_out->Name()))
              .set_input_lr(graph->GetOp(learning_rate->Name()))
              .set_input_grad(graph->GetOp(grad->Name() + "_regularized_grad"))
              .set_input_momentum(mu_tensor)
              .set_attr_use_nesterov(use_nesterov);
      graph::funcs::update_input_dtype(op,
                                       {{"var", param_out->dtype()},
                                        {"accum", velocity_out->dtype()},
                                        {"grad", grad->dtype()}});
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(merged_momentum, custom_graph::MergedMomentumAdapter);

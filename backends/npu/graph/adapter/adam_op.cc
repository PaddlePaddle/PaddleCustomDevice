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

class AdamAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode& ctx,
           custom_graph::GEGraph* graph) override {
    auto* param = ctx.Input("Param");
    auto* grad = ctx.Input("Grad");
    auto* mom1 = ctx.Input("Moment1");
    auto* mom2 = ctx.Input("Moment2");
    auto* lr = ctx.Input("LearningRate");

    auto* beta1_pow = ctx.Input("Beta1Pow");
    auto* beta2_pow = ctx.Input("Beta2Pow");

    auto* param_out = ctx.Output("ParamOut");
    auto* mom1_out = ctx.Output("Moment1Out");
    auto* mom2_out = ctx.Output("Moment2Out");
    auto* beta1_pow_out = ctx.Output("Beta1PowOut");
    auto* beta2_pow_out = ctx.Output("Beta2PowOut");

    bool skip_update = false;
    if (ctx.HasInput("SkipUpdate")) {
      auto* skip_update_tensor = ctx.Input("SkipUpdate");
    }
    // skip_update=true, just copy input to output, and TensorCopy will call
    // mutable_data
    if (skip_update) {
      graph::utils::log() << "[ERROR] skip_update=true is not supported.\n";
      exit(1);
      return;
    }

    bool use_global_beta_pow = ctx.Attr<bool>("use_global_beta_pow");
    if (ctx.HasInput("Beta1Tensor")) {
      auto beta1 = ctx.Input("Beta1Tensor");
      graph->RecordNode(ctx.Name() + "_beta1", graph->GetOp(beta1->Name()));
    } else {
      auto beta1 = ctx.Attr<float>("beta1");
      auto beta1_node =
          graph::funcs::constant<float>({1}, std::vector<float>{beta1});
      graph->RecordNode(ctx.Name() + "_beta1", beta1_node);
    }

    if (ctx.HasInput("Beta2Tensor")) {
      auto beta2 = ctx.Input("Beta2Tensor");
      graph->RecordNode(ctx.Name() + "_beta2", graph->GetOp(beta2->Name()));
    } else {
      auto beta2 = ctx.Attr<float>("beta2");
      auto beta2_node =
          graph::funcs::constant<float>({1}, std::vector<float>{beta2});
      graph->RecordNode(ctx.Name() + "_beta2", beta2_node);
    }

    if (ctx.HasInput("EpsilonTensor")) {
      auto epsilon = ctx.Input("EpsilonTensor");
      graph->RecordNode(ctx.Name() + "_epsilon", graph->GetOp(epsilon->Name()));
    } else {
      auto epsilon = ctx.Attr<float>("epsilon");
      auto epsilon_node =
          graph::funcs::constant<float>({1}, std::vector<float>{epsilon});
      graph->RecordNode(ctx.Name() + "_epsilon", epsilon_node);
    }

    auto adam_op = ge::op::ApplyAdamD()
                       .set_input_var(graph->GetOp(param->Name()))
                       .set_input_m(graph->GetOp(mom1->Name()))
                       .set_input_v(graph->GetOp(mom2->Name()))
                       .set_input_beta1_power(graph->GetOp(beta1_pow->Name()))
                       .set_input_beta2_power(graph->GetOp(beta2_pow->Name()))
                       .set_input_lr(graph->GetOp(lr->Name()))
                       .set_input_beta1(graph->GetOp(ctx.Name() + "_beta1"))
                       .set_input_beta2(graph->GetOp(ctx.Name() + "_beta2"))
                       .set_input_epsilon(graph->GetOp(ctx.Name() + "_epsilon"))
                       .set_input_grad(graph->GetOp(grad->Name()))
                       .set_attr_use_locking(false)
                       .set_attr_use_nesterov(false);

    if (param_out->Name() != param->Name()) {
      graph::utils::log() << "[ERROR] param_out != param\n";
      exit(1);
    }
    if (mom1_out->Name() != mom1->Name()) {
      graph::utils::log() << "[ERROR] mom1_out != mom1\n";
      exit(1);
    }
    if (mom2_out->Name() != mom2->Name()) {
      graph::utils::log() << "[ERROR] mom2_out != mom2\n";
      exit(1);
    }

    if (!use_global_beta_pow) {
      if (beta1_pow_out->Name() != beta1_pow->Name() ||
          beta2_pow_out->Name() != beta2_pow->Name()) {
        graph::utils::log()
            << "[ERROR] beta1_pow_out->Name() != beta1_pow->Name() || "
               "beta2_pow_out->Name() != beta2_pow->Name()\n";
        exit(1);
      }
      auto beta1_pow_out_node =
          ge::op::Mul()
              .set_input_x1(graph->GetOp(beta1_pow->Name()))
              .set_input_x2(graph->GetOp(ctx.Name() + "_beta1"));
      auto beta2_pow_out_node =
          ge::op::Mul()
              .set_input_x1(graph->GetOp(beta2_pow->Name()))
              .set_input_x2(graph->GetOp(ctx.Name() + "_beta2"));
      auto assign1 = ge::op::Assign()
                         .set_input_ref(graph->GetOp(beta1_pow->Name()))
                         .set_input_value(beta1_pow_out_node);
      auto assign2 = ge::op::Assign()
                         .set_input_ref(graph->GetOp(beta2_pow->Name()))
                         .set_input_value(beta2_pow_out_node);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(adam, custom_graph::AdamAdapter);

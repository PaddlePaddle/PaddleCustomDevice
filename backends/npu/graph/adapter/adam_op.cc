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

  void run(const Context& ctx) override {
    auto& param = ctx.Input("Param");
    auto& grad = ctx.Input("Grad");
    auto& mom1 = ctx.Input("Moment1");
    auto& mom2 = ctx.Input("Moment2");
    auto& lr = ctx.Input("LearningRate");

    auto& beta1_pow = ctx.Input("Beta1Pow");
    auto& beta2_pow = ctx.Input("Beta2Pow");

    auto& param_out = ctx.Output("ParamOut");
    auto& mom1_out = ctx.Output("Moment1Out");
    auto& mom2_out = ctx.Output("Moment2Out");
    auto& beta1_pow_out = ctx.Output("Beta1PowOut");
    auto& beta2_pow_out = ctx.Output("Beta2PowOut");

    bool skip_update = false;
    bool use_global_beta_pow = ctx.Attr<bool>("use_global_beta_pow");

    if (ctx.HasInput("SkipUpdate")) {
      auto& skip_update_tensor = ctx.Input("SkipUpdate");
    }
    // skip_update=true, just copy input to output, and TensorCopy will call
    // mutable_data
    if (skip_update) {
      graph::utils::log() << "[ERROR] skip_update=true is not supported."
                          << std::endl;
      exit(1);
      return;
    }

    Tensor beta1, beta2, epsilon;
    if (ctx.HasInput("Beta1Tensor")) {
      beta1 = ctx.Input("Beta1Tensor");
    } else {
      OpCommand::FillConstant(
          beta1, {1}, std::vector<float>({ctx.Attr<float>("beta1")}));
    }
    if (ctx.HasInput("Beta2Tensor")) {
      beta2 = ctx.Input("Beta2Tensor");
    } else {
      OpCommand::FillConstant(
          beta2, {1}, std::vector<float>({ctx.Attr<float>("beta2")}));
    }
    if (ctx.HasInput("EpsilonTensor")) {
      epsilon = ctx.Input("EpsilonTensor");
    } else {
      OpCommand::FillConstant(
          epsilon, {1}, std::vector<float>({ctx.Attr<float>("epsilon")}));
    }

    OpCommand("ApplyAdamD")
        .Input(param)
        .Input(mom1)
        .Input(mom2)
        .Input(beta1_pow)
        .Input(beta2_pow)
        .Input(lr)
        .Input(beta1)
        .Input(beta2)
        .Input(epsilon)
        .Input(grad)
        .Attr("use_locking", false)
        .Attr("use_nesterov", false);

    if (param_out.Name() != param.Name()) {
      graph::utils::log() << "[ERROR] param_out != param" << std::endl;
      exit(1);
    }
    if (mom1_out.Name() != mom1.Name()) {
      graph::utils::log() << "[ERROR] mom1_out != mom1" << std::endl;
      exit(1);
    }
    if (mom2_out.Name() != mom2.Name()) {
      graph::utils::log() << "[ERROR] mom2_out != mom2" << std::endl;
      exit(1);
    }

    if (!use_global_beta_pow) {
      if (beta1_pow_out.Name() != beta1_pow.Name() ||
          beta2_pow_out.Name() != beta2_pow.Name()) {
        graph::utils::log()
            << "[ERROR] beta1_pow_out.Name() != beta1_pow.Name() || "
               "beta2_pow_out.Name() != beta2_pow.Name()"
            << std::endl;
        exit(1);
      }

      // OpCommand("Muls").Input(beta1_pow).Output(beta1_pow_out);
      // OpCommand("Muls").Input(beta2_pow).Output(beta2_pow_out);
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(adam, custom_graph::AdamAdapter);

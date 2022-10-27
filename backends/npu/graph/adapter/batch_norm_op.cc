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

class BatchNormOpAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context &ctx) override {
    auto &x = ctx.Input("X");
    auto &scale = ctx.Input("Scale");
    auto &bias = ctx.Input("Bias");
    auto &running_mean = ctx.Input("Mean");
    auto &running_var = ctx.Input("Variance");
    auto &y = ctx.Output("Y");

    float epsilon = ctx.Attr<float>("epsilon");
    float momentum = ctx.Attr<float>("momentum");
    bool is_test = ctx.Attr<bool>("is_test");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    bool trainable_stats = ctx.Attr<bool>("trainable_statistics");
    std::string data_layout = ctx.Attr<std::string>("data_layout");

    auto x_dims = x.Shape();
    bool test_mode = is_test && (!trainable_stats);
    bool training = !test_mode && !use_global_stats;
    graph::utils::log() << "[INFO] batch_norm data_layout: " << data_layout
                        << std::endl;

    if (!training) {
      OpCommand("BNInfer")
          .Input(x, "x")
          .Input(scale)
          .Input(bias)
          .Input(running_mean)
          .Input(running_var)
          .Output(y, "y")
          .Attr("epsilon", epsilon);
    } else {
      auto &mean_out = ctx.Output("MeanOut");
      auto &variance_out = ctx.Output("VarianceOut");
      auto &saved_mean = ctx.Output("SavedMean");
      auto &saved_variance = ctx.Output("SavedVariance");

      // if MomentumTensor is set, use MomentumTensor value, momentum
      // is only used in this training branch
      if (ctx.HasInput("MomentumTensor")) {
        graph::utils::log()
            << "[ERROR] MomentumTensor is not supported." << std::endl;
        exit(1);
      }

      Tensor sum, square_sum, tmp_mean, tmp_variance;
      OpCommand("BNTrainingReduce")
          .Input(x.SetFormat(data_layout), "x")
          .Output(sum.SetFormat("NCHW"), "sum")
          .Output(square_sum.SetFormat("NCHW"), "square_sum")
          .Attr("epsilon", epsilon);

      OpCommand("BNTrainingUpdate")
          .Input(x, "x")
          .Input(sum, "sum")
          .Input(square_sum, "square_sum")
          .Input(scale.SetFormat("NCHW"), "scale")
          .Input(bias.SetFormat("NCHW"), "offset")
          .Input(running_mean.SetFormat("NCHW"), "mean")
          .Input(running_var.SetFormat("NCHW"), "variance")
          .Output(y.SetFormat(data_layout), "y")
          .Output(tmp_mean)
          .Output(tmp_variance)
          .Output(saved_mean)
          .Output(saved_variance)
          .Attr("epsilon", epsilon)
          .Attr("momentum", momentum);
    }
  }
};

class BatchNormGradOpAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const Context &ctx) override {
    auto &x = ctx.Input("X");
    auto &d_y = ctx.Input(paddle::framework::GradVarName("Y"));
    auto &scale = ctx.Input("Scale");
    auto &bias = ctx.Input("Bias");

    std::string data_layout = ctx.Attr<std::string>("data_layout");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    bool is_test = ctx.Attr<bool>("is_test");
    float epsilon = ctx.Attr<float>("epsilon");

    use_global_stats = is_test || use_global_stats;

    if (ctx.HasOutput(paddle::framework::GradVarName("Scale")) &&
        ctx.HasOutput(paddle::framework::GradVarName("Bias"))) {
      auto &d_scale = ctx.Output(paddle::framework::GradVarName("Scale"));
      auto &d_bias = ctx.Output(paddle::framework::GradVarName("Bias"));
      if (use_global_stats) {
        auto &running_mean = ctx.Input("Mean");
        auto &running_variance = ctx.Input("Variance");

        OpCommand("BNTrainingUpdateGrad")
            .Input(d_y.SetFormat(data_layout), "grads")
            .Input(x.SetFormat(data_layout), "x")
            .Input(running_mean)
            .Input(running_variance)
            .Output(d_scale, "diff_scale")
            .Output(d_bias, "diff_offset")
            .Attr("epsilon", epsilon);
      } else {
        auto &saved_mean = ctx.Input("SavedMean");
        // SavedVariance have been reverted in forward operator
        auto &saved_inv_variance = ctx.Input("SavedVariance");

        OpCommand("BNTrainingUpdateGrad")
            .Input(d_y.SetFormat(data_layout), "grads")
            .Input(x.SetFormat(data_layout), "x")
            .Input(saved_mean)
            .Input(saved_inv_variance)
            .Output(d_scale, "diff_scale")
            .Output(d_bias, "diff_offset")
            .Attr("epsilon", epsilon);
      }
    }
    if (ctx.HasOutput(paddle::framework::GradVarName("X"))) {
      auto &d_x = ctx.Output(paddle::framework::GradVarName("X"));
      if (use_global_stats) {
        auto &running_variance = ctx.Input("Variance");

        OpCommand("BNInferGrad")
            .Input(d_y.SetFormat(data_layout), "grads")
            .Input(scale)
            .Input(running_variance)
            .Output(d_x)
            .Attr("epsilon", epsilon);
      } else {
        auto &saved_mean = ctx.Input("SavedMean");
        // SavedVariance have been reverted in forward operator
        auto &saved_inv_variance = ctx.Input("SavedVariance");
        auto &d_scale = ctx.Output(paddle::framework::GradVarName("Scale"));
        auto &d_bias = ctx.Output(paddle::framework::GradVarName("Bias"));
        OpCommand("BNTrainingReduceGrad")
            .Input(d_y.SetFormat(data_layout), "grads")
            .Input(x.SetFormat(data_layout), "x")
            .Input(d_scale)
            .Input(d_bias)
            .Input(scale)
            .Input(saved_mean)
            .Input(saved_inv_variance)
            .Output(d_x.SetFormat(data_layout), "y")
            .Attr("epsilon", epsilon);
      }
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(batch_norm, custom_graph::BatchNormOpAdapter);
REG_OP_ADAPTER(batch_norm_grad, custom_graph::BatchNormGradOpAdapter);

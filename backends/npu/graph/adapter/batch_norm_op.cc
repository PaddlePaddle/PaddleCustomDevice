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

  void run(const paddle::framework::ir::OpNode &ctx,
           custom_graph::GEGraph *graph) override {
    float epsilon = ctx.Attr<float>("epsilon");
    float momentum = ctx.Attr<float>("momentum");
    bool is_test = ctx.Attr<bool>("is_test");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    bool trainable_stats = ctx.Attr<bool>("trainable_statistics");

    bool test_mode = is_test && (!trainable_stats);
    bool training = !test_mode && !use_global_stats;

    std::string data_layout = ctx.Attr<std::string>("data_layout");

    graph::utils::log() << "[INFO] batch_norm data_layout: " << data_layout
                        << std::endl;

    auto *running_mean = ctx.Input("Mean");
    auto *running_var = ctx.Input("Variance");
    auto *scale = ctx.Input("Scale");
    auto *bias = ctx.Input("Bias");
    auto *x = ctx.Input("X");
    auto x_dims = x->dims();
    // PADDLE_ENFORCE_EQ(
    //     (x_dims.size() == 4UL || x_dims.size() == 3UL),
    //     true,
    //     platform::errors::InvalidArgument(
    //         "The input tensor X's dimension must equal to 3 or 4. "
    //         " But got X's shape = [%s], X's dimension = [%d].",
    //         x_dims.to_str(),
    //         x_dims.size()));
    auto *y = ctx.Output("Y");

    if (!training) {
      auto bn_infer = ge::op::BNInfer()
                          .set_input_x(graph->GetOp(x->Name()))
                          .set_input_scale(graph->GetOp(scale->Name()))
                          .set_input_offset(graph->GetOp(bias->Name()))
                          .set_input_mean(graph->GetOp(running_mean->Name()))
                          .set_input_variance(graph->GetOp(running_var->Name()))
                          .set_attr_epsilon(epsilon);
      graph::funcs::update_input_format(bn_infer,
                                        {{"x", data_layout},
                                         {"scale", data_layout},
                                         {"offset", data_layout},
                                         {"mean", data_layout},
                                         {"variance", data_layout}});
      graph::funcs::update_output_format(bn_infer, {{"y", data_layout}});
      graph->AddOp(y->Name(), bn_infer);
    } else {
      auto *mean_out = ctx.Output("MeanOut");
      auto *variance_out = ctx.Output("VarianceOut");
      auto *saved_mean = ctx.Output("SavedMean");
      auto *saved_variance = ctx.Output("SavedVariance");

      // if MomentumTensor is set, use MomentumTensor value, momentum
      // is only used in this training branch
      if (ctx.HasInput("MomentumTensor")) {
        graph::utils::log()
            << "[ERROR] MomentumTensor is not supported." << std::endl;
        exit(1);
      }

      auto bn_training_reduce = ge::op::BNTrainingReduce()
                                    .set_input_x(graph->GetOp(x->Name()))
                                    .SetAttr("epsilon", epsilon);
      graph::funcs::update_input_format(bn_training_reduce,
                                        {{"x", data_layout}});
      graph::funcs::update_output_format(
          bn_training_reduce, {{"sum", "NCHW"}, {"square_sum", "NCHW"}});
      graph::funcs::update_input_dtype(bn_training_reduce, {{"x", x->dtype()}});

      auto bn_training_update =
          ge::op::BNTrainingUpdate()
              .set_input_x(graph->GetOp(x->Name()))
              .set_input_sum(bn_training_reduce, "sum")
              .set_input_square_sum(bn_training_reduce, "square_sum")
              .set_input_scale(graph->GetOp(scale->Name()))
              .set_input_offset(graph->GetOp(bias->Name()))
              .set_input_mean(graph->GetOp(running_mean->Name()))
              .set_input_variance(graph->GetOp(running_var->Name()))
              .set_attr_epsilon(epsilon)
              .set_attr_factor(momentum);
      graph::funcs::update_input_format(bn_training_update,
                                        {{"x", data_layout},
                                         {"sum", "NCHW"},
                                         {"square_sum", "NCHW"},
                                         {"scale", "NCHW"},
                                         {"offset", "NCHW"},
                                         {"mean", "NCHW"},
                                         {"variance", "NCHW"}});
      graph::funcs::update_output_format(bn_training_update,
                                         {{"y", data_layout},
                                          {"mean", "NCHW"},
                                          {"variance", "NCHW"},
                                          {"batch_mean", "NCHW"},
                                          {"batch_variance", "NCHW"}});
      graph::funcs::update_output_dtype(
          bn_training_update,
          {{"y", y->dtype()},
           {"batch_mean", saved_mean->dtype()},
           {"batch_variance", saved_variance->dtype()}});
      auto y_node = graph::funcs::get_output_by_name(
          bn_training_update, y->dims(), y->dtype(), "y");
      auto batch_mean_node =
          graph::funcs::get_output_by_name(bn_training_update,
                                           mean_out->dims(),
                                           mean_out->dtype(),
                                           "batch_mean");
      auto batch_variance_node =
          graph::funcs::get_output_by_name(bn_training_update,
                                           variance_out->dims(),
                                           variance_out->dtype(),
                                           "batch_variance");

      graph->AddOp(y->Name(), y_node);
      graph->AddOp(saved_mean->Name(), batch_mean_node);
      graph->AddOp(saved_variance->Name(), batch_variance_node);
    }
  }
};

class BatchNormGradOpAdapter : public custom_graph::OpAdapter {
 public:
  using OpAdapter::OpAdapter;

  void run(const paddle::framework::ir::OpNode &ctx,
           custom_graph::GEGraph *graph) override {
    auto *x = ctx.Input("X");
    auto *d_y = ctx.Input(paddle::framework::GradVarName("Y"));
    auto *scale = ctx.Input("Scale");
    auto *bias = ctx.Input("Bias");
    auto *saved_mean = ctx.Input("SavedMean");
    // SavedVariance have been reverted in forward operator
    auto *saved_inv_variance = ctx.Input("SavedVariance");
    std::string data_layout = ctx.Attr<std::string>("data_layout");
    bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    bool is_test = ctx.Attr<bool>("is_test");
    float epsilon = ctx.Attr<float>("epsilon");

    auto *d_x = ctx.Output(paddle::framework::GradVarName("X"));
    auto *d_scale = ctx.Output(paddle::framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output(paddle::framework::GradVarName("Bias"));

    use_global_stats = is_test || use_global_stats;

    if (d_scale && d_bias) {
      if (use_global_stats) {
        const auto *running_mean = ctx.Input("Mean");
        const auto *running_variance = ctx.Input("Variance");

        auto bn_training_update_grad =
            ge::op::BNTrainingUpdateGrad()
                .set_input_grads(graph->GetOp(d_y->Name()))
                .set_input_x(graph->GetOp(x->Name()))
                .set_input_batch_mean(graph->GetOp(running_mean->Name()))
                .set_input_batch_variance(
                    graph->GetOp(running_variance->Name()))
                .set_attr_epsilon(epsilon);
        graph::funcs::update_input_dtype(
            bn_training_update_grad,
            {{"grads", d_y->dtype()}, {"x", x->dtype()}});
        graph::funcs::update_output_dtype(bn_training_update_grad,
                                          {{"diff_scale", d_scale->dtype()},
                                           {"diff_offset", d_bias->dtype()}});
        graph::funcs::update_input_format(
            bn_training_update_grad,
            {{"grads", data_layout}, {"x", data_layout}});

        graph->AddOp(d_scale->Name(),
                     graph::funcs::get_output_by_name(bn_training_update_grad,
                                                      d_scale->dims(),
                                                      d_scale->dtype(),
                                                      "diff_scale"));
        graph->AddOp(d_bias->Name(),
                     graph::funcs::get_output_by_name(bn_training_update_grad,
                                                      d_bias->dims(),
                                                      d_bias->dtype(),
                                                      "diff_offset"));
      } else {
        auto bn_training_update_grad =
            ge::op::BNTrainingUpdateGrad()
                .set_input_grads(graph->GetOp(d_y->Name()))
                .set_input_x(graph->GetOp(x->Name()))
                .set_input_batch_mean(graph->GetOp(saved_mean->Name()))
                .set_input_batch_variance(
                    graph->GetOp(saved_inv_variance->Name()))
                .set_attr_epsilon(epsilon);
        graph::funcs::update_input_dtype(
            bn_training_update_grad,
            {{"grads", d_y->dtype()}, {"x", x->dtype()}});
        graph::funcs::update_output_dtype(bn_training_update_grad,
                                          {{"diff_scale", d_scale->dtype()},
                                           {"diff_offset", d_bias->dtype()}});
        graph::funcs::update_input_format(
            bn_training_update_grad,
            {{"grads", data_layout}, {"x", data_layout}});
        graph->AddOp(d_scale->Name(),
                     graph::funcs::get_output_by_name(bn_training_update_grad,
                                                      d_scale->dims(),
                                                      d_scale->dtype(),
                                                      "diff_scale"));
        graph->AddOp(d_bias->Name(),
                     graph::funcs::get_output_by_name(bn_training_update_grad,
                                                      d_bias->dims(),
                                                      d_bias->dtype(),
                                                      "diff_offset"));
      }
    }
    if (d_x) {
      if (use_global_stats) {
        const auto *running_var = ctx.Input("Variance");
        auto bn_infer_grad =
            ge::op::BNInferGrad()
                .set_input_grads(graph->GetOp(d_y->Name()))
                .set_input_scale(graph->GetOp(scale->Name()))
                .set_input_batch_variance(graph->GetOp(running_var->Name()))
                .set_attr_epsilon(epsilon);
        graph::funcs::update_input_format(bn_infer_grad, "grads", data_layout);
        graph::funcs::update_output_format(
            bn_infer_grad, "x_backprop", data_layout);

        graph->AddOp(d_x->Name(), bn_infer_grad);
      } else {
        auto bn_training_update_grad =
            ge::op::BNTrainingReduceGrad()
                .set_input_grads(graph->GetOp(d_y->Name()))
                .set_input_x(graph->GetOp(x->Name()))
                .set_input_diff_scale(graph->GetOp(d_scale->Name()))
                .set_input_diff_offset(graph->GetOp(d_bias->Name()))
                .set_input_scale(graph->GetOp(scale->Name()))
                .set_input_batch_mean(graph->GetOp(saved_mean->Name()))
                .set_input_batch_variance(
                    graph->GetOp(saved_inv_variance->Name()))
                .set_attr_epsilon(epsilon);
        graph::funcs::update_input_dtype(
            bn_training_update_grad,
            {{"grads", d_y->dtype()}, {"x", x->dtype()}});
        graph::funcs::update_output_dtype(bn_training_update_grad,
                                          {{"y", d_x->dtype()}});
        graph::funcs::update_input_format(
            bn_training_update_grad,
            {{"grads", data_layout}, {"x", data_layout}});
        graph::funcs::update_output_format(bn_training_update_grad,
                                           {{"y", data_layout}});
        graph->AddOp(d_x->Name(), bn_training_update_grad);
      }
    }
  }
};

}  // namespace custom_graph

REG_OP_ADAPTER(batch_norm, custom_graph::BatchNormOpAdapter);
REG_OP_ADAPTER(batch_norm_grad, custom_graph::BatchNormGradOpAdapter);

// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>

#include "custom_engine/ir_translator/translator_registry.h"

namespace custom_engine {
static GcuOpPtr TranslateBatchNorm(
    GcuBuilderPtr gcu_builder,
    const pir::Operation* op,
    const std::vector<std::vector<GcuOpPtr>>& gcu_op_inputs) {
  auto input_x = *(gcu_op_inputs[0][0]);
  auto mean = *(gcu_op_inputs[1][0]);
  auto variance = *(gcu_op_inputs[2][0]);
  auto scale = *(gcu_op_inputs[3][0]);
  auto bias = *(gcu_op_inputs[4][0]);

  // Get attributes
  const auto& attributes = op->attributes();

  bool is_test = attributes.at("is_test").dyn_cast<pir::BoolAttribute>().data();
  float momentum =
      attributes.at("momentum").dyn_cast<pir::FloatAttribute>().data();
  float epsilon =
      attributes.at("epsilon").dyn_cast<pir::FloatAttribute>().data();
  std::string data_format =
      attributes.at("data_format").dyn_cast<pir::StrAttribute>().AsString();
  bool use_global_stats =
      attributes.at("use_global_stats").dyn_cast<pir::BoolAttribute>().data();
  bool trainable_statistics = attributes.at("trainable_statistics")
                                  .dyn_cast<pir::BoolAttribute>()
                                  .data();
  //   bool fuse_with_relu =
  //       attributes.at("fuse_with_relu").dyn_cast<pir::BoolAttribute>().data();

  bool test_mode = is_test && (!trainable_statistics);

  auto in_shape = input_x.GetType().GetShape();
  auto in_rank = input_x.GetType().GetShape().size();
  bool is_bn_1d = (in_rank < 4);
  bool is_bn_2d = (in_rank == 4);
  bool is_bn_3d = (in_rank == 5);
  int feature_index = 3;
  int64_t N = 1;
  if (is_bn_1d) {
    PADDLE_ENFORCE_EQ(
        data_format,
        "NCHW",
        phi::errors::InvalidArgument(
            "BatchNorm1D data layout should be NCHW, but current is: %s",
            data_format));
    if (in_rank == 3) {
      // NCL -> NCL1
      input_x =
          builder::Reshape(input_x, {in_shape[0], in_shape[1], in_shape[2], 1});
    } else if (in_rank == 2) {
      // NC -> NC11
      input_x = builder::Reshape(input_x, {in_shape[0], in_shape[1], 1, 1});
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("Unimplement rank 1 or 0 bn"));
    }
    feature_index = 1;
  } else if (is_bn_2d) {
    PADDLE_ENFORCE_EQ(
        data_format == "NCHW" || data_format == "NHWC",
        true,
        phi::errors::InvalidArgument(
            "BatchNorm2D only support NCHW or NHWC, but current is: %s",
            data_format));
    if (data_format == "NCHW") {
      // NCHW -> NHWC
      input_x = builder::Transpose(input_x, {0, 2, 3, 1});
    }
    feature_index = 3;
  } else if (is_bn_3d) {
    if (data_format == "NCHW") {
      // NCDHW -> NDHWC
      input_x = builder::Transpose(input_x, {0, 2, 3, 4, 1});
    }
    feature_index = 4;
  }
  in_shape = input_x.GetType().GetShape();

  if (!test_mode) {
    auto ptype = input_x.GetType().GetPrimitiveType();
    std::vector<builder::PrimitiveType> tuple_dtype(3, ptype);
    int64_t channel_num = in_shape[feature_index];
    std::vector<std::vector<int64_t>> tuple_shape{
        in_shape, {channel_num}, {channel_num}};

    builder::Type batch_normal_outputs_type(tuple_shape, tuple_dtype);
    auto tuple = builder::BatchNormTraining(input_x,
                                            scale,
                                            bias,
                                            epsilon,
                                            feature_index,
                                            batch_normal_outputs_type);
    auto current_mean = builder::GetTupleElement(tuple, 1);
    auto current_variance = builder::GetTupleElement(tuple, 2);

    std::vector<float> v_momentum(channel_num, momentum);
    auto momentum_op = builder::Const(gcu_builder,
                                      static_cast<void*>(v_momentum.data()),
                                      builder::Type({channel_num}, ptype));
    std::vector<float> v_momentum_sum(channel_num, 1 - momentum);
    auto momentum_sub_op =
        builder::Const(gcu_builder,
                       static_cast<void*>(v_momentum_sum.data()),
                       builder::Type({channel_num}, ptype));
    auto running_mean = mean * momentum_op + current_mean * momentum_sub_op;
    auto running_variance =
        variance * momentum_op + current_variance * momentum_sub_op;

    auto output_y = builder::GetTupleElement(tuple, 0);
    if (is_bn_2d) {
      if (data_format == "NCHW") {
        // NHWC -> NCHW
        output_y = builder::Transpose(output_y, {0, 3, 1, 2});
      }
    } else if (is_bn_3d) {
      if (data_format == "NCHW") {
        // NDHWC -> NCDHW
        output_y = builder::Transpose(output_y, {0, 4, 1, 2, 3});
      }
    } else if (is_bn_1d) {
      if (in_rank == 3) {
        output_y =
            builder::Reshape(output_y, {in_shape[0], in_shape[1], in_shape[2]});
      } else {
        output_y = builder::Reshape(output_y, {in_shape[0], in_shape[1]});
      }
    }
    builder::Op reserve_space = builder::Const(
        gcu_builder,
        nullptr,
        builder::Type({0}, variance.GetType().GetPrimitiveType()));

    std::vector<GcuOp> outputs{output_y,
                               running_mean,
                               running_variance,
                               current_mean,
                               current_variance,
                               reserve_space};
    tuple_shape.clear();
    tuple_dtype.clear();
    for (uint i = 0; i < outputs.size(); i++) {
      tuple_shape.push_back(outputs[i].GetType().GetShape());
      tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
    }
    builder::Type outputs_type(tuple_shape, tuple_dtype);
    auto result = builder::Tuple(outputs, outputs_type);
    return std::make_shared<GcuOp>(result);
  } else {
    auto ptype = input_x.GetType().GetPrimitiveType();
    builder::Type batch_normal_outputs_type(in_shape, ptype);
    auto output_y = builder::BatchNormInference(input_x,
                                                scale,
                                                bias,
                                                mean,
                                                variance,
                                                epsilon,
                                                feature_index,
                                                batch_normal_outputs_type);
    if (is_bn_2d) {
      if (data_format == "NCHW") {
        output_y = builder::Transpose(output_y, {0, 3, 1, 2});
      }
    } else if (is_bn_3d) {
      if (data_format == "NCHW") {
        output_y = builder::Transpose(output_y, {0, 4, 1, 2, 3});
      }
    } else if (is_bn_1d) {
      if (in_rank == 3) {
        output_y =
            builder::Reshape(output_y, {in_shape[0], in_shape[1], in_shape[2]});
      } else {
        output_y = builder::Reshape(output_y, {in_shape[0], in_shape[1]});
      }
    }
    auto running_mean = builder::Reshape(mean, mean.GetType());
    auto running_variance = builder::Reshape(variance, variance.GetType());
    builder::Op current_mean =
        builder::Const(gcu_builder,
                       nullptr,
                       builder::Type({0}, mean.GetType().GetPrimitiveType()));
    builder::Op current_variance = builder::Const(
        gcu_builder,
        nullptr,
        builder::Type({0}, variance.GetType().GetPrimitiveType()));
    builder::Op reserve_space = builder::Const(
        gcu_builder,
        nullptr,
        builder::Type({0}, variance.GetType().GetPrimitiveType()));

    std::vector<GcuOp> outputs{output_y,
                               running_mean,
                               running_variance,
                               current_mean,
                               current_variance,
                               reserve_space};
    std::vector<builder::PrimitiveType> tuple_dtype;
    std::vector<std::vector<int64_t>> tuple_shape;
    for (uint i = 0; i < outputs.size(); i++) {
      tuple_shape.push_back(outputs[i].GetType().GetShape());
      tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
    }
    builder::Type outputs_type(tuple_shape, tuple_dtype);
    auto result = builder::Tuple(outputs, outputs_type);
    return std::make_shared<GcuOp>(result);
  }
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_batch_norm, custom_engine::TranslateBatchNorm)
REGISTER_OP_TRANSLATOR(pd_op_batch_norm_, custom_engine::TranslateBatchNorm)

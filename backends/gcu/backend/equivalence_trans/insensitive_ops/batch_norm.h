/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "backend/register/register.h"

namespace backend {
const char *const kBatchNorm = "batch_norm";
const char *const kBatchNormInfer = "batch_norm_infer";
const char *const kBatchNormGrad = "batch_norm_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, BatchNormEquivalenceTrans) {
  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  auto momentum = PADDLE_GET_CONST(float, op->GetAttr("momentum"));
  auto data_layout = PADDLE_GET_CONST(std::string, op->GetAttr("data_layout"));
  auto is_test = PADDLE_GET_CONST(bool, op->GetAttr("is_test"));
  auto trainable_stats =
      PADDLE_GET_CONST(bool, op->GetAttr("trainable_statistics"));
  bool test_mode = is_test && (!trainable_stats);

  GcuOp input_x = *(map_inputs["X"].at(0));
  GcuOp scale = *(map_inputs["Scale"].at(0));
  GcuOp bias = *(map_inputs["Bias"].at(0));
  GcuOp mean = *(map_inputs["Mean"].at(0));
  GcuOp variance = *(map_inputs["Variance"].at(0));

  auto in_shape = input_x.GetType().GetShape();
  auto in_rank = input_x.GetType().GetShape().size();
  bool is_bn_1d = in_rank < 4 ? true : false;
  bool is_bn_2d = in_rank == 4 ? true : false;
  bool is_bn_3d = in_rank == 5 ? true : false;
  int feature_index = 3;
  int64_t N = 1;
  if (is_bn_1d) {
    PADDLE_ENFORCE_EQ(
        data_layout,
        "NCHW",
        phi::errors::InvalidArgument(
            "BatchNorm1D data layout should be NCHW, but current is: %s",
            data_layout));
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
        data_layout == "NCHW" || data_layout == "NHWC",
        true,
        phi::errors::InvalidArgument(
            "BatchNorm2D only support NCHW or NHWC, but current is: %s",
            data_layout));
    if (data_layout == "NCHW") {
      // NCHW -> NHWC
      input_x = builder::Transpose(input_x, {0, 2, 3, 1});
    }
    feature_index = 3;
  } else if (is_bn_3d) {
    if (data_layout == "NCHW") {
      // NCDHW -> NDHWC
      input_x = builder::Transpose(input_x, {0, 2, 3, 4, 1});
    }
    feature_index = 4;
  }
  in_shape = input_x.GetType().GetShape();

  // put ouputs in order of: Y, MeanOut, VarianceOut, SavedMean, SavedVariance,
  auto output_name_map = op->Outputs();
  std::vector<std::string> output_names{
      "Y", "MeanOut", "VarianceOut", "SavedMean", "SavedVariance"};
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  if (!test_mode) {
    auto ptype = input_x.GetType().GetPrimitiveType();
    std::vector<builder::PrimitiveType> tuple_dtype(3, ptype);
    int64_t channel_num = in_shape[feature_index];
    std::vector<std::vector<int64_t>> tuple_shape{
        in_shape, {channel_num}, {channel_num}};

    GcuType batch_normal_outputs_type(tuple_shape, tuple_dtype);
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
                                      static_cast<void *>(v_momentum.data()),
                                      GcuType({channel_num}, ptype));
    std::vector<float> v_momentum_sum(channel_num, 1 - momentum);
    auto momentum_sub_op =
        builder::Const(gcu_builder,
                       static_cast<void *>(v_momentum_sum.data()),
                       GcuType({channel_num}, ptype));
    auto running_mean = mean * momentum_op + current_mean * momentum_sub_op;
    auto running_variance =
        variance * momentum_op + current_variance * momentum_sub_op;

    auto output_y = builder::GetTupleElement(tuple, 0);
    if (is_bn_2d) {
      if (data_layout == "NCHW") {
        // NHWC -> NCHW
        output_y = builder::Transpose(output_y, {0, 3, 1, 2});
      }
    } else if (is_bn_3d) {
      if (data_layout == "NCHW") {
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

    std::vector<GcuOp> outputs{output_y,
                               running_mean,
                               running_variance,
                               current_mean,
                               current_variance};
    for (size_t i = 1; i < output_names.size(); ++i) {
      output_names_attr += ";" + output_name_map[output_names[i]][0];
    }
    tuple_shape.clear();
    tuple_dtype.clear();
    for (uint i = 0; i < outputs.size(); i++) {
      tuple_shape.push_back(outputs[i].GetType().GetShape());
      tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
    }
    GcuType outputs_type(tuple_shape, tuple_dtype);
    auto result = builder::Tuple(outputs, outputs_type);
    result.SetAttribute(kAttrOpOutVarName,
                        builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(result);
  } else {
    auto ptype = input_x.GetType().GetPrimitiveType();
    GcuType batch_normal_outputs_type(in_shape, ptype);
    auto output_y = builder::BatchNormInference(input_x,
                                                scale,
                                                bias,
                                                mean,
                                                variance,
                                                epsilon,
                                                feature_index,
                                                batch_normal_outputs_type);
    if (is_bn_2d) {
      if (data_layout == "NCHW") {
        output_y = builder::Transpose(output_y, {0, 3, 1, 2});
      }
    } else if (is_bn_3d) {
      if (data_layout == "NCHW") {
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
    builder::Op current_mean;
    bool init_saved_mean =
        PADDLE_GET_CONST(bool, op->GetAttr("init_saved_mean"));
    if (init_saved_mean) {
      current_mean = builder::ZerosLike(mean);
    } else {
      current_mean =
          builder::Const(gcu_builder,
                         nullptr,
                         builder::Type({0}, mean.GetType().GetPrimitiveType()));
    }
    builder::Op current_variance;
    bool init_saved_variance =
        PADDLE_GET_CONST(bool, op->GetAttr("init_saved_variance"));
    if (init_saved_variance) {
      current_variance = builder::ZerosLike(variance);
    } else {
      current_variance = builder::Const(
          gcu_builder,
          nullptr,
          builder::Type({0}, variance.GetType().GetPrimitiveType()));
    }

    std::vector<GcuOp> outputs{output_y,
                               running_mean,
                               running_variance,
                               current_mean,
                               current_variance};
    for (size_t i = 1; i < output_names.size(); ++i) {
      output_names_attr += ";" + output_name_map[output_names[i]][0];
    }
    std::vector<builder::PrimitiveType> tuple_dtype;
    std::vector<std::vector<int64_t>> tuple_shape;
    for (uint i = 0; i < outputs.size(); i++) {
      tuple_shape.push_back(outputs[i].GetType().GetShape());
      tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
    }
    GcuType outputs_type(tuple_shape, tuple_dtype);
    auto result = builder::Tuple(outputs, outputs_type);
    result.SetAttribute(kAttrOpOutVarName,
                        builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(result);
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, BatchNormInferEquivalenceTrans) {
  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  auto data_layout = PADDLE_GET_CONST(std::string, op->GetAttr("data_layout"));
  GcuOp input_x = *(map_inputs["X"].at(0));
  GcuOp scale = *(map_inputs["Scale"].at(0));
  GcuOp bias = *(map_inputs["Bias"].at(0));
  GcuOp mean = *(map_inputs["Mean"].at(0));
  GcuOp variance = *(map_inputs["Variance"].at(0));
  auto in_shape = input_x.GetType().GetShape();
  auto in_rank = input_x.GetType().GetShape().size();
  bool is_bn_1d = in_rank < 4 ? true : false;
  bool is_bn_2d = in_rank == 4 ? true : false;
  bool is_bn_3d = in_rank == 5 ? true : false;
  int feature_index = 3;
  int64_t N = 1;
  if (is_bn_1d) {
    PADDLE_ENFORCE_EQ(
        data_layout,
        "NCHW",
        phi::errors::InvalidArgument("BatchNorm1D input data layout "
                                     "must be NCHW, current is: %s",
                                     data_layout));
    if (in_rank == 3) {
      input_x =
          builder::Reshape(input_x, {in_shape[0], in_shape[1], in_shape[2], 1});
    } else if (in_rank == 2) {
      input_x = builder::Reshape(input_x, {in_shape[0], in_shape[1], 1, 1});
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("Unimplement rank 1 or 0 bn"));
    }
    feature_index = 1;
  } else if (is_bn_2d) {
    PADDLE_ENFORCE_EQ(
        data_layout == "NCHW" || data_layout == "NHWC",
        true,
        phi::errors::InvalidArgument(
            "BatchNorm2D only support NCHW or NHWC, but current is: %s",
            data_layout));
    if (data_layout == "NCHW") {
      input_x = builder::Transpose(input_x, {0, 2, 3, 1});
    }
    feature_index = 3;
  } else if (is_bn_3d) {
    if (data_layout == "NCHW") {
      input_x = builder::Transpose(input_x, {0, 2, 3, 4, 1});
    }
    feature_index = 4;
  }
  auto input_shape = input_x.GetType().GetShape();
  auto ptype = input_x.GetType().GetPrimitiveType();
  GcuType batch_normal_outputs_type(input_shape, ptype);
  auto output_y = builder::BatchNormInference(input_x,
                                              scale,
                                              bias,
                                              mean,
                                              variance,
                                              epsilon,
                                              feature_index,
                                              batch_normal_outputs_type);
  if (is_bn_2d) {
    if (data_layout == "NCHW") {
      output_y = builder::Transpose(output_y, {0, 3, 1, 2});
    }
  } else if (is_bn_3d) {
    if (data_layout == "NCHW") {
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
  return std::make_shared<GcuOp>(output_y);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, BatchNormGradEquivalenceTrans) {
  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  auto data_layout = PADDLE_GET_CONST(std::string, op->GetAttr("data_layout"));
  auto output_name_map = op->Outputs();

  GcuOp input_x = *(map_inputs["X"].at(0));
  GcuOp scale = *(map_inputs["Scale"].at(0));
  GcuOp mean = *(map_inputs["SavedMean"].at(0));
  GcuOp variance = *(map_inputs["SavedVariance"].at(0));
  GcuOp y_grad = *(map_inputs["Y@GRAD"].at(0));

  auto in_shape = input_x.GetType().GetShape();
  auto in_rank = input_x.GetType().GetShape().size();
  auto y_grad_shape = y_grad.GetType().GetShape();
  auto y_grad_rank = y_grad_shape.size();
  if (y_grad_rank != in_rank) {
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "The rank of Y@GRAD must be equal to input_x"));
  }
  bool is_bn_1d = in_rank < 4 ? true : false;
  bool is_bn_2d = in_rank == 4 ? true : false;
  bool is_bn_3d = in_rank == 5 ? true : false;

  int feature_index = 3;
  if (is_bn_1d) {
    PADDLE_ENFORCE_EQ(
        data_layout,
        "NCHW",
        phi::errors::InvalidArgument("BatchNormGrad1D 2D input data layout "
                                     "must be NCHW, current is: %s",
                                     data_layout));
    if (in_rank == 3) {
      input_x =
          builder::Reshape(input_x, {in_shape[0], in_shape[1], in_shape[2], 1});
      y_grad = builder::Reshape(
          y_grad, {y_grad_shape[0], y_grad_shape[1], y_grad_shape[2], 1});
    } else if (in_rank == 2) {
      input_x = builder::Reshape(input_x, {in_shape[0], in_shape[1], 1, 1});
      y_grad =
          builder::Reshape(y_grad, {y_grad_shape[0], y_grad_shape[1], 1, 1});
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("Unimplement rank 1 or 0 bn"));
    }
    feature_index = 1;
  } else if (is_bn_2d) {
    if (data_layout == "NCHW") {
      input_x = builder::Transpose(input_x, {0, 2, 3, 1});
      y_grad = builder::Transpose(y_grad, {0, 2, 3, 1});
    }
    feature_index = 3;
  } else if (is_bn_3d) {
    if (data_layout == "NCHW") {
      input_x = builder::Transpose(input_x, {0, 2, 3, 4, 1});
      y_grad = builder::Transpose(y_grad, {0, 2, 3, 4, 1});
    }
    feature_index = 4;
  }
  in_shape = input_x.GetType().GetShape();

  auto out_type = input_x.GetType();
  auto x_trans_shape = input_x.GetType().GetShape();
  y_grad_shape = y_grad.GetType().GetShape();
  if (x_trans_shape[feature_index] != scale.GetType().GetShape()[0]) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Scale size must be equal to input channel"));
  }
  if (x_trans_shape[feature_index] != variance.GetType().GetShape()[0]) {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Variance size must be equal to input channel"));
  }
  if (x_trans_shape[feature_index] != mean.GetType().GetShape()[0]) {
    PADDLE_THROW(
        phi::errors::Unimplemented("Mean size must be equal to input channel"));
  }

  auto ptype = input_x.GetType().GetPrimitiveType();
  std::vector<builder::PrimitiveType> tuple_dtype(3, ptype);
  std::vector<std::vector<int64_t>> tuple_shape{x_trans_shape,
                                                {x_trans_shape[feature_index]},
                                                {x_trans_shape[feature_index]}};
  GcuType batch_normal_outputs_type(tuple_shape, tuple_dtype);

  auto gradop = builder::BatchNormGrad(input_x,
                                       scale,
                                       mean,
                                       variance,
                                       y_grad,
                                       epsilon,
                                       feature_index,
                                       batch_normal_outputs_type);

  std::vector<GcuOp> output;
  std::vector<std::vector<int64_t>> output_shape;
  std::vector<builder::PrimitiveType> output_dtype;
  std::vector<std::string> output_name;
  if (output_name_map.count("X@GRAD") != 0 &&
      output_name_map["X@GRAD"].size() > 0) {
    auto tout = builder::GetTupleElement(gradop, 0);
    if (is_bn_2d) {
      if (data_layout == "NCHW") {
        tout = builder::Transpose(tout, {0, 3, 1, 2});
      }
    } else if (is_bn_3d) {
      if (data_layout == "NCHW") {
        tout = builder::Transpose(tout, {0, 4, 1, 2, 3});
      }
    } else if (is_bn_1d) {
      if (in_rank == 3) {
        tout = builder::Reshape(tout, {in_shape[0], in_shape[1], in_shape[2]});
      } else {
        tout = builder::Reshape(tout, {in_shape[0], in_shape[1]});
      }
    }
    output.push_back(tout);
    output_shape.push_back(tout.GetType().GetShape());
    output_dtype.push_back(ptype);
    output_name.push_back("X@GRAD");
  }
  if (output_name_map.count("Scale@GRAD") != 0 &&
      output_name_map["Scale@GRAD"].size() > 0) {
    auto gte1 = builder::GetTupleElement(gradop, 1);
    output.push_back(gte1);
    output_shape.push_back({x_trans_shape[feature_index]});
    output_dtype.push_back(ptype);
    output_name.push_back("Scale@GRAD");
  }
  if (output_name_map.count("Bias@GRAD") != 0 &&
      output_name_map["Bias@GRAD"].size() > 0) {
    auto gte2 = builder::GetTupleElement(gradop, 2);
    output.push_back(gte2);
    output_shape.push_back({x_trans_shape[feature_index]});
    output_dtype.push_back(ptype);
    output_name.push_back("Bias@GRAD");
  }

  GcuType output_type(output_shape, output_dtype);

  std::string output_names_attr{output_name_map[output_name[0]][0]};
  for (size_t i = 1; i < output_name.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_name[i]][0];
  }
  auto res = builder::Tuple(output, output_type);
  res.SetAttribute(kAttrOpOutVarName,
                   builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(res);
}

EQUIVALENCE_TRANS_FUNC_REG(kBatchNorm, INSENSITIVE, BatchNormEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kBatchNormInfer,
                           INSENSITIVE,
                           BatchNormInferEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kBatchNormGrad,
                           INSENSITIVE,
                           BatchNormGradEquivalenceTrans);

}  // namespace backend

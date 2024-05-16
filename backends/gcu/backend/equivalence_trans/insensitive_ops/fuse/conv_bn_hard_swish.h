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
#include <string>
#include <vector>

#include "backend/register/register.h"

namespace backend {
const char *const kConvBnHardSwish = "conv_bn_hard_swish";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               op,
                               map_inputs,
                               running_mode,
                               ConvBnHardSwishEquivalenceTrans) {
  using backend::EquivalenceTransformer;
  auto x = *(map_inputs["X"].at(0));
  auto filter = *(map_inputs["Filter"].at(0));

  VLOG(10) << "=== Start ConvBnHardSwish trans ===";
  // attrs
  auto data_format = PADDLE_GET_CONST(std::string, op->GetAttr("data_format"));
  auto groups = PADDLE_GET_CONST(int64_t, op->GetAttr("groups"));
  auto padding_algorithm =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_algorithm"));
  auto strides_attr =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
  auto paddings_attr =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("paddings"));
  auto dilations_attr =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dilations"));

  std::vector<int64_t> strides(strides_attr.begin(), strides_attr.end());
  std::vector<int64_t> paddings(paddings_attr.begin(), paddings_attr.end());
  std::vector<int64_t> dilations(dilations_attr.begin(), dilations_attr.end());

  std::vector<builder::Op> ops;
  int64_t group = static_cast<int64_t>(groups);
  // sorted_archetype_names :"Input", "Filter", "Bias", "ResidualData"
  // necessary input
  if (map_inputs.count("X") != 0) {
    VLOG(10) << "inputs size:" << map_inputs["X"].size();
    auto op_ptr = map_inputs["X"].at(0);
    auto input_shape = op_ptr->GetType().GetShape();
    if (data_format == "NCHW") {
      ops.emplace_back(builder::Transpose(*op_ptr, {0, 2, 3, 1}));
    } else {
      ops.emplace_back(*op_ptr);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, phi::errors::NotFound("lack of [Input] gcu op"));
  }
  if (map_inputs.count("Filter") != 0) {
    VLOG(10) << "Filter size:" << map_inputs["Filter"].size();
    auto op_ptr = map_inputs["Filter"].at(0);
    if (running_mode == RunningMode::ADAPTIVE) {
      ops.emplace_back(*op_ptr);
    } else {
      ops.emplace_back(builder::Transpose(*op_ptr, {2, 3, 1, 0}));
    }
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, phi::errors::NotFound("lack of [Filter] gcu op"));
  }

  VLOG(10) << "input op number:" << ops.size();

  if (padding_algorithm == "SAME") {
    auto input_shape = ops[0].GetType().GetShape();
    auto kernel_shape = ops[1].GetType().GetShape();
    int64_t ih = input_shape[1];
    int64_t iw = input_shape[2];
    int64_t kh = kernel_shape[0];
    int64_t kw = kernel_shape[1];
    auto pad_h = get_same_padding_value(ih, kh, strides[0]);
    auto pad_w = get_same_padding_value(iw, kw, strides[1]);
    paddings = {pad_h[0], pad_h[1], pad_w[0], pad_w[1]};
  } else {
    if (paddings.size() == 1) {
      paddings = {paddings[0], paddings[0], paddings[0], paddings[0]};
    } else if (paddings.size() == 2) {
      paddings = {paddings[0], paddings[0], paddings[1], paddings[1]};
    } else if (paddings.size() == 8) {
      if (data_format == "NCHW") {
        paddings = {paddings[4], paddings[5], paddings[6], paddings[7]};
      } else if (data_format == "NHWC") {
        paddings = {paddings[2], paddings[3], paddings[4], paddings[5]};
      }
    }
  }
  auto conv2d = builder::Conv2D(ops,
                                group,
                                "NOTSET",  // auto_pad
                                "NHWC",    // layout
                                strides,
                                paddings,
                                dilations);

  conv2d.SetAttribute("op_type", builder::Attribute("Conv2DInference"));

  if (data_format == "NCHW") {
    conv2d = builder::Transpose(conv2d, {0, 3, 1, 2});
  }

  // bn
  // auto input_x = conv2d;
  auto mean = *(map_inputs["Mean"].at(0));
  auto bias = *(map_inputs["Bias"].at(0));
  auto scale = *(map_inputs["Scale"].at(0));
  auto variance = *(map_inputs["Variance"].at(0));

  auto is_test = PADDLE_GET_CONST(bool, op->GetAttr("is_test"));
  auto data_layout = PADDLE_GET_CONST(std::string, op->GetAttr("data_layout"));
  auto momentum = PADDLE_GET_CONST(float, op->GetAttr("momentum"));
  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  auto trainable_stats =
      PADDLE_GET_CONST(bool, op->GetAttr("trainable_statistics"));
  auto use_global_stats =
      PADDLE_GET_CONST(bool, op->GetAttr("use_global_stats"));
  bool test_mode = is_test && (!trainable_stats);

  auto in_shape = conv2d.GetType().GetShape();
  auto in_rank = conv2d.GetType().GetShape().size();
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
      conv2d =
          builder::Reshape(conv2d, {in_shape[0], in_shape[1], in_shape[2], 1});
    } else if (in_rank == 2) {
      // NC -> NC11
      conv2d = builder::Reshape(conv2d, {in_shape[0], in_shape[1], 1, 1});
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
      conv2d = builder::Transpose(conv2d, {0, 2, 3, 1});
    }
    feature_index = 3;
  } else if (is_bn_3d) {
    if (data_layout == "NCHW") {
      // NCDHW -> NDHWC
      conv2d = builder::Transpose(conv2d, {0, 2, 3, 4, 1});
    }
    feature_index = 4;
  }
  in_shape = conv2d.GetType().GetShape();
  auto ptype = conv2d.GetType().GetPrimitiveType();
  GcuType batch_normal_outputs_type(in_shape, ptype);
  auto output_y = builder::BatchNormInference(conv2d,
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
  // hardswish
  if (in_rank == 4) {
    return std::make_shared<GcuOp>(builder::Transpose(
        builder::HardSwish(builder::Transpose(output_y, {0, 2, 3, 1})),
        {0, 3, 1, 2}));
  } else if (in_rank == 5) {
    return std::make_shared<GcuOp>(builder::Transpose(
        builder::HardSwish(builder::Transpose(output_y, {0, 2, 3, 4, 1})),
        {0, 4, 1, 2, 3}));
  }
  return std::make_shared<GcuOp>(builder::HardSwish(output_y));
}

EQUIVALENCE_TRANS_FUNC_REG(kConvBnHardSwish,
                           INSENSITIVE,
                           ConvBnHardSwishEquivalenceTrans);

}  // namespace backend

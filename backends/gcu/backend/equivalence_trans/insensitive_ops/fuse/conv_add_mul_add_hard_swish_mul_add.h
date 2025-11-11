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

#include "backend/equivalence_trans/insensitive_ops/fuse/utility.h"
#include "backend/register/register.h"

namespace backend {
const char *const kConvAddMulAddHardSwishMulAdd =
    "conv_add_mul_add_hard_swish_mul_add";
const char *const kDepthwiseConvAddMulAddHardSwishMulAdd =
    "depthwise_conv_add_mul_add_hard_swish_mul_add";
const char *const kDepthwiseConvAddMulAdd = "depthwise_conv_add_mul_add";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               op,
                               map_inputs,
                               running_mode,
                               ConvAddMulAddHardSwishMulAddEquivalenceTrans) {
  using backend::EquivalenceTransformer;
  auto x = *(map_inputs["X"].at(0));
  auto filter = *(map_inputs["Filter"].at(0));

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
  auto y1 = *(map_inputs["Y1"].at(0));
  auto x1 = *(map_inputs["X1"].at(0));
  auto y2 = *(map_inputs["Y2"].at(0));
  auto x2 = *(map_inputs["X2"].at(0));
  auto y3 = *(map_inputs["Y3"].at(0));
  auto axis1 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis1")));
  auto axis2 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis2")));
  auto axis3 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis3")));
  auto axis4 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis4")));
  auto axis5 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis5")));

  auto add_out = add_or_mul_op(conv2d, y1, axis1, true, running_mode);
  auto mul_out = add_or_mul_op(x1, add_out, axis2, false, running_mode);
  auto add2_out = add_or_mul_op(mul_out, y2, axis3, true, running_mode);
  auto hardswish = add2_out;

  auto in_rank = conv2d.GetType().GetShape().size();
  if (in_rank == 4) {
    hardswish = builder::Transpose(
        builder::HardSwish(builder::Transpose(add2_out, {0, 2, 3, 1})),
        {0, 3, 1, 2});
  } else if (in_rank == 5) {
    hardswish = builder::Transpose(
        builder::HardSwish(builder::Transpose(add2_out, {0, 2, 3, 4, 1})),
        {0, 4, 1, 2, 3});
  } else {
    hardswish = builder::HardSwish(add2_out);
  }
  auto mul2_out = add_or_mul_op(x2, hardswish, axis4, false, running_mode);
  auto add3_out = add_or_mul_op(mul2_out, y3, axis5, true, running_mode);
  return std::make_shared<GcuOp>(add3_out);
}

EQUIVALENCE_TRANS_FUNC_REG(kConvAddMulAddHardSwishMulAdd,
                           INSENSITIVE,
                           ConvAddMulAddHardSwishMulAddEquivalenceTrans);

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder,
    op,
    map_inputs,
    running_mode,
    DepthwiseConvAddMulAddHardSwishMulAddEquivalenceTrans) {
  using backend::EquivalenceTransformer;
  auto x = *(map_inputs["X"].at(0));
  auto filter = *(map_inputs["Filter"].at(0));

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
  auto y1 = *(map_inputs["Y1"].at(0));
  auto x1 = *(map_inputs["X1"].at(0));
  auto y2 = *(map_inputs["Y2"].at(0));
  auto x2 = *(map_inputs["X2"].at(0));
  auto y3 = *(map_inputs["Y3"].at(0));
  auto axis1 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis1")));
  auto axis2 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis2")));
  auto axis3 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis3")));
  auto axis4 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis4")));
  auto axis5 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis5")));
  auto add_out = add_or_mul_op(conv2d, y1, axis1, true, running_mode);

  auto mul_out = add_or_mul_op(x1, add_out, axis2, false, running_mode);
  auto add2_out = add_or_mul_op(mul_out, y2, axis3, true, running_mode);
  auto hardswish = add2_out;

  auto in_rank = conv2d.GetType().GetShape().size();
  if (in_rank == 4) {
    hardswish = builder::Transpose(
        builder::HardSwish(builder::Transpose(add2_out, {0, 2, 3, 1})),
        {0, 3, 1, 2});
  } else if (in_rank == 5) {
    hardswish = builder::Transpose(
        builder::HardSwish(builder::Transpose(add2_out, {0, 2, 3, 4, 1})),
        {0, 4, 1, 2, 3});
  } else {
    hardswish = builder::HardSwish(add2_out);
  }
  auto mul2_out = add_or_mul_op(x2, hardswish, axis4, false, running_mode);
  auto add3_out = add_or_mul_op(mul2_out, y3, axis5, true, running_mode);
  return std::make_shared<GcuOp>(add3_out);
}

EQUIVALENCE_TRANS_FUNC_REG(
    kDepthwiseConvAddMulAddHardSwishMulAdd,
    INSENSITIVE,
    DepthwiseConvAddMulAddHardSwishMulAddEquivalenceTrans);

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               op,
                               map_inputs,
                               running_mode,
                               DepthwiseConvAddMulAddEquivalenceTrans) {
  using backend::EquivalenceTransformer;
  auto x = *(map_inputs["X"].at(0));
  auto filter = *(map_inputs["Filter"].at(0));

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

  auto y1 = *(map_inputs["Y1"].at(0));
  auto x1 = *(map_inputs["X1"].at(0));
  auto y2 = *(map_inputs["Y2"].at(0));

  // add -> mul -> add
  auto axis1 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis1")));
  auto axis2 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis2")));
  auto axis3 =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis3")));
  auto add_out = add_or_mul_op(conv2d, y1, axis1, true, running_mode);

  auto mul_out = add_or_mul_op(x1, add_out, axis2, false, running_mode);
  auto add2_out = add_or_mul_op(mul_out, y2, axis3, true, running_mode);
  return std::make_shared<GcuOp>(add2_out);
}

EQUIVALENCE_TRANS_FUNC_REG(kDepthwiseConvAddMulAdd,
                           INSENSITIVE,
                           DepthwiseConvAddMulAddEquivalenceTrans);

}  // namespace backend

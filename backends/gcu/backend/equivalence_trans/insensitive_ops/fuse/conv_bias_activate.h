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
const char *const kConvBias = "conv_bias";
const char *const kConvBiasRelu = "conv_bias_relu";
const char *const kConvBiasHardSigmoid = "conv_bias_hard_sigmoid";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, ConvBiasEquivalenceTrans) {
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

  auto y = *(map_inputs["Y"].at(0));
  auto axis = static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  auto add_out = add_or_mul_op(conv2d, y, axis, true, running_mode);
  return std::make_shared<GcuOp>(add_out);
}

EQUIVALENCE_TRANS_FUNC_REG(kConvBias, INSENSITIVE, ConvBiasEquivalenceTrans);

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, ConvBiasReluEquivalenceTrans) {
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

  auto y = *(map_inputs["Y"].at(0));
  auto axis = static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  auto add_out = add_or_mul_op(conv2d, y, axis, true, running_mode);

  auto rank = add_out.GetType().GetShape().size();
  if (running_mode == RunningMode::ADAPTIVE) {
    if (rank == 4) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::Relu(builder::Transpose(add_out, {0, 2, 3, 1})),
          {0, 3, 1, 2}));
    } else if (rank == 5) {
      return std::make_shared<GcuOp>(builder::Transpose(
          builder::Relu(builder::Transpose(add_out, {0, 2, 3, 4, 1})),
          {0, 4, 1, 2, 3}));
    }
  }
  return std::make_shared<GcuOp>(builder::Relu(add_out));
}

EQUIVALENCE_TRANS_FUNC_REG(kConvBiasRelu,
                           INSENSITIVE,
                           ConvBiasReluEquivalenceTrans);

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               op,
                               map_inputs,
                               running_mode,
                               ConvBiasHardSigmoidEquivalenceTrans) {
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

  auto y = *(map_inputs["Y"].at(0));
  auto axis = static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  auto slope = PADDLE_GET_CONST(float, op->GetAttr("slope"));
  auto offset = PADDLE_GET_CONST(float, op->GetAttr("offset"));

  auto add_out = add_or_mul_op(conv2d, y, axis, true, running_mode);

  return std::make_shared<GcuOp>(builder::HardSigmoid(add_out, slope, offset));
}

EQUIVALENCE_TRANS_FUNC_REG(kConvBiasHardSigmoid,
                           INSENSITIVE,
                           ConvBiasHardSigmoidEquivalenceTrans);
}  // namespace backend

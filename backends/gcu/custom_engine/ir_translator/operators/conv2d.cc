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
namespace {
std::vector<int64_t> GetPaddingWithAlgorithmSame(int64_t dim,
                                                 int64_t ksize,
                                                 int64_t stride) {
  int64_t pad_along_dim = 0;
  if (dim % stride == 0) {
    pad_along_dim = std::max(ksize - stride, static_cast<int64_t>(0));
  } else {
    pad_along_dim = std::max(ksize - (dim % stride), static_cast<int64_t>(0));
  }
  int64_t pad_low = pad_along_dim / 2;
  int64_t pad_high = pad_along_dim - pad_low;
  std::vector<int64_t> padding{pad_low, pad_high};
  return std::move(padding);
}

int64_t GetConv2dTransposeOutDim(int64_t input_dim,
                                 int64_t ksize,
                                 int64_t stride,
                                 int64_t dilation,
                                 int64_t pad_low,
                                 int64_t pad_high,
                                 int64_t output_padding) {
  int64_t expanded_input_size = (input_dim - 1) * stride + 1;
  int64_t effective_filter_size = (ksize - 1) * dilation + 1;
  int64_t output_dim = expanded_input_size - 1 + output_padding +
                       effective_filter_size - pad_low - pad_high;
  return output_dim;
}

std::vector<int64_t> GetConv2dTransposePadding(
    const std::vector<int64_t>& input_spatial_dims,
    const std::vector<int64_t>& output_spatial_dims,
    const std::vector<int64_t>& ksize,
    const std::vector<int64_t>& stride,
    const std::vector<int64_t>& dilation,
    const std::vector<int64_t>& padding,
    const std::vector<int64_t>& output_padding,
    const std::string& auto_pad) {
  std::vector<int64_t> padding_value;
  for (size_t i = 0; i < input_spatial_dims.size(); ++i) {
    int64_t expanded_input_size = (input_spatial_dims[i] - 1) * stride[i] + 1;
    int64_t effective_filter_size = (ksize[i] - 1) * dilation[i] + 1;
    int64_t pad_before = effective_filter_size - 1 - padding[i * 2];
    int64_t padded_out_size =
        output_spatial_dims[i] + effective_filter_size - 1;
    int64_t pad_after = padded_out_size - expanded_input_size - pad_before;
    padding_value.emplace_back(pad_before);
    padding_value.emplace_back(pad_after);
  }
  return padding_value;
}

}  // namespace

static GcuOpPtr TranslateConv2d(
    GcuBuilderPtr gcu_builder,
    const pir::Operation* op,
    const std::vector<std::vector<GcuOpPtr>>& gcu_op_inputs) {
  auto input = *(gcu_op_inputs[0][0]);
  auto filter = *(gcu_op_inputs[1][0]);

  // Get attributes
  const auto& attributes = op->attributes();

  auto strides_array = attributes.at("strides").dyn_cast<pir::ArrayAttribute>();
  std::vector<int64_t> strides;
  for (size_t i = 0; i < strides_array.size(); ++i) {
    strides.push_back(
        strides_array.at(i).dyn_cast<pir::Int32Attribute>().data());
  }

  auto paddings_array =
      attributes.at("paddings").dyn_cast<pir::ArrayAttribute>();
  std::vector<int64_t> paddings;
  for (size_t i = 0; i < paddings_array.size(); ++i) {
    paddings.push_back(
        paddings_array.at(i).dyn_cast<pir::Int32Attribute>().data());
  }

  std::string padding_algorithm = attributes.at("padding_algorithm")
                                      .dyn_cast<pir::StrAttribute>()
                                      .AsString();

  auto dilations_array =
      attributes.at("dilations").dyn_cast<pir::ArrayAttribute>();
  std::vector<int64_t> dilations;
  for (size_t i = 0; i < dilations_array.size(); ++i) {
    dilations.push_back(
        dilations_array.at(i).dyn_cast<pir::Int32Attribute>().data());
  }

  int64_t groups =
      attributes.at("groups").dyn_cast<pir::Int32Attribute>().data();

  std::string data_format =
      attributes.at("data_format").dyn_cast<pir::StrAttribute>().AsString();

  std::vector<builder::Op> input_ops;
  if (data_format == "NCHW") {
    input_ops.emplace_back(builder::Transpose(input, {0, 2, 3, 1}));
    input_ops.emplace_back(builder::Transpose(filter, {2, 3, 1, 0}));
  } else {
    input_ops.emplace_back(input);
    input_ops.emplace_back(filter);
  }

  VLOG(3) << "TranslateConv2d, input shape:"
          << custom_engine::VectorToStr<int64_t>(input.GetType().GetShape())
          << ", filter shape:"
          << custom_engine::VectorToStr<int64_t>(filter.GetType().GetShape())
          << ", strides:" << custom_engine::VectorToStr<int64_t>(strides)
          << ", paddings:" << custom_engine::VectorToStr<int64_t>(paddings)
          << ", dilations:" << custom_engine::VectorToStr<int64_t>(dilations)
          << ", groups:" << groups
          << ", padding_algorithm:" << padding_algorithm
          << ", data_format:" << data_format;

  if (padding_algorithm == "SAME") {
    auto input_shape = input_ops[0].GetType().GetShape();
    auto kernel_shape = input_ops[1].GetType().GetShape();
    int64_t ih = input_shape[1];
    int64_t iw = input_shape[2];
    int64_t kh = kernel_shape[0];
    int64_t kw = kernel_shape[1];
    auto pad_h = GetPaddingWithAlgorithmSame(ih, kh, strides[0]);
    auto pad_w = GetPaddingWithAlgorithmSame(iw, kw, strides[1]);
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
  auto conv2d = builder::Conv2D(input_ops,
                                groups,
                                "NOTSET",  // auto_pad
                                "NHWC",    // layout
                                strides,
                                paddings,
                                dilations);

  conv2d.SetAttribute("op_type", builder::Attribute("Conv2DInference"));
  if (groups > 1) {
    VLOG(3) << "TranslateConv2d for pd_op.depthwise_conv2d";
  } else {
    VLOG(3) << "TranslateConv2d for pd_op.conv2d";
  }

  if (data_format == "NCHW") {
    auto transpose = builder::Transpose(conv2d, {0, 3, 1, 2});
    return std::make_shared<GcuOp>(transpose);
  } else {
    return std::make_shared<GcuOp>(conv2d);
  }
}

static GcuOpPtr TranslateConv2DTranspose(
    GcuBuilderPtr gcu_builder,
    const pir::Operation* op,
    const std::vector<std::vector<GcuOpPtr>>& gcu_op_inputs) {
  auto input = *(gcu_op_inputs[0][0]);
  auto filter = *(gcu_op_inputs[1][0]);
  //   auto output_size_tensor = *(gcu_op_inputs[2][0]);
  //   PADDLE_ENFORCE_EQ(output_size_tensor.IsConstant(), true,
  //                     common::errors::PreconditionNotMet(
  //                         "Input[2] output_size_tensor is not a Constant."));
  //   auto output_size = output_size_tensor.GetConstData<int64_t>();

  // Get attributes
  const auto& attributes = op->attributes();

  auto strides_array = attributes.at("strides").dyn_cast<pir::ArrayAttribute>();
  std::vector<int64_t> strides;
  for (size_t i = 0; i < strides_array.size(); ++i) {
    strides.push_back(
        strides_array.at(i).dyn_cast<pir::Int32Attribute>().data());
  }

  auto paddings_array =
      attributes.at("paddings").dyn_cast<pir::ArrayAttribute>();
  std::vector<int64_t> paddings;
  for (size_t i = 0; i < paddings_array.size(); ++i) {
    paddings.push_back(
        paddings_array.at(i).dyn_cast<pir::Int32Attribute>().data());
  }

  auto output_padding_array =
      attributes.at("output_padding").dyn_cast<pir::ArrayAttribute>();
  std::vector<int64_t> output_paddings;
  for (size_t i = 0; i < output_padding_array.size(); ++i) {
    output_paddings.push_back(
        output_padding_array.at(i).dyn_cast<pir::Int32Attribute>().data());
  }

  std::string padding_algorithm = attributes.at("padding_algorithm")
                                      .dyn_cast<pir::StrAttribute>()
                                      .AsString();

  auto dilations_array =
      attributes.at("dilations").dyn_cast<pir::ArrayAttribute>();
  std::vector<int64_t> dilations;
  for (size_t i = 0; i < dilations_array.size(); ++i) {
    dilations.push_back(
        dilations_array.at(i).dyn_cast<pir::Int32Attribute>().data());
  }

  int64_t groups =
      attributes.at("groups").dyn_cast<pir::Int32Attribute>().data();

  std::string data_format =
      attributes.at("data_format").dyn_cast<pir::StrAttribute>().AsString();

  std::vector<builder::Op> input_ops;
  if (data_format == "NCHW") {
    input_ops.emplace_back(builder::Transpose(input, {0, 2, 3, 1}));
    input_ops.emplace_back(builder::Transpose(filter, {2, 3, 1, 0}));
  } else {
    input_ops.emplace_back(input);
    input_ops.emplace_back(filter);
  }

  auto input_shape = input_ops[0].GetType().GetShape();
  auto kernel_shape = input_ops[1].GetType().GetShape();
  int64_t ih = input_shape[1];
  int64_t iw = input_shape[2];
  int64_t kh = kernel_shape[0];
  int64_t kw = kernel_shape[1];
  if (padding_algorithm == "SAME") {
    auto pad_h = GetPaddingWithAlgorithmSame(ih, kh, strides[0]);
    auto pad_w = GetPaddingWithAlgorithmSame(iw, kw, strides[1]);
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
  if (output_paddings.size() == 0) {
    output_paddings = {0, 0};
  } else if (output_paddings.size() == 1) {
    output_paddings = {output_paddings[0], output_paddings[0]};
  }
  auto oh = GetConv2dTransposeOutDim(ih,
                                     kh,
                                     strides[0],
                                     dilations[0],
                                     paddings[0],
                                     paddings[1],
                                     output_paddings[0]);
  auto ow = GetConv2dTransposeOutDim(iw,
                                     kw,
                                     strides[1],
                                     dilations[1],
                                     paddings[2],
                                     paddings[3],
                                     output_paddings[1]);
  const std::vector<int64_t> input_spatial_dims{ih, iw};
  const std::vector<int64_t> output_spatial_dims{oh, ow};
  const std::vector<int64_t> ksize{kh, kw};
  auto real_padding = GetConv2dTransposePadding(input_spatial_dims,
                                                output_spatial_dims,
                                                ksize,
                                                strides,
                                                dilations,
                                                paddings,
                                                output_paddings,
                                                "NOTSET");
  builder::ConvDimensionNumbers dims_attr(/*input_batch_dimension=*/0,
                                          /*input_feature_dimension=*/3,
                                          /*input_spatial_dimensions=*/{1, 2},
                                          /*kernel_input_feature_dimension=*/3,
                                          /*kernel_output_feature_dimension=*/2,
                                          /*kernel_spatial_dimensions=*/{0, 1},
                                          /*output_batch_dimension=*/0,
                                          /*output_feature_dimension=*/3,
                                          /*output_spatial_dimensions=*/{1, 2});
  auto conv2d_transpose = builder::Conv(input_ops[0],
                                        input_ops[1],
                                        dims_attr,
                                        /*window_strides=*/{1, 1},
                                        /*padding=*/real_padding,
                                        /*lhs_dilation=*/strides,
                                        /*rhs_dilation=*/dilations,
                                        /*window_reversal=*/{},
                                        /*auto_pad=*/"",
                                        /*feature_group_count=*/groups,
                                        /*batch_group_count=*/1,
                                        /*precision_config=*/{});
  conv2d_transpose.SetAttribute("op_type",
                                builder::Attribute("Conv2DBackpropInput"));
  if (data_format == "NCHW") {
    auto transpose = builder::Transpose(conv2d_transpose, {0, 3, 1, 2});
    return std::make_shared<GcuOp>(transpose);
  } else {
    return std::make_shared<GcuOp>(conv2d_transpose);
  }
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_conv2d, custom_engine::TranslateConv2d)
REGISTER_OP_TRANSLATOR(pd_op_depthwise_conv2d, custom_engine::TranslateConv2d)
REGISTER_OP_TRANSLATOR(pd_op_conv2d_transpose,
                       custom_engine::TranslateConv2DTranspose)

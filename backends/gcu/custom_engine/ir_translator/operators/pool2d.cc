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
}  // namespace

static GcuOpPtr TranslatePool2d(
    GcuBuilderPtr gcu_builder,
    const pir::Operation* op,
    const std::vector<std::vector<GcuOpPtr>>& gcu_op_inputs) {
  auto input = *(gcu_op_inputs[0][0]);
  auto kernel_size_tensor = *(gcu_op_inputs[1][0]);

  // Get attributes
  const auto& attributes = op->attributes();

  auto strides_array = attributes.at("strides").dyn_cast<pir::ArrayAttribute>();
  std::vector<int64_t> strides;
  for (size_t i = 0; i < strides_array.size(); ++i) {
    strides.push_back(
        strides_array.at(i).dyn_cast<pir::Int64Attribute>().data());
  }

  auto paddings_array =
      attributes.at("paddings").dyn_cast<pir::ArrayAttribute>();
  std::vector<int64_t> paddings;
  for (size_t i = 0; i < paddings_array.size(); ++i) {
    paddings.push_back(
        paddings_array.at(i).dyn_cast<pir::Int64Attribute>().data());
  }

  bool ceil_mode =
      attributes.at("ceil_mode").dyn_cast<pir::BoolAttribute>().data();

  bool exclusive =
      attributes.at("exclusive").dyn_cast<pir::BoolAttribute>().data();

  std::string data_format =
      attributes.at("data_format").dyn_cast<pir::StrAttribute>().AsString();

  std::string pooling_type =
      attributes.at("pooling_type").dyn_cast<pir::StrAttribute>().AsString();

  bool global_pooling =
      attributes.at("global_pooling").dyn_cast<pir::BoolAttribute>().data();

  bool adaptive =
      attributes.at("adaptive").dyn_cast<pir::BoolAttribute>().data();

  std::string padding_algorithm = attributes.at("padding_algorithm")
                                      .dyn_cast<pir::StrAttribute>()
                                      .AsString();

  PADDLE_ENFORCE_EQ(kernel_size_tensor.IsConstant(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Input[1] kernel_size_tensor is not a Constant."));
  auto ksize = kernel_size_tensor.GetConstData<int64_t>();

  std::vector<int64_t> spatial_dims;
  auto input_shape = input.GetType().GetShape();
  int64_t ih = 0;
  int64_t iw = 0;
  if (data_format == "NCHW") {
    spatial_dims = {2, 3};
    ih = input_shape[2];
    iw = input_shape[3];
  } else if (data_format == "NHWC") {
    spatial_dims = {1, 2};
    ih = input_shape[1];
    iw = input_shape[2];
  }
  int64_t kh = 0;
  int64_t kw = 0;
  if (adaptive) {
    std::vector<int64_t> output_spatial_size;
    for (auto dim : ksize) {
      output_spatial_size.emplace_back(dim);
    }
    if (ih % output_spatial_size[0] != 0 || iw % output_spatial_size[1] != 0) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "only support: MOD(oh, ih) == 0 && MOD(ow, iw) == 0"));
      return nullptr;
    }
    kh = ih / output_spatial_size[0];
    kw = iw / output_spatial_size[1];
    strides = {kh, kw};
  } else {
    kh = ksize[0];
    kw = ksize[1];
  }
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
  std::vector<int64_t> kernel_shape = {kh, kw};
  bool do_transpose =
      input.GetType().GetShape().size() == 4 && data_format == "NCHW";
  if (do_transpose) {
    if (pooling_type == "max") {
      if (global_pooling) {
        input = builder::Transpose(input, {0, 2, 3, 1});
        spatial_dims = {1, 2};
        auto tmp_op = std::make_shared<GcuOp>(
            builder::GlobalMaxPool(input, spatial_dims));
        return std::make_shared<GcuOp>(
            builder::Transpose(*tmp_op, {0, 3, 1, 2}));
      }
      input = builder::Transpose(input, {0, 2, 3, 1});
      auto tmp_op = std::make_shared<GcuOp>(builder::MaxPool2D(input,
                                                               kernel_shape,
                                                               ceil_mode,
                                                               false,
                                                               "NOTSET",
                                                               "NHWC",
                                                               strides,
                                                               paddings));
      return std::make_shared<GcuOp>(builder::Transpose(*tmp_op, {0, 3, 1, 2}));
    } else if (pooling_type == "avg") {
      if (global_pooling) {
        input = builder::Transpose(input, {0, 2, 3, 1});
        spatial_dims = {1, 2};
        std::vector<int64_t> out_shape = {input.GetType().GetShape().at(0),
                                          1,
                                          1,
                                          input.GetType().GetShape().at(3)};
        // force output shape because of hlir infershape question
        auto out_type =
            builder::Type(out_shape, input.GetType().GetPrimitiveType());
        auto tmp_op = std::make_shared<GcuOp>(
            builder::GlobalAveragePool(input, spatial_dims, out_type));
        return std::make_shared<GcuOp>(
            builder::Transpose(*tmp_op, {0, 3, 1, 2}));
      }
      input = builder::Transpose(input, {0, 2, 3, 1});
      spatial_dims = {1, 2};
      auto tmp_op = std::make_shared<GcuOp>(builder::AveragePool(input,
                                                                 spatial_dims,
                                                                 kernel_shape,
                                                                 ceil_mode,
                                                                 false,
                                                                 strides,
                                                                 paddings,
                                                                 "NOTSET"));
      return std::make_shared<GcuOp>(builder::Transpose(*tmp_op, {0, 3, 1, 2}));
    } else {
      PADDLE_THROW(
          phi::errors::Unimplemented("Unsupported "
                                     "pooling_type: %s",
                                     pooling_type.c_str()));
      return nullptr;
    }
  } else {
    if (pooling_type == "max") {
      if (global_pooling) {
        return std::make_shared<GcuOp>(
            builder::GlobalMaxPool(input, spatial_dims));
      }
      return std::make_shared<GcuOp>(builder::MaxPool2D(input,
                                                        kernel_shape,
                                                        ceil_mode,
                                                        false,
                                                        "NOTSET",
                                                        data_format.c_str(),
                                                        strides,
                                                        paddings));
    } else if (pooling_type == "avg") {
      if (global_pooling) {
        return std::make_shared<GcuOp>(
            builder::GlobalAveragePool(input, spatial_dims));
      }
      return std::make_shared<GcuOp>(builder::AveragePool(input,
                                                          spatial_dims,
                                                          kernel_shape,
                                                          ceil_mode,
                                                          false,
                                                          strides,
                                                          paddings,
                                                          "NOTSET"));
    } else {
      PADDLE_THROW(
          phi::errors::Unimplemented("Unsupported pooling"
                                     "_type: %s",
                                     pooling_type.c_str()));
      return nullptr;
    }
  }
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_pool2d, custom_engine::TranslatePool2d)

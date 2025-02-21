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
static GcuOpPtr TranslateNearestInterp(
    GcuBuilderPtr gcu_builder,
    const pir::Operation* op,
    const std::vector<std::vector<GcuOpPtr>>& gcu_op_inputs) {
  auto input = *(gcu_op_inputs[0][0]);
  auto out_size = *(gcu_op_inputs[1][0]);
  auto size_tensor = *(gcu_op_inputs[2][0]);
  auto scale_tensor = *(gcu_op_inputs[3][0]);

  // Get attributes
  const auto& attributes = op->attributes();

  std::string data_format =
      attributes.at("data_format").dyn_cast<pir::StrAttribute>().AsString();

  if (data_format != "NCHW" && data_format != "NHWC") {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "only support NCHW/NHWC for NearestInterpOp"));
    return nullptr;
  }

  int64_t out_d = static_cast<int64_t>(
      attributes.at("out_d").dyn_cast<pir::Int32Attribute>().data());
  int64_t out_h = static_cast<int64_t>(
      attributes.at("out_h").dyn_cast<pir::Int32Attribute>().data());
  int64_t out_w = static_cast<int64_t>(
      attributes.at("out_w").dyn_cast<pir::Int32Attribute>().data());

  auto scale_array = attributes.at("scale").dyn_cast<pir::ArrayAttribute>();
  std::vector<float> vec_scale;
  for (size_t i = 0; i < scale_array.size(); ++i) {
    vec_scale.push_back(
        scale_array.at(i).dyn_cast<pir::FloatAttribute>().data());
  }

  std::string interp_method =
      attributes.at("interp_method").dyn_cast<pir::StrAttribute>().AsString();
  bool align_corners =
      attributes.at("align_corners").dyn_cast<pir::BoolAttribute>().data();
  int32_t align_mode =
      attributes.at("align_mode").dyn_cast<pir::Int32Attribute>().data();

  if (data_format == "NCHW") {
    input = builder::Transpose(input, {0, 2, 3, 1});
  }
  auto input_shape = input.GetType().GetShape();

  builder::Op out_hw_op;
  builder::Op scale_op;

  if (size_tensor.IsValid()) {
    out_hw_op = size_tensor;
  } else if (out_size.IsValid()) {
    out_hw_op = out_size;
  } else if (scale_tensor.IsValid()) {
    scale_op = scale_tensor;
  } else {
    if (vec_scale.size() == 2) {
      out_h = static_cast<int64_t>(input_shape[1] * vec_scale[0]);
      out_w = static_cast<int64_t>(input_shape[2] * vec_scale[1]);
    } else {
      out_h = static_cast<int64_t>(input_shape[1] * vec_scale[0]);
      out_w = static_cast<int64_t>(input_shape[2] * vec_scale[0]);
    }
  }

  if (out_hw_op.IsValid() && out_hw_op.IsConstant()) {
    out_h = out_hw_op.GetConstData<int32_t>().at(0);
    out_w = out_hw_op.GetConstData<int32_t>().at(1);
  } else if (scale_op.IsValid() && scale_op.IsConstant()) {
    if (scale_op.GetType().GetSize() > 1) {
      float scale_h = scale_op.GetConstData<int64_t>()[0];
      float scale_w = scale_op.GetConstData<int64_t>()[1];
      out_h = static_cast<int64_t>(input_shape[1] * scale_h);
      out_w = static_cast<int64_t>(input_shape[2] * scale_w);
    } else {
      float scale = scale_op.GetConstData<int64_t>()[0];
      out_h = static_cast<int64_t>(input_shape[1] * scale);
      out_w = static_cast<int64_t>(input_shape[2] * scale);
    }
  }

  builder::Op sizes;
  builder::Op scales;
  if (out_h > 0 && out_w > 0) {
    std::vector<int64_t> output_shape = {
        input_shape[0], out_h, out_w, input_shape[3]};
    builder::Type sizes_type{{4}, builder::PrimitiveType::S64()};
    sizes = builder::Const(gcu_builder, output_shape, sizes_type);
    scales = builder::Empty(gcu_builder, builder::PrimitiveType::F32());
  } else if (out_hw_op.IsValid()) {
    scales = builder::Empty(gcu_builder, builder::PrimitiveType::F32());
    auto type = builder::Type({1}, out_hw_op.GetType().GetPrimitiveType());
    auto n_op = builder::Const(gcu_builder, input_shape[0], type);
    auto c_op = builder::Const(gcu_builder, input_shape[3], type);
    sizes = builder::Concatenate({n_op, out_hw_op, c_op}, 0);
  } else if (scale_op.IsValid()) {
    sizes = builder::Empty(gcu_builder, builder::PrimitiveType::S64());
    auto type = builder::Type({1}, scale_op.GetType().GetPrimitiveType());
    auto n_scale_op = builder::Const(gcu_builder, 1.0, type);
    auto c_scale_op = builder::Const(gcu_builder, 1.0, type);
    if (scale_op.GetType().GetSize() > 1) {
      scales = builder::Concatenate({n_scale_op, scale_op, c_scale_op}, 0);
    } else {
      scales =
          builder::Concatenate({n_scale_op, scale_op, scale_op, c_scale_op}, 0);
    }
  }

  std::vector<float> roi_val{1};
  builder::Type roi_type(builder::PrimitiveType::F32());
  auto roi =
      builder::Const(gcu_builder, static_cast<void*>(roi_val.data()), roi_type);

  if (interp_method != "bilinear") {
    // Based on Comment in `paddle/fluid/operators/interpolate_v2_op.cc:
    // InterpolateV2OpMaker`, align_mode is used for bilinear.
    align_mode = 0;
  }

  std::map<std::string, int64_t> mode_map{
      {"nearest", 0}, {"linear", 1}, {"cubic", 2}};
  std::map<std::string, int64_t> nearest_mode_map{{"simple", 0},
                                                  {"round_prefer_floor", 1},
                                                  {"round_prefer_ceil", 2},
                                                  {"floor", 3},
                                                  {"ceil", 4}};
  std::map<std::string, int64_t> coordinate_transformation_mode_map{
      {"half_pixel", 0},
      {"asymmetric", 1},
      {"pytorch_half_pixel", 2},
      {"tf_half_pixel_for_nn", 3},
      {"align_corners", 4},
      {"tf_crop_and_resize", 5}};
  int64_t mode = mode_map[interp_method];
  int64_t coordinate_transformation_mode = 0;
  if (align_corners) {
    coordinate_transformation_mode = 4;
  } else if (align_mode == 0) {
    coordinate_transformation_mode =
        coordinate_transformation_mode_map["half_pixel"];
  } else if (align_mode == 1) {
    coordinate_transformation_mode =
        coordinate_transformation_mode_map["asymmetric"];
  }

  bool exclude_outside = false;
  int64_t nearest_mode = nearest_mode_map["round_prefer_floor"];
  float extrapolation_value = 0.0;
  float cubic_coeff_a = -0.75;
  std::vector<int64_t> resize_dimensions;
  auto resize = builder::Resize(input,
                                roi,
                                scales,
                                sizes,
                                mode,
                                coordinate_transformation_mode,
                                exclude_outside,
                                nearest_mode,
                                extrapolation_value,
                                cubic_coeff_a,
                                resize_dimensions);

  if (data_format == "NHWC") {
    return std::make_shared<GcuOp>(resize);
  } else {
    return std::make_shared<GcuOp>(builder::Transpose(resize, {0, 3, 1, 2}));
  }
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_nearest_interp,
                       custom_engine::TranslateNearestInterp)

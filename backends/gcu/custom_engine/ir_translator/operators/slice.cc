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

static GcuOpPtr TranslateSlice(
    GcuBuilderPtr gcu_builder,
    const pir::Operation *op,
    const std::vector<std::vector<GcuOpPtr>> &gcu_op_inputs) {
  // Get attributes
  const auto &attributes = op->attributes();
  auto axes_list =
      attributes.at("axes").dyn_cast<pir::ArrayAttribute>().AsVector();
  std::vector<int64_t> axes;
  if (axes_list.size() > 0) {
    PADDLE_ENFORCE_EQ(axes_list[0].isa<pir::Int64Attribute>(),
                      true,
                      common::errors::Unimplemented(
                          "the 0th axes MUST be pir::Int64Attribute"));
    for (size_t i = 0; i < axes_list.size(); ++i) {
      axes.push_back(axes_list[i].dyn_cast<pir::Int64Attribute>().data());
    }
  }

  auto infer_flags_list =
      attributes.at("infer_flags").dyn_cast<pir::ArrayAttribute>().AsVector();
  std::vector<int64_t> infer_flags;
  if (infer_flags_list.size() > 0) {
    PADDLE_ENFORCE_EQ(infer_flags_list[0].isa<pir::Int64Attribute>(),
                      true,
                      common::errors::Unimplemented(
                          "the 0th infer_flags MUST be pir::Int64Attribute"));
    for (size_t i = 0; i < infer_flags_list.size(); ++i) {
      infer_flags.push_back(
          infer_flags_list[i].dyn_cast<pir::Int64Attribute>().data());
    }
  }

  auto decrease_axis_list =
      attributes.at("decrease_axis").dyn_cast<pir::ArrayAttribute>().AsVector();
  std::vector<int64_t> decrease_axis;
  if (decrease_axis_list.size() > 0) {
    PADDLE_ENFORCE_EQ(decrease_axis_list[0].isa<pir::Int64Attribute>(),
                      true,
                      common::errors::Unimplemented(
                          "the 0th decrease_axis MUST be pir::Int64Attribute"));
    for (size_t i = 0; i < decrease_axis_list.size(); ++i) {
      decrease_axis.push_back(
          decrease_axis_list[i].dyn_cast<pir::Int64Attribute>().data());
    }
  }

  auto input = *(gcu_op_inputs[0][0]);

  auto starts_tensor = *(gcu_op_inputs[1][0]);
  PADDLE_ENFORCE_EQ(starts_tensor.IsConstant(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Input[1] starts_tensor is not a Constant."));
  auto starts = starts_tensor.GetConstData<int64_t>();

  auto ends_tensor = *(gcu_op_inputs[2][0]);
  PADDLE_ENFORCE_EQ(ends_tensor.IsConstant(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Input[1] ends_tensor is not a Constant."));
  auto ends = ends_tensor.GetConstData<int64_t>();

  auto rank = input.GetType().GetRank();
  const std::vector<int64_t> &input_shapes = input.GetType().GetShape();
  std::vector<int64_t> start_indices(rank, 0);
  std::vector<int64_t> limit_indices = input_shapes;
  for (size_t i = 0; i < axes.size(); ++i) {
    int dim = axes[i];
    if (dim < 0) {
      dim += rank;
    }
    start_indices[dim] =
        starts[i] < 0 ? starts[i] + input_shapes[dim] : starts[i];
    start_indices[dim] = std::max(start_indices[dim], 0L);
    start_indices[dim] = std::min(start_indices[dim], input_shapes[dim]);

    limit_indices[dim] = ends[i] < 0 ? ends[i] + input_shapes[dim] : ends[i];
    limit_indices[dim] = std::min(limit_indices[dim], input_shapes[dim]);
    limit_indices[dim] = std::max(limit_indices[dim], 0L);
  }
  std::vector<int64_t> strides(rank, 1);

  auto slice = builder::Slice(input, start_indices, limit_indices, strides);

  if (decrease_axis.size() == 0) {
    return std::make_shared<GcuOp>(slice);
  } else {
    auto slice_shape = slice.GetType().GetShape();
    std::vector<int64_t> new_shape;
    size_t iter = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(slice_shape.size()); ++i) {
      if (iter < decrease_axis.size() && i == decrease_axis[iter]) {
        ++iter;
      } else {
        new_shape.emplace_back(slice_shape[i]);
      }
    }
    if (new_shape.empty()) {
      new_shape.emplace_back(1);
    }
    return std::make_shared<GcuOp>(builder::Reshape(slice, new_shape));
  }
}

}  // namespace custom_engine

REGISTER_OP_TRANSLATOR(pd_op_slice, custom_engine::TranslateSlice)

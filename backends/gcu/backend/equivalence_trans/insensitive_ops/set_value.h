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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "backend/equivalence_trans/utils.h"
#include "backend/register/register.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace backend {
const char* const kSetValue = "set_value";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, SetValueEquivalenceTrans) {
  auto CreateBindingFunc =
      [](std::shared_ptr<builder::Builder> builder,
         const std::vector<BindingFuncType>& func_types,
         const std::vector<builder::PrimitiveType>& ptypes,
         const std::string& base_name) -> std::vector<std::string> {
    PADDLE_ENFORCE_EQ(ptypes.size(),
                      func_types.size(),
                      phi::errors::NotFound(
                          "func_types:%zu must have same size with ptypes:%zu",
                          func_types.size(),
                          ptypes.size()));
    std::vector<std::string> func_names;
    func_names.reserve(func_types.size());
    builder::Op arg0, arg1, out;
    for (size_t i = 0; i < func_types.size(); ++i) {
      std::string func_name = base_name + std::to_string(i);
      builder->AddFunc(func_name.c_str());
      builder::Type scalar_type({1}, ptypes[i]);
      arg0 = builder->CreateInput(scalar_type, func_name.c_str());
      arg1 = builder->CreateInput(scalar_type, func_name.c_str());
      switch (func_types[i]) {
        case BindingFuncType::IDENTITY:
          out = arg0;
          break;
        case BindingFuncType::ADD:
          out = arg0 + arg1;
          break;
        case BindingFuncType::MAX:
          out = builder::Max(arg0, arg1);
          break;
        case BindingFuncType::MIN:
          out = builder::Min(arg0, arg1);
          break;
        case BindingFuncType::GE:
          out = builder::GreaterEqual(arg0, arg1);
          break;
        case BindingFuncType::OR:
          out = builder::Or(arg0, arg1);
          break;
        default:
          PADDLE_THROW(phi::errors::NotFound("Unsupport BindingFuncType."));
          break;
      }
      builder->SetOutput({out}, func_name.c_str());
      func_names.emplace_back(func_name);
    }
    return std::move(func_names);
  };

  auto input = *(map_inputs["Input"].at(0));
  auto input_shape = input.GetType().GetShape();

  // auto dtype = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto axes = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("axes"));
  auto starts = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("starts"));
  auto ends = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("ends"));
  auto steps = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("steps"));
  auto decrease_axes =
      PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("decrease_axes"));
  auto none_axes =
      PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("none_axes"));
  auto bool_values =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("bool_values"));
  auto fp32_values =
      PADDLE_GET_CONST(std::vector<float>, op->GetAttr("fp32_values"));
  auto int32_values =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("int32_values"));
  auto int64_values =
      PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("int64_values"));
  auto fp64_values =
      PADDLE_GET_CONST(std::vector<double>, op->GetAttr("fp64_values"));
  auto shape = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));

  // create value_t
  builder::Op value_t;
  if (map_inputs.count("ValueTensor") != 0) {
    value_t = *(map_inputs["ValueTensor"].at(0));
    shape = value_t.GetType().GetShape();
  } else {
    void* value_ptr = nullptr;
    builder::Type value_type;
    if (bool_values.size() != 0) {
      value_type = builder::Type(shape, builder::PrimitiveType::PRED());
      value_ptr = static_cast<void*>(bool_values.data());

    } else if (fp32_values.size() != 0) {
      value_type = builder::Type(shape, builder::PrimitiveType::F32());
      value_ptr = static_cast<void*>(fp32_values.data());
    } else if (int32_values.size() != 0) {
      value_type = builder::Type(shape, builder::PrimitiveType::S32());
      value_ptr = static_cast<void*>(int32_values.data());
    } else if (int64_values.size() != 0) {
      value_type = builder::Type(shape, builder::PrimitiveType::S64());
      value_ptr = static_cast<void*>(int64_values.data());
    } else if (fp64_values.size() != 0) {
      // value_type = builder::Type(shape, builder::PrimitiveType::F64());
      // value_ptr = static_cast<void*>(fp64_values.data());
      __THROW_ERROR_INTERNAL__(
          phi::errors::InvalidArgument("fp64 not support yet."));
    }

    PADDLE_ENFORCE_NE(
        value_ptr,
        nullptr,
        phi::errors::InvalidArgument("value_ptr should not be nullptr"));

    value_t = builder::Const(input.GetBuilder(), value_ptr, value_type);
  }

  if (map_inputs.count("StartsTensorList") != 0) {
    auto start_tensor_list = *(map_inputs["StartsTensorList"].at(0));
    starts = start_tensor_list.GetConstData<int64_t>();
  }

  if (map_inputs.count("EndsTensorList") != 0) {
    auto end_tensor_list = *(map_inputs["EndsTensorList"].at(0));
    ends = end_tensor_list.GetConstData<int64_t>();
  }

  if (map_inputs.count("StepsTensorList") != 0) {
    auto step_tensor_list = *(map_inputs["StepsTensorList"].at(0));
    steps = step_tensor_list.GetConstData<int64_t>();
  }

  auto in_dims = phi::make_ddim(input.GetType().GetShape());
  phi::funcs::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends, &steps);
  auto slice_dims =
      phi::funcs::GetSliceDims(in_dims, axes, starts, ends, &steps);
  auto decrease_slice_dims =
      phi::funcs::GetDecreasedDims(slice_dims, decrease_axes);

  auto slice_dims_for_assign = decrease_slice_dims;

  if (!none_axes.empty()) {
    size_t none_axes_cur = 0, decrease_axes_cur = 0;
    std::vector<int64_t> slice_dims_with_none;
    for (int i = 0; i < slice_dims.size(); ++i) {
      while (none_axes_cur < none_axes.size() &&
             none_axes[none_axes_cur] <= i) {
        slice_dims_with_none.push_back(1);
        none_axes_cur++;
      }
      if (decrease_axes_cur < decrease_axes.size() &&
          decrease_axes[decrease_axes_cur] == i) {
        decrease_axes_cur++;
      } else {
        slice_dims_with_none.push_back(slice_dims[i]);
      }
    }
    while (none_axes_cur < none_axes.size()) {
      slice_dims_with_none.push_back(1);
      none_axes_cur++;
    }

    slice_dims_for_assign = phi::make_ddim(slice_dims_with_none);
  }

  auto starts_indices = std::vector<int64_t>(in_dims.size(), 0);
  auto ends_indices = std::vector<int64_t>(in_dims.size(), 0);
  auto strides_indices = std::vector<int64_t>(in_dims.size(), 0);

  for (int i = 0; i < in_dims.size(); ++i) {
    starts_indices[i] = 0;
    ends_indices[i] = slice_dims[i];
    strides_indices[i] = 1;
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int axis_index = axes[i];
    starts_indices[axis_index] = starts[i];
    ends_indices[axis_index] = ends[i];
    strides_indices[axis_index] = steps[i];
  }

  int64_t stride_step = phi::product(in_dims);
  std::vector<int64_t> index_indices(1, 0);
  for (size_t i = 0; i < strides_indices.size(); ++i) {
    auto index_size = index_indices.size();
    stride_step /= in_dims[i];
    for (size_t j = 0; j < index_size; ++j) {
      auto start_index = *index_indices.begin();
      if (strides_indices[i] > 0) {
        for (int64_t k = starts_indices[i]; k < ends_indices[i];
             k += strides_indices[i]) {
          index_indices.push_back(start_index + k * stride_step);
        }
      } else {
        for (int64_t k = starts_indices[i]; k > ends_indices[i];
             k += strides_indices[i]) {
          index_indices.push_back(start_index + k * stride_step);
        }
      }
      index_indices.erase(index_indices.begin());
    }
  }

  if (index_indices.size() == 0) {
    return std::make_shared<GcuOp>(input);
  }

  PADDLE_ENFORCE_EQ(
      static_cast<int64_t>(index_indices.size()),
      phi::product(slice_dims_for_assign),
      phi::errors::InvalidArgument(
          "OP(set_value) error index indices and value update not match "));

  builder::Op value_temp;
  auto slice_dims_for_assign_v = phi::vectorize(slice_dims_for_assign);

  PADDLE_ENFORCE_GE(
      slice_dims_for_assign_v.size(),
      shape.size(),
      phi::errors::InvalidArgument(
          "OP(set_value) error: the rank of slice_dims_for_assign_v must "
          "larger than or equal the rank of value shape. "));

  if (slice_dims_for_assign_v == value_t.GetType().GetShape()) {
    value_temp = value_t;
  } else {
    auto value_type = builder::Type(slice_dims_for_assign_v,
                                    value_t.GetType().GetPrimitiveType());
    std::vector<int64_t> broadcast_dims;

    if (shape.size() >= 1) {
      auto value_rank = shape.size();
      auto slice_rank = slice_dims_for_assign_v.size();
      for (size_t i = slice_rank - value_rank; i < slice_rank; i++) {
        broadcast_dims.push_back(i);
      }
    }

    value_temp = builder::BroadcastInDim(value_t, broadcast_dims, value_type);
  }

  std::vector<int64_t> index_shape = {
      static_cast<int64_t>(index_indices.size()), 1};
  int64_t index_vector_dim = 1;

  std::vector<int64_t> update_window_dims;
  std::vector<int64_t> inserted_window_dims;
  std::vector<int64_t> scatter_dims_to_operand_dims;

  update_window_dims.emplace_back(1);

  inserted_window_dims.emplace_back(0);

  for (int64_t i = 0; i < index_shape[index_vector_dim]; ++i) {
    scatter_dims_to_operand_dims.emplace_back(i);
  }

  builder::ScatterDimensionNumbers dim_numbers(update_window_dims,
                                               inserted_window_dims,
                                               scatter_dims_to_operand_dims,
                                               index_vector_dim);

  auto input_ptype = input.GetType().GetPrimitiveType();

  auto tmp_region_list = CreateBindingFunc(
      input.GetBuilder(), {BindingFuncType::IDENTITY}, {input_ptype}, "body_");
  std::vector<const char*> region_list;
  for (auto& region : tmp_region_list) region_list.push_back(region.c_str());

  auto index_indices_type =
      builder::Type(index_shape, builder::PrimitiveType::S64());

  GcuOp index_indices_op =
      builder::Const(input.GetBuilder(),
                     static_cast<void*>(index_indices.data()),
                     index_indices_type);
  int64_t input_numel = phi::product(in_dims);
  int64_t index_numel = index_indices.size();

  // do reshape
  auto reshaped_input_type = builder::Type({input_numel, 1}, input_ptype);
  auto reshaped_value_type = builder::Type({index_numel, 1}, input_ptype);

  input = builder::Reshape(input, reshaped_input_type);
  value_temp = builder::Reshape(value_temp, reshaped_value_type);

  auto out = builder::Scatter(
      input, index_indices_op, value_temp, dim_numbers, region_list);

  // reshape back
  out = builder::Reshape(out, builder::Type(input_shape, input_ptype));

  return std::make_shared<GcuOp>(out);
}

EQUIVALENCE_TRANS_FUNC_REG(kSetValue, INSENSITIVE, SetValueEquivalenceTrans);

}  // namespace backend

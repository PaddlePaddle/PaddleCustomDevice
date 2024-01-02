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

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "backend/register/register.h"

namespace backend {
const char* const kStridedSlice = "strided_slice";
const char* const kStridedSliceGrad = "strided_slice_grad";
static void StridedSliceOutDims(const std::vector<int64_t>& starts,
                                const std::vector<int64_t>& ends,
                                const std::vector<int64_t>& strides,
                                const std::vector<int>& axes,
                                const std::vector<int>& infer_flags,
                                const phi::DDim in_dims,
                                const std::vector<int>& decrease_axis,
                                int64_t* out_dims_vector,
                                const size_t size,
                                bool infer_shape) {
  for (int i = 0; i < in_dims.size(); i++) {
    out_dims_vector[i] = in_dims[i];
  }
  int64_t stride_index, start_index, end_index;
  for (size_t i = 0; i < size; i++) {
    int axes_index = axes[i];
    start_index = starts[i];
    end_index = ends[i];
    stride_index = strides[i];
    bool decrease_axis_affect = false;
    if (start_index == -1 && end_index == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    if (decrease_axis_affect) {
      out_dims_vector[axes_index] = 1;
      continue;
    }
    if (infer_shape && infer_flags[i] == -1) {
      out_dims_vector[axes_index] = -1;
      continue;
    }

    PADDLE_ENFORCE_NE(stride_index,
                      0,
                      phi::errors::InvalidArgument(
                          "stride index in StridedSlice operator is 0."));
    int64_t axis_size = in_dims[axes_index];

    if (axis_size < 0) {
      continue;
    }

    if (start_index < 0) {
      start_index = start_index + axis_size;
      start_index = std::max<int64_t>(start_index, 0);
    }
    if (end_index < 0) {
      if (!(end_index == -1 && stride_index < 0)) {  // skip None stop condition
        end_index = end_index + axis_size;
        if (end_index < 0) {
          end_index = 0;
        }
      }
    }

    if (stride_index < 0) {
      start_index = start_index + 1;
      end_index = end_index + 1;
    }

    bool neg_dim_condition = ((stride_index < 0 && (start_index < end_index)) ||
                              (stride_index > 0 && (start_index > end_index)));
    PADDLE_ENFORCE_EQ(neg_dim_condition,
                      false,
                      phi::errors::InvalidArgument(
                          "The start index and end index are invalid for their "
                          "corresponding stride."));

    int64_t left =
        std::max(static_cast<int64_t>(0), std::min(start_index, end_index));
    int64_t right = std::min(axis_size, std::max(start_index, end_index));
    int64_t step = std::abs(stride_index);

    auto out_dims_index = (std::abs(right - left) + step - 1) / step;

    out_dims_vector[axes_index] = out_dims_index;
  }
}

static void StridedSliceFunctor(int64_t* starts,
                                int64_t* ends,
                                int64_t* strides,
                                const int* axes,
                                int* reverse_axis,
                                const phi::DDim dims,
                                const std::vector<int>& infer_flags,
                                const std::vector<int>& decrease_axis,
                                const size_t size) {
  for (size_t axis = 0; axis < size; axis++) {
    int64_t axis_size = dims[axes[axis]];
    int axis_index = axis;
    if (axis_size < 0) {
      starts[axis_index] = 0;
      ends[axis_index] = 1;
      strides[axis_index] = 1;
    }
    bool decrease_axis_affect = false;
    if (starts[axis_index] == -1 && ends[axis_index] == 0 &&
        infer_flags[axis_index] == -1) {
      auto ret = std::find(
          decrease_axis.begin(), decrease_axis.end(), axes[axis_index]);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }
    // stride must not be zero
    if (starts[axis_index] < 0) {
      starts[axis_index] = starts[axis_index] + axis_size;
      starts[axis_index] = std::max<int64_t>(starts[axis_index], 0);
    }
    if (ends[axis_index] < 0) {
      if (!(ends[axis_index] == -1 &&
            strides[axis_index] < 0)) {  // skip None stop condition
        ends[axis_index] = ends[axis_index] + axis_size;
        if (ends[axis_index] < 0) {
          ends[axis_index] = 0;
        }
      }
    }
    if (decrease_axis_affect) {
      if (strides[axis_index] < 0) {
        ends[axis_index] = starts[axis_index] - 1;
      } else {
        ends[axis_index] = starts[axis_index] + 1;
      }
    }

    if (strides[axis_index] < 0) {
      reverse_axis[axis_index] = 1;
      strides[axis_index] = -strides[axis_index];
      if (starts[axis_index] > ends[axis_index]) {
        // swap the reverse
        auto end_dim = axis_size - 1 < starts[axis_index] ? axis_size - 1
                                                          : starts[axis_index];
        auto offset = (end_dim - ends[axis_index]) % strides[axis_index];
        offset = offset == 0 ? strides[axis_index] : offset;

        starts[axis_index] = starts[axis_index] + offset;
        ends[axis_index] = ends[axis_index] + offset;
      }
      std::swap(starts[axis_index], ends[axis_index]);
    } else {
      reverse_axis[axis_index] = 0;
      strides[axis_index] = strides[axis_index];
    }
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, op, map_inputs, running_mode, StridedSliceEquivalenceTrans) {
  builder::Op input = *(map_inputs["Input"].at(0));
  std::vector<int64_t> input_shapes = input.GetType().GetShape();
  const int64_t input_rank = input_shapes.size();

  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  std::vector<int64_t> strides;
  if (map_inputs.count("StartsTensorList") != 0) {
    for (size_t i = 0; i < map_inputs["StartsTensorList"].size(); ++i)
      starts.push_back(
          map_inputs["StartsTensorList"][i]->GetConstData<int>()[0]);
  } else if (map_inputs.count("StartsTensor") != 0) {
    auto start_tensor_op = *(map_inputs["StartsTensor"][0]);
    auto data_ptr = start_tensor_op.GetConstData<int>();
    for (int64_t i = 0;
         i < static_cast<int64_t>(start_tensor_op.GetType().GetSize());
         ++i)
      starts.push_back(data_ptr[i]);
  } else {
    auto starts_i32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("starts"));
    starts =
        std::move(std::vector<int64_t>(starts_i32.begin(), starts_i32.end()));
  }

  if (map_inputs.count("EndsTensorList") != 0) {
    for (size_t i = 0; i < map_inputs["EndsTensorList"].size(); ++i)
      ends.push_back(map_inputs["EndsTensorList"][i]->GetConstData<int>()[0]);
  } else if (map_inputs.count("EndsTensor") != 0) {
    auto ends_tensor_op = *(map_inputs["EndsTensor"][0]);
    auto data_ptr = ends_tensor_op.GetConstData<int>();
    for (int64_t i = 0;
         i < static_cast<int64_t>(ends_tensor_op.GetType().GetSize());
         ++i)
      ends.push_back(data_ptr[i]);
  } else {
    auto ends_i32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("ends"));
    ends = std::move(std::vector<int64_t>(ends_i32.begin(), ends_i32.end()));
  }

  if (map_inputs.count("StridesTensorList") != 0) {
    for (size_t i = 0; i < map_inputs["StridesTensorList"].size(); ++i)
      strides.push_back(
          map_inputs["StridesTensorList"][i]->GetConstData<int>()[0]);
  } else if (map_inputs.count("StridesTensor") != 0) {
    auto strides_tensor_op = *(map_inputs["StridesTensor"][0]);
    auto data_ptr = strides_tensor_op.GetConstData<int>();
    for (int64_t i = 0;
         i < static_cast<int64_t>(strides_tensor_op.GetType().GetSize());
         ++i)
      strides.push_back(data_ptr[i]);
  } else {
    auto strides_i32 =
        PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
    strides =
        std::move(std::vector<int64_t>(strides_i32.begin(), strides_i32.end()));
  }

  auto axes = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  auto infer_flags =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("infer_flags"));
  auto decrease_axis =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("decrease_axis"));

  auto in_dims = phi::make_ddim(input_shapes);
  // out dims calculation
  std::vector<int64_t> out_dims_vector(input_rank, -1);
  StridedSliceOutDims(starts,
                      ends,
                      strides,
                      axes,
                      infer_flags,
                      in_dims,
                      decrease_axis,
                      out_dims_vector.data(),
                      axes.size(),
                      false);

  // check whether need to reverse (false: stride > 0; true: stride < 0)
  std::vector<int> reverse_vector(axes.size(), 0);
  StridedSliceFunctor(starts.data(),
                      ends.data(),
                      strides.data(),
                      axes.data(),
                      reverse_vector.data(),
                      in_dims,
                      infer_flags,
                      decrease_axis,
                      axes.size());

  // construct the starts_indices, ends_indices and strides_indices tensor for
  // calling StridedSlice op
  std::vector<int64_t> starts_indices(input_rank, 0);
  std::vector<int64_t> ends_indices(out_dims_vector);
  std::vector<int64_t> strides_indices(input_rank, 1);
  for (size_t axis = 0; axis < axes.size(); axis++) {
    int32_t axis_index = axes[axis];
    starts_indices[axis_index] = starts[axis];
    ends_indices[axis_index] = std::min(ends[axis], in_dims[axis_index]);
    strides_indices[axis_index] = strides[axis];
  }

  auto out =
      builder::Slice(input, starts_indices, ends_indices, strides_indices);

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }
  if (need_reverse) {
    std::vector<int64_t> reverse_axis_vector;
    for (size_t axis = 0; axis < axes.size(); axis++) {
      if (reverse_vector[axis] == 1) {
        reverse_axis_vector.push_back(axes[axis]);
      }
    }
    out = builder::Reverse(out, reverse_axis_vector);
  }

  auto out_dims_origin = out_dims_vector;
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          out_dims_vector[decrease_axis[i]],
          1,
          phi::errors::InvalidArgument(
              "the size of decrease dimension should be 1, but received %d.",
              out_dims_vector[decrease_axis[i]]));
      out_dims_origin[decrease_axis[i]] = 0;
    }

    for (size_t i = 0; i < out_dims_origin.size(); ++i) {
      if (out_dims_origin[i] != 0) {
        new_out_shape.push_back(out_dims_origin[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }
    out_dims_origin = new_out_shape;
  }
  if (decrease_axis.size() > 0) out = builder::Reshape(out, out_dims_origin);

  return std::make_shared<GcuOp>(out);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               op,
                               map_inputs,
                               running_mode,
                               StridedSliceGradEquivalenceTrans) {
  builder::Op input = *(map_inputs["Input"].at(0));
  builder::Op out_grad = *(map_inputs["Out@GRAD"].at(0));

  std::vector<int64_t> input_shape = input.GetType().GetShape();
  const int64_t input_rank = input_shape.size();

  std::vector<int64_t> starts;
  std::vector<int64_t> ends;
  std::vector<int64_t> strides;
  if (map_inputs.count("StartsTensorList") != 0) {
    for (size_t i = 0; i < map_inputs["StartsTensorList"].size(); ++i)
      starts.push_back(
          map_inputs["StartsTensorList"][i]->GetConstData<int>()[0]);
  } else if (map_inputs.count("StartsTensor") != 0) {
    auto start_tensor_op = *(map_inputs["StartsTensor"][0]);
    auto data_ptr = start_tensor_op.GetConstData<int>();
    for (int64_t i = 0;
         i < static_cast<int64_t>(start_tensor_op.GetType().GetSize());
         ++i)
      starts.push_back(data_ptr[i]);
  } else {
    auto starts_i32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("starts"));
    starts =
        std::move(std::vector<int64_t>(starts_i32.begin(), starts_i32.end()));
  }

  if (map_inputs.count("EndsTensorList") != 0) {
    for (size_t i = 0; i < map_inputs["EndsTensorList"].size(); ++i)
      ends.push_back(map_inputs["EndsTensorList"][i]->GetConstData<int>()[0]);
  } else if (map_inputs.count("EndsTensor") != 0) {
    auto ends_tensor_op = *(map_inputs["EndsTensor"][0]);
    auto data_ptr = ends_tensor_op.GetConstData<int>();
    for (int64_t i = 0;
         i < static_cast<int64_t>(ends_tensor_op.GetType().GetSize());
         ++i)
      ends.push_back(data_ptr[i]);
  } else {
    auto ends_i32 = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("ends"));
    ends = std::move(std::vector<int64_t>(ends_i32.begin(), ends_i32.end()));
  }

  if (map_inputs.count("StridesTensorList") != 0) {
    for (size_t i = 0; i < map_inputs["StridesTensorList"].size(); ++i)
      strides.push_back(
          map_inputs["StridesTensorList"][i]->GetConstData<int>()[0]);
  } else if (map_inputs.count("StridesTensor") != 0) {
    auto strides_tensor_op = *(map_inputs["StridesTensor"][0]);
    auto data_ptr = strides_tensor_op.GetConstData<int>();
    for (int64_t i = 0;
         i < static_cast<int64_t>(strides_tensor_op.GetType().GetSize());
         ++i)
      strides.push_back(data_ptr[i]);
  } else {
    auto strides_i32 =
        PADDLE_GET_CONST(std::vector<int>, op->GetAttr("strides"));
    strides =
        std::move(std::vector<int64_t>(strides_i32.begin(), strides_i32.end()));
  }

  auto axes = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  auto infer_flags =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("infer_flags"));
  auto decrease_axis =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("decrease_axis"));

  auto in_dims = phi::make_ddim(input_shape);
  // out dims calculation
  std::vector<int64_t> out_dims_vector(input_rank, -1);
  StridedSliceOutDims(starts,
                      ends,
                      strides,
                      axes,
                      infer_flags,
                      in_dims,
                      decrease_axis,
                      out_dims_vector.data(),
                      axes.size(),
                      false);

  // check whether need to reverse (false: stride > 0; true: stride < 0)
  std::vector<int> reverse_vector(axes.size(), 0);
  StridedSliceFunctor(starts.data(),
                      ends.data(),
                      strides.data(),
                      axes.data(),
                      reverse_vector.data(),
                      in_dims,
                      infer_flags,
                      decrease_axis,
                      axes.size());

  bool need_reverse = false;
  for (size_t axis = 0; axis < axes.size(); axis++) {
    if (reverse_vector[axis] == 1) {
      need_reverse = true;
      break;
    }
  }
  if (need_reverse) {
    std::vector<int64_t> reverse_axis_vector;
    for (size_t axis = 0; axis < axes.size(); axis++) {
      if (reverse_vector[axis] == 1) {
        reverse_axis_vector.push_back(axes[axis]);
      }
    }
    out_grad = builder::Reverse(out_grad, reverse_axis_vector);
    // TODO(ALL TBD): maybe need to swap starts with ends, and make strides as
    // positive.
  }

  for (size_t axis = 0; axis < axes.size(); ++axis) {
    int32_t axis_index = axes[axis];
    if (starts[axis] < 0) starts[axis] = starts[axis] + input_shape[axis_index];
    if (ends[axis] < 0) ends[axis] = ends[axis] + input_shape[axis_index];
    starts[axis] = std::max(starts[axis], 0L);
    starts[axis] = std::min(starts[axis], input_shape[axis_index]);
    ends[axis] = std::min(ends[axis], input_shape[axis_index]);
    ends[axis] = std::max(ends[axis], 0L);
  }

  std::vector<int64_t> padding_low(input_rank, 0);
  std::vector<int64_t> padding_high(input_rank, 0);
  std::vector<int64_t> padding_interior(input_rank, 0);
  for (size_t axis = 0; axis < axes.size(); ++axis) {
    int32_t axis_index = axes[axis];
    padding_low[axis_index] = starts[axis];
    padding_interior[axis_index] = std::abs(strides[axis]) - 1;
    int64_t steps =
        std::floor(static_cast<double>(ends[axis] - 1 - starts[axis]) /
                   strides[axis]) +
        1;
    auto true_ends = steps * strides[axis] + starts[axis] - (strides[axis] - 1);
    padding_high[axis_index] = input_shape[axis_index] - true_ends;
  }

  auto padding_value_op = builder::Const(
      gcu_builder, 0.0f, builder::Type(out_grad.GetType().GetPrimitiveType()));
  auto out = builder::Pad(out_grad,
                          padding_value_op,
                          0 /*constant mode*/,
                          padding_low,
                          padding_high,
                          padding_interior);
  return std::make_shared<GcuOp>(out);
}

EQUIVALENCE_TRANS_FUNC_REG(kStridedSlice,
                           INSENSITIVE,
                           StridedSliceEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kStridedSliceGrad,
                           INSENSITIVE,
                           StridedSliceGradEquivalenceTrans);

}  // namespace backend

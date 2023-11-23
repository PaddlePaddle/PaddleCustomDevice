// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {
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

template <typename T, typename Context>
void StridedSliceKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const std::vector<int>& axes,
                        const phi::IntArray& starts,
                        const phi::IntArray& ends,
                        const phi::IntArray& strides,
                        phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "strided_slice", strided_slice);

    std::vector<int64_t> starts_data = starts.GetData();
    std::vector<int64_t> ends_data = ends.GetData();
    std::vector<int64_t> strides_data = strides.GetData();
    const int64_t input_rank = x.dims().size();
    std::vector<int> infer_flags(axes.size(), 1);
    std::vector<int> decrease_axis;

    // out dims calculation
    std::vector<int64_t> out_dims_vector(input_rank, -1);
    StridedSliceOutDims(starts_data,
                        ends_data,
                        strides_data,
                        axes,
                        infer_flags,
                        x.dims(),
                        decrease_axis,
                        out_dims_vector.data(),
                        axes.size(),
                        false);

    // check whether need to reverse (false: stride > 0; true: stride < 0)
    std::vector<int> reverse_vector(axes.size(), 0);
    StridedSliceFunctor(starts_data.data(),
                        ends_data.data(),
                        strides_data.data(),
                        axes.data(),
                        reverse_vector.data(),
                        x.dims(),
                        infer_flags,
                        decrease_axis,
                        axes.size());

    auto tmp_x = x;
    auto tmp_out = EmptyTensor(dev_ctx, out->meta());
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
      tmp_out = EmptyTensor(dev_ctx, phi::DataType::INT32, out->dims());
    }

    tmp_out = slice(dev_ctx,
                    tmp_x,
                    vector_s64(axes),
                    starts_data,
                    ends_data,
                    strides_data,
                    tmp_out);

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
      tmp_out = reverse(dev_ctx, tmp_out, reverse_axis_vector);
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
    if (decrease_axis.size() > 0) {
      tmp_out = reshape(dev_ctx, tmp_out, out_dims_origin);
    }

    if (x.dtype() == phi::DataType::INT64) {
      cast(dev_ctx, tmp_out, phi::DataType::INT64, out);
    } else {
      TensorCopy(dev_ctx, tmp_out, false, out);
    }

    PADDLE_GCU_KERNEL_END("strided_slice", strided_slice);
  } else {
    TensorNameMap input_names;
    input_names["Input"] = {"x"};

    TensorValueMap inputs;
    inputs["Input"] = {const_cast<DenseTensor*>(&x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    std::vector<int> infer_flags(axes.size(), 1);
    std::vector<int> decrease_axis;
    std::vector<int> starts_list = GetIntList(starts.GetData());
    std::vector<int> ends_list = GetIntList(ends.GetData());
    std::vector<int> strides_list = GetIntList(strides.GetData());

    GcuAttributeMap attrs;
    attrs["starts"] = starts_list;
    attrs["ends"] = ends_list;
    attrs["strides"] = strides_list;
    attrs["axes"] = axes;
    attrs["infer_flags"] = infer_flags;
    attrs["decrease_axis"] = decrease_axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "strided_slice",
              dev_ctx);
  }
}

template <typename T, typename Context>
void StridedSliceGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& out_grad,
                            const std::vector<int>& axes,
                            const phi::IntArray& starts,
                            const phi::IntArray& ends,
                            const phi::IntArray& strides,
                            DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  TensorNameMap input_names;
  input_names["Input"] = {"x"};
  input_names[GradVarName("Out")] = {"out_grad"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<DenseTensor*>(&x)};
  inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

  TensorNameMap output_names;
  output_names[GradVarName("X")] = {"x_grad"};

  TensorValueMap outputs;
  outputs[GradVarName("X")] = {x_grad};

  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  std::vector<int> starts_list = GetIntList(starts.GetData());
  std::vector<int> ends_list = GetIntList(ends.GetData());
  std::vector<int> strides_list = GetIntList(strides.GetData());

  GcuAttributeMap attrs;
  attrs["starts"] = starts_list;
  attrs["ends"] = ends_list;
  attrs["strides"] = strides_list;
  attrs["axes"] = axes;
  attrs["infer_flags"] = infer_flags;
  attrs["decrease_axis"] = decrease_axis;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            "strided_slice_grad",
            dev_ctx);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(strided_slice,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(strided_slice_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceGradKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

static void StridedSliceOutDims(const std::vector<int64_t>& starts,
                                const std::vector<int64_t>& ends,
                                const std::vector<int64_t>& strides,
                                const std::vector<int>& axes,
                                const std::vector<int>& infer_flags,
                                const phi::DDim x_dims,
                                const std::vector<int>& decrease_axis,
                                int64_t* out_dims_vector,
                                const size_t size,
                                bool infer_shape) {
  for (int i = 0; i < x_dims.size(); i++) {
    out_dims_vector[i] = x_dims[i];
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

    int64_t axis_size = x_dims[axes_index];
    if (axis_size < 0) {
      continue;
    }

    if (start_index < 0) {
      start_index = start_index + axis_size;
      start_index = std::max<int64_t>(start_index, 0);
    }
    if (end_index < 0) {
      if (!(end_index == -1 && stride_index < 0)) {
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
                          "The start index and end index are invalid for "
                          "their corresponding stride."));

    int64_t left =
        std::max(static_cast<int64_t>(0), std::min(start_index, end_index));
    int64_t right = std::min(axis_size, std::max(start_index, end_index));
    int64_t step = std::abs(stride_index);

    auto out_dims_index = (std::abs(right - left) + step - 1) / step;

    out_dims_vector[axes_index] = out_dims_index;
  }
}

static void ProcessStridedSliceParams(
    const std::vector<int>& axes,
    const phi::DDim& input_dims,
    const std::vector<int64_t>& starts,
    const std::vector<int64_t>& ends,
    const std::vector<int64_t>& strides,
    const std::vector<int>& infer_flags,
    const std::vector<int>& decrease_axis,
    std::vector<int>* starts_indices_vector,
    std::vector<int>* ends_indices_vector,
    std::vector<int>* strides_indices_vector) {
  for (size_t axis = 0; axis < axes.size(); axis++) {
    int64_t start = starts[axis];
    int64_t end = ends[axis];
    int64_t stride = strides[axis];

    int axis_index = axes[axis];
    int64_t dim_size = input_dims[axis_index];

    bool decrease_axis_affect = false;
    if (start == -1 && end == 0 && infer_flags[axis] == -1) {
      auto ret =
          std::find(decrease_axis.begin(), decrease_axis.end(), axis_index);
      if (ret != decrease_axis.end()) {
        decrease_axis_affect = true;
      }
    }

    if (stride < 0) {
      if (start < 0) {
        start = std::max(start, -dim_size);
      } else {
        start = std::min(start, dim_size - 1) - dim_size;
      }
      if (end < 0) {
        end = std::max(end, -dim_size - 1);
      } else {
        end = end - dim_size;
      }
    } else {
      if (start < 0) {
        start = std::max(start, -dim_size) + dim_size;
      } else {
        start = std::min(start, dim_size - 1);
      }
      if (end < 0) {
        end = end + dim_size;
      } else {
        end = std::min(end, dim_size);
      }
    }

    if (decrease_axis_affect) {
      if (stride < 0) {
        end = start - 1;
      } else {
        end = start + 1;
      }
    }

    (*starts_indices_vector)[axis] = static_cast<int>(start);
    (*ends_indices_vector)[axis] = static_cast<int>(end);
    (*strides_indices_vector)[axis] = static_cast<int>(stride);
  }
}

template <typename T, typename Context>
void StridedSliceRawKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const std::vector<int>& axes,
                           const phi::IntArray& starts,
                           const phi::IntArray& ends,
                           const phi::IntArray& strides,
                           const std::vector<int>& infer_flags,
                           const std::vector<int>& decrease_axis,
                           phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA StridedSliceRawKernel";

  auto x_dims = x.dims();

  auto starts_ = starts.GetData();
  auto ends_ = ends.GetData();
  auto strides_ = strides.GetData();

  int x_dims_size = x_dims.size();
  std::vector<int64_t> out_dims_vector(x_dims.size(), -1);

  // out dims calculation
  StridedSliceOutDims(starts_,
                      ends_,
                      strides_,
                      axes,
                      infer_flags,
                      x_dims,
                      decrease_axis,
                      out_dims_vector.data(),
                      axes.size(),
                      false);

  phi::DDim out_dims(phi::make_ddim(out_dims_vector));

  // construct the starts_indices, ends_indices and strides_indices tensor for
  // calling StridedSlice op
  std::vector<int> starts_indices_vector(axes.size(), 0);
  std::vector<int> ends_indices_vector(axes.size(), 0);
  std::vector<int> strides_indices_vector(axes.size(), 0);

  ProcessStridedSliceParams(axes,
                            x_dims,
                            starts_,
                            ends_,
                            strides_,
                            infer_flags,
                            decrease_axis,
                            &starts_indices_vector,
                            &ends_indices_vector,
                            &strides_indices_vector);

  auto out_dims_origin = out_dims;
  if (decrease_axis.size() > 0) {
    std::vector<int64_t> new_out_shape;
    for (size_t i = 0; i < decrease_axis.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          out_dims[decrease_axis[i]],
          1,
          phi::errors::InvalidArgument(
              "the size of decrease dimension should be 1, but received %d.",
              out_dims[decrease_axis[i]]));
      out_dims_origin[decrease_axis[i]] = 0;
    }

    for (int i = 0; i < out_dims_origin.size(); ++i) {
      if (out_dims_origin[i] != 0) {
        new_out_shape.push_back(out_dims_origin[i]);
      }
    }
    if (new_out_shape.size() == 0) {
      new_out_shape.push_back(1);
    }

    out_dims_origin = phi::make_ddim(new_out_shape);
  }

  out->Resize(out_dims_origin);
  dev_ctx.template Alloc<T>(out);

  std::vector<int64_t> decrease_axis_v(decrease_axis.begin(),
                                       decrease_axis.end());

  sdaa_ops::doSliceTensor(dev_ctx,
                          x,
                          axes,
                          starts_indices_vector,
                          ends_indices_vector,
                          strides_indices_vector,
                          decrease_axis_v,
                          out);
}

template <typename T, typename Context>
void StridedSliceKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const std::vector<int>& axes,
                        const phi::IntArray& starts,
                        const phi::IntArray& ends,
                        const phi::IntArray& strides,
                        phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA StridedSliceKernel";
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  custom_kernel::StridedSliceRawKernel<T, Context>(
      dev_ctx, x, axes, starts, ends, strides, infer_flags, decrease_axis, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(strided_slice_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceRawKernel,
                          bool,
                          float,
                          phi::dtype::float16,
                          double,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(strided_slice,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceKernel,
                          bool,
                          float,
                          phi::dtype::float16,
                          double,
                          int32_t,
                          int64_t) {}

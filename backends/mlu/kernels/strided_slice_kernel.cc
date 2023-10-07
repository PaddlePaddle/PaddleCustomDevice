// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

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

    (*starts_indices_vector)[axis_index] = static_cast<int>(start);
    (*ends_indices_vector)[axis_index] = static_cast<int>(end);
    (*strides_indices_vector)[axis_index] = static_cast<int>(stride);
  }
}

template <typename T, typename Context, size_t D>
void StridedSliceCompute(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const std::vector<int>& axes,
                         const phi::IntArray& starts_array,
                         const phi::IntArray& ends_array,
                         const phi::IntArray& strides_array,
                         const std::vector<int>& infer_flags,
                         const std::vector<int>& decrease_axis,
                         phi::DenseTensor* out) {
  auto in_dims = x.dims();
  auto starts = starts_array.GetData();
  auto ends = ends_array.GetData();
  auto strides = strides_array.GetData();

  // out dims calculation
  std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
  custom_kernel::StridedSliceOutDims(starts,
                                     ends,
                                     strides,
                                     axes,
                                     infer_flags,
                                     in_dims,
                                     decrease_axis,
                                     out_dims_vector.data(),
                                     axes.size(),
                                     false);
  phi::DDim out_dims(phi::make_ddim(out_dims_vector));

  // construct the starts_indices, ends_indices and strides_indices tensor for
  // calling StridedSlice op
  std::vector<int> starts_indices_vector(D, 0);
  std::vector<int> ends_indices_vector(out_dims_vector.begin(),
                                       out_dims_vector.end());
  std::vector<int> strides_indices_vector(D, 1);

  custom_kernel::ProcessStridedSliceParams(axes,
                                           in_dims,
                                           starts,
                                           ends,
                                           strides,
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

  MLUCnnlTensorDesc in_desc(x);
  MLUCnnlTensorDesc out_desc(
      out_dims_vector.size(), out_dims_vector.data(), ToCnnlDataType<T>());
  MLUCnnl::StridedSlice(dev_ctx,
                        starts_indices_vector.data(),
                        ends_indices_vector.data(),
                        strides_indices_vector.data(),
                        in_desc.get(),
                        GetBasePtr(&x),
                        out_desc.get(),
                        GetBasePtr(out));
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
  int rank = x.dims().size();
  switch (rank) {
    case 1:
      custom_kernel::StridedSliceCompute<T, Context, 1>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 2:
      custom_kernel::StridedSliceCompute<T, Context, 2>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 3:
      custom_kernel::StridedSliceCompute<T, Context, 3>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 4:
      custom_kernel::StridedSliceCompute<T, Context, 4>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 5:
      custom_kernel::StridedSliceCompute<T, Context, 5>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 6:
      custom_kernel::StridedSliceCompute<T, Context, 6>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 7:
      custom_kernel::StridedSliceCompute<T, Context, 7>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    case 8:
      custom_kernel::StridedSliceCompute<T, Context, 8>(dev_ctx,
                                                        x,
                                                        axes,
                                                        starts,
                                                        ends,
                                                        strides,
                                                        infer_flags,
                                                        decrease_axis,
                                                        out);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The rank of input is supported up to 8."));
      break;
  }
}

template <typename T, typename Context, size_t D>
void StridedSliceGradCompute(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& out_grad,
                             const std::vector<int>& axes,
                             const phi::IntArray& starts_array,
                             const phi::IntArray& ends_array,
                             const phi::IntArray& strides_array,
                             const std::vector<int>& infer_flags,
                             const std::vector<int>& decrease_axis,
                             phi::DenseTensor* x_grad) {
  auto input_dims = x.dims();
  x_grad->Resize(input_dims);
  dev_ctx.template Alloc<T>(x_grad);

  auto starts = starts_array.GetData();
  auto ends = ends_array.GetData();
  auto strides = strides_array.GetData();

  std::vector<int64_t> out_dims_vector(input_dims.size(), -1);
  custom_kernel::StridedSliceOutDims(starts,
                                     ends,
                                     strides,
                                     axes,
                                     infer_flags,
                                     input_dims,
                                     decrease_axis,
                                     out_dims_vector.data(),
                                     axes.size(),
                                     false);

  std::vector<int> starts_indices_vector(D, 0);
  std::vector<int> ends_indices_vector(out_dims_vector.begin(),
                                       out_dims_vector.end());
  std::vector<int> strides_indices_vector(D, 1);

  custom_kernel::ProcessStridedSliceParams(axes,
                                           input_dims,
                                           starts,
                                           ends,
                                           strides,
                                           infer_flags,
                                           decrease_axis,
                                           &starts_indices_vector,
                                           &ends_indices_vector,
                                           &strides_indices_vector);

  MLUCnnlTensorDesc out_grad_desc(
      out_dims_vector.size(), out_dims_vector.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc x_grad_desc(x);
  MLUCnnl::StridedSliceGrad(dev_ctx,
                            starts_indices_vector.data(),
                            ends_indices_vector.data(),
                            strides_indices_vector.data(),
                            out_grad_desc.get(),
                            GetBasePtr(&out_grad),
                            x_grad_desc.get(),
                            GetBasePtr(x_grad));
}

template <typename T, typename Context>
void StridedSliceRawGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               const phi::DenseTensor& out_grad,
                               const std::vector<int>& axes,
                               const phi::IntArray& starts,
                               const phi::IntArray& ends,
                               const phi::IntArray& strides,
                               const std::vector<int>& infer_flags,
                               const std::vector<int>& decrease_axis,
                               phi::DenseTensor* x_grad) {
  int rank = x.dims().size();

  switch (rank) {
    case 1:
      custom_kernel::StridedSliceGradCompute<T, Context, 1>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 2:
      custom_kernel::StridedSliceGradCompute<T, Context, 2>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 3:
      custom_kernel::StridedSliceGradCompute<T, Context, 3>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 4:
      custom_kernel::StridedSliceGradCompute<T, Context, 4>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 5:
      custom_kernel::StridedSliceGradCompute<T, Context, 5>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 6:
      custom_kernel::StridedSliceGradCompute<T, Context, 6>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 7:
      custom_kernel::StridedSliceGradCompute<T, Context, 7>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    case 8:
      custom_kernel::StridedSliceGradCompute<T, Context, 8>(dev_ctx,
                                                            x,
                                                            out_grad,
                                                            axes,
                                                            starts,
                                                            ends,
                                                            strides,
                                                            infer_flags,
                                                            decrease_axis,
                                                            x_grad);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The rank of input is supported up to 8."));
      break;
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
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  custom_kernel::StridedSliceRawKernel<T, Context>(
      dev_ctx, x, axes, starts, ends, strides, infer_flags, decrease_axis, out);
}

template <typename T, typename Context>
void StridedSliceGradKernel(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& out_grad,
                            const std::vector<int>& axes,
                            const phi::IntArray& starts,
                            const phi::IntArray& ends,
                            const phi::IntArray& strides,
                            phi::DenseTensor* x_grad) {
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  custom_kernel::StridedSliceRawGradKernel<T, Context>(dev_ctx,
                                                       x,
                                                       out_grad,
                                                       axes,
                                                       starts,
                                                       ends,
                                                       strides,
                                                       infer_flags,
                                                       decrease_axis,
                                                       x_grad);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(strided_slice_raw,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceRawKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(strided_slice_raw_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceRawGradKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(strided_slice,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(strided_slice_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::StridedSliceGradKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

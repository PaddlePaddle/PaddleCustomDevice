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

void UpdateAttr(const phi::DDim& in_dims,
                const std::vector<int> axes,
                const std::vector<int> starts,
                const std::vector<int> ends,
                std::vector<int>* offsets,
                std::vector<int>* size) {
  int cnt = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    int start = 0;
    int end = in_dims[i];
    // NOTE(zhiqiu): Becareful that cnt may > axes.size() and result in
    // overflow.
    int axis = cnt < static_cast<int>(axes.size()) ? axes[cnt] : -1;
    if (axis == i) {
      start = starts[cnt];
      if (start < 0) {
        start = (start + in_dims[i]);
      }
      start = std::max(start, static_cast<int>(0));
      end = ends[cnt];
      if (end < 0) {
        end = (end + in_dims[i]);
      }
      end = std::min(end, static_cast<int>(in_dims[i]));
      cnt++;
    }

    (*offsets)[i] = start;
    (*size)[i] = end - start;
  }
}

template <typename T = int64_t>
inline void CheckAndUpdateSliceAttrs(const phi::DDim in_dims,
                                     const std::vector<T>& axes,
                                     std::vector<T>* starts,
                                     std::vector<T>* ends,
                                     std::vector<int64_t>* steps = nullptr,
                                     std::vector<T>* infer_flags = nullptr) {
  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    PADDLE_ENFORCE_LT(
        axis,
        in_dims.size(),
        phi::errors::InvalidArgument(
            "The axis value should be less than the rank of input, "
            "but received axes[%d] = %d, rank of input is %d.",
            i,
            axis,
            in_dims.size()));

    if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
      continue;
    }

    T dim_value = in_dims[axis];

    if (dim_value > 0) {
      T step = steps == nullptr ? 1 : (*steps)[i];
      PADDLE_ENFORCE_NE(
          step,
          0,
          phi::errors::InvalidArgument(
              "Step should not be 0, but received step = %d.", step));

      T start = (*starts)[i] < 0 ? ((*starts)[i] + dim_value) : (*starts)[i];
      start = std::max(start, static_cast<T>(0));

      T end =
          0 < step && (*ends)[i] < 0 ? ((*ends)[i] + dim_value) : (*ends)[i];
      end = std::min(end, dim_value);

      if (step > 0) {
        start = std::min(start, dim_value);
        end = std::max(end, static_cast<T>(0));
        PADDLE_ENFORCE_GE(
            end,
            start,
            phi::errors::InvalidArgument(
                "When step > 0, end should be greater than start, but "
                "received end = %d, start = %d.",
                end,
                start));
      } else {
        // NOTE(liym27): When step < 0, start should less and equal to
        // dim_value-1
        // "end is -1" means contain the 0-th element of this axis.
        start = std::min(start, dim_value - 1);
        if (end < -1) {
          end += dim_value;
        }
        end = std::max(end, static_cast<T>(-1));
        PADDLE_ENFORCE_GE(
            start,
            end,
            phi::errors::InvalidArgument(
                "When step < 0, start should be greater than end, but "
                "received start = %d, end = %d.",
                start,
                end));
      }

      (*starts)[i] = start;
      (*ends)[i] = end;
    } else if (dim_value == 0) {
      (*starts)[i] = 0;
      (*ends)[i] = 0;
    }
  }
}

template <typename T = int64_t>
inline phi::DDim GetSliceDims(const phi::DDim in_dims,
                              const std::vector<T>& axes,
                              const std::vector<T>& starts,
                              const std::vector<T>& ends,
                              std::vector<T>* steps = nullptr,
                              std::vector<T>* infer_flags = nullptr) {
  phi::DDim slice_dims(in_dims);

  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
      slice_dims[axis] = -1;
      continue;
    }

    T start = starts[i];
    T end = ends[i];
    T step = steps == nullptr ? 1 : (*steps)[i];

    if (step > 0) {
      slice_dims[axis] = (end - start + step - 1) / step;
    } else {
      slice_dims[axis] = (end - start + step + 1) / step;
    }
  }
  return slice_dims;
}

template <typename T = int64_t>
inline phi::DDim GetDecreasedDims(const phi::DDim slice_dims,
                                  const std::vector<T>& decrease_axes,
                                  std::vector<T>* infer_flags = nullptr) {
  phi::DDim decreased_dims(slice_dims);
  std::vector<uint8_t> decrease_flag(slice_dims.size(), 0);
  if (decrease_axes.size() > 0) {
    for (size_t i = 0; i < decrease_axes.size(); ++i) {
      T axis = decrease_axes[i];
      decrease_flag[axis] = 1;
      if (infer_flags && (*infer_flags)[i] != -1) {
        PADDLE_ENFORCE_EQ(decreased_dims[axis],
                          1,
                          phi::errors::InvalidArgument(
                              "Decrease dim should be 1, but now received %d",
                              decreased_dims[axis]));
      }
    }

    std::vector<T> new_shape;
    for (int i = 0; i < decreased_dims.size(); ++i) {
      if (decrease_flag[i] == 0) {
        new_shape.push_back(decreased_dims[i]);
      }
    }

    // NOTE(liym27): Paddle does not support that the rank of Tensor is 0, and
    // uses [1] instead.
    if (new_shape.size() == 0) {
      new_shape.push_back(1);
    }

    decreased_dims = phi::make_ddim(new_shape);
  }
  return decreased_dims;
}

template <typename T, typename Context>
void SliceRawKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const std::vector<int64_t>& axes_t,
                    const phi::IntArray& starts_array,
                    const phi::IntArray& ends_array,
                    const std::vector<int64_t>& infer_flags,
                    const std::vector<int64_t>& decrease_axis,
                    phi::DenseTensor* out) {
  std::vector<int> axes(axes_t.begin(), axes_t.end());
  auto starts_int = starts_array.GetData();
  auto ends_int = ends_array.GetData();
  std::vector<int> starts(starts_int.begin(), starts_int.end());
  std::vector<int> ends(ends_int.begin(), ends_int.end());

  PADDLE_ENFORCE_EQ(
      starts.size(),
      axes.size(),
      phi::errors::InvalidArgument(
          "The size of starts must be equal to the size of axes."));
  PADDLE_ENFORCE_EQ(ends.size(),
                    axes.size(),
                    phi::errors::InvalidArgument(
                        "The size of ends must be equal to the size of axes."));

  const auto& in_dims = x.dims();
  auto slice_dims = out->dims();
  bool reset_slice_dims = false;
  // Infer output dims
  for (size_t i = 0; i < axes.size(); ++i) {
    // when start == -1 && end == start+1
    if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
      auto ret = std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
      if (ret != decrease_axis.end()) {
        ends[i] = in_dims[axes[i]];
      }
    }
  }

  custom_kernel::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
  slice_dims = custom_kernel::GetSliceDims<int>(
      in_dims, axes, starts, ends, nullptr, nullptr);
  reset_slice_dims = true;
  auto out_dims = custom_kernel::GetDecreasedDims(slice_dims, decrease_axis);

  out->Resize(out_dims);

  if (slice_dims.size() != in_dims.size() && !reset_slice_dims) {
    custom_kernel::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
    slice_dims = custom_kernel::GetSliceDims<int>(
        in_dims, axes, starts, ends, nullptr, nullptr);
  }

  int in_dim_size = x.dims().size();
  if (static_cast<int>(axes.size()) != in_dim_size) {
    std::vector<int> tmp_starts(in_dim_size, 0);
    const auto& in_dims_vec = phi::vectorize(x.dims());
    std::vector<int> tmp_ends(in_dims_vec.begin(), in_dims_vec.end());
    for (size_t i = 0; i < axes.size(); ++i) {
      tmp_starts[axes[i]] = starts[i];
      tmp_ends[axes[i]] = ends[i];
    }
    starts.swap(tmp_starts);
    ends.swap(tmp_ends);
  }
  std::vector<int> strides(in_dim_size, 1);

  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc out_desc(slice_dims.size(),
                             phi::vectorize(slice_dims).data(),
                             ToCnnlDataType<T>());
  MLUCnnl::StridedSlice(dev_ctx,
                        starts.data(),
                        ends.data(),
                        strides.data(),
                        input_desc.get(),
                        GetBasePtr(&x),
                        out_desc.get(),
                        GetBasePtr(out));
}

template <typename T, typename Context>
void SliceGradRawKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& out_grad,
                        const std::vector<int64_t>& axes_t,
                        const phi::IntArray& starts_array,
                        const phi::IntArray& ends_array,
                        const std::vector<int64_t>& infer_flags,
                        const std::vector<int64_t>& decrease_axis,
                        phi::DenseTensor* x_grad) {
  std::vector<int> axes(axes_t.begin(), axes_t.end());
  auto starts_int = starts_array.GetData();
  auto ends_int = ends_array.GetData();

  std::vector<int> starts(starts_int.begin(), starts_int.end());
  std::vector<int> ends(ends_int.begin(), ends_int.end());

  const auto& in_dims = x.dims();
  auto slice_dims = out_grad.dims();
  if (slice_dims.size() != in_dims.size()) {
    custom_kernel::CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
    slice_dims = custom_kernel::GetSliceDims<int>(
        in_dims, axes, starts, ends, nullptr, nullptr);
  }

  int in_dim_size = x.dims().size();
  if (static_cast<int>(axes.size()) != in_dim_size) {
    std::vector<int> tmp_starts(in_dim_size, 0);
    const auto& in_dims_vec = phi::vectorize(x.dims());
    std::vector<int> tmp_ends(in_dims_vec.begin(), in_dims_vec.end());
    for (size_t i = 0; i < axes.size(); ++i) {
      tmp_starts[axes[i]] = starts[i];
      tmp_ends[axes[i]] = ends[i];
    }
    starts.swap(tmp_starts);
    ends.swap(tmp_ends);
  }
  std::vector<int> strides(in_dim_size, 1);

  dev_ctx.template Alloc<T>(x_grad);

  MLUCnnlTensorDesc dout_desc(slice_dims.size(),
                              phi::vectorize(slice_dims).data(),
                              ToCnnlDataType<T>());
  MLUCnnlTensorDesc x_grad_desc(*x_grad);
  MLUCnnl::StridedSliceGrad(dev_ctx,
                            starts.data(),
                            ends.data(),
                            strides.data(),
                            dout_desc.get(),
                            GetBasePtr(&out_grad),
                            x_grad_desc.get(),
                            GetBasePtr(x_grad));
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(slice,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SliceRawKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int32_t,
                          int64_t,
                          bool) {}
PD_REGISTER_PLUGIN_KERNEL(slice_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SliceGradRawKernel,
                          phi::dtype::float16,
                          float,
                          double,
                          int32_t,
                          int64_t,
                          bool) {}

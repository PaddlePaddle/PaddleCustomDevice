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
void SetValueKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::IntArray& starts,
                    const phi::IntArray& ends,
                    const phi::IntArray& steps,
                    const std::vector<int64_t>& axes,
                    const std::vector<int64_t>& decrease_axes,
                    const std::vector<int64_t>& none_axes,
                    const std::vector<int64_t>& shape,
                    const std::vector<phi::Scalar>& values,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();

  auto in_dims = x.dims();
  custom_kernel::CheckAndUpdateSliceAttrs(
      in_dims, axes, &starts_local, &ends_local, &steps_local);
  auto slice_dims = custom_kernel::GetSliceDims(
      in_dims, axes, starts_local, ends_local, &steps_local);
  auto decrease_slice_dims =
      custom_kernel::GetDecreasedDims(slice_dims, decrease_axes);

  auto slice_dims_for_assign = decrease_slice_dims;
  if (!none_axes.empty()) {
    std::vector<int64_t> slice_dims_with_none;
    size_t none_axes_cur = 0, decrease_axes_cur = 0;
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
  int in_size = in_dims.size();
  int starts_indices[in_size] = {0};
  int ends_indices[in_size] = {0};
  int strides_indices[in_size] = {0};

  for (int i = 0; i < in_dims.size(); ++i) {
    starts_indices[i] = 0;
    ends_indices[i] = static_cast<int>(slice_dims[i]);
    strides_indices[i] = 1;
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int axis_index = axes[i];
    starts_indices[axis_index] = static_cast<int>(starts_local[i]);
    ends_indices[axis_index] = static_cast<int>(ends_local[i]);
    strides_indices[axis_index] = static_cast<int>(steps_local[i]);
  }

  std::vector<T> assgin_values;
  assgin_values.reserve(values.size());
  for (const auto& val : values) {
    assgin_values.push_back(val.to<T>());
  }
  phi::DenseTensor value_t;
  value_t.Resize(phi::make_ddim(shape));
  custom_kernel::TensorFromVector(dev_ctx, assgin_values, dev_ctx, &value_t);
  dev_ctx.Wait();
  value_t.Resize(phi::make_ddim(shape));

  phi::DenseTensor value_temp;
  if (slice_dims_for_assign == value_t.dims()) {
    value_temp = value_t;
  } else {
    value_temp.Resize(slice_dims_for_assign);
    dev_ctx.template Alloc<T>(&value_temp);
    MLUCnnlTensorDesc value_t_desc(value_t);
    MLUCnnlTensorDesc value_temp_desc(value_temp);
    MLUCnnl::BroadcastTo(dev_ctx,
                         value_t_desc.get(),
                         GetBasePtr(&value_t),
                         value_temp_desc.get(),
                         GetBasePtr(&value_temp));
  }

  int64_t input_numel = phi::product(in_dims);
  int64_t value_numel = phi::product(value_temp.dims());
  phi::DenseTensor in_temp, out_temp, val_temp, index_out;
  int64_t stride_step = phi::product(in_dims);
  std::vector<int64_t> index_indices(stride_step);
  std::iota(index_indices.begin(), index_indices.end(), 0);
  phi::DenseTensor index_temp;
  in_temp = x;
  val_temp = value_temp;
  custom_kernel::TensorFromVector(dev_ctx, index_indices, dev_ctx, &index_temp);
  dev_ctx.Wait();
  index_temp.Resize(in_dims);
  auto index_dims = in_dims;
  for (int i = 0; i < in_dims.size(); ++i) {
    if (starts_indices[i] < 0 || ends_indices[i] < 0) {
      starts_indices[i] -= in_dims[i];
      ends_indices[i] -= in_dims[i];
    }
    if (strides_indices[i] > 0)
      index_dims[i] =
          static_cast<int>((ends_indices[i] - starts_indices[i] - 1) /
                           strides_indices[i]) +
          1;
    else
      index_dims[i] =
          static_cast<int>((ends_indices[i] - starts_indices[i] + 1) /
                           strides_indices[i]) +
          1;
  }
  auto new_in_dims = phi::make_ddim({input_numel});
  auto new_val_dims = phi::make_ddim({value_numel});
  in_temp.Resize(new_in_dims);
  val_temp.Resize(new_val_dims);
  index_out.Resize(index_dims);
  dev_ctx.template Alloc<int64_t>(&index_out);
  cnnlScatterRefMode_t mode = CNNL_SCATTERREF_UPDATE;
  MLUCnnlTensorDesc x_desc(in_temp);
  MLUCnnlTensorDesc indices_desc(index_temp);
  MLUCnnlTensorDesc indices_out_desc(index_out);
  MLUCnnlTensorDesc updates_desc(val_temp);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::StridedSlice(dev_ctx,
                        starts_indices,
                        ends_indices,
                        strides_indices,
                        indices_desc.get(),
                        GetBasePtr(&index_temp),
                        indices_out_desc.get(),
                        GetBasePtr(&index_out));
  PADDLE_ENFORCE_EQ(
      static_cast<int64_t>(phi::product(index_out.dims())),
      phi::product(slice_dims_for_assign),
      phi::errors::InvalidArgument(
          "OP(set_value) error index indices and value update not match "));
  Tensor index_final;
  index_final = index_out;
  int64_t indices_numel = phi::product(index_dims);
  auto new_index_dims = phi::make_ddim({indices_numel});
  index_final.Resize(new_index_dims);
  MLUCnnlTensorDesc indices_final_desc(index_final);
  MLUCnnl::ScatterRefFunctor(dev_ctx,
                             x_desc.get(),
                             GetBasePtr(&in_temp),
                             updates_desc.get(),
                             GetBasePtr(&val_temp),
                             indices_final_desc.get(),
                             GetBasePtr(&index_final),
                             mode);
  in_temp.Resize(in_dims);
  TensorCopy(dev_ctx, in_temp, false, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    set_value, mlu, ALL_LAYOUT, custom_kernel::SetValueKernel, float, int) {}

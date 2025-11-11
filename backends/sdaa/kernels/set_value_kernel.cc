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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

// This function is used to check if the value_dims size is less than
// decrease_slice_dims size.
inline void CheckIsDimsMatch(const phi::DenseTensor& input,
                             phi::DenseTensor* output) {
  std::vector<int64_t> input_dims = phi::vectorize(input.dims());
  std::vector<int64_t> out_dims = phi::vectorize(output->dims());

  if (input_dims.size() <= out_dims.size()) return;

  std::vector<int64_t> out_temp_dims1(out_dims), out_temp_dims2(out_dims);

  int dims_diff = input_dims.size() - out_dims.size();

  out_temp_dims1.insert(out_temp_dims1.begin(), dims_diff, 1);
  out_temp_dims2.insert(out_temp_dims2.end(), dims_diff, 1);

  if (input_dims == out_temp_dims1) {
    output->Resize(phi::make_ddim(out_temp_dims1));
    return;
  }
  if (input_dims == out_temp_dims2) {
    output->Resize(phi::make_ddim(out_temp_dims2));
    return;
  }
  PADDLE_THROW(phi::errors::InvalidArgument(
      "The shape of tensor assigned value must match the shape "
      "of target shape: %d, but now shape is %d.",
      output->dims().to_str(),
      input.dims().to_str()));
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

    if (in_dims[axis] == -1) {
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
    if (isEnvEnable("FLAGS_set_to_1d") && new_shape.size() == 0) {
      // NOTE(zoooo0820): Hack procssing to 1-D, when axes decrease to 0-D in
      // slice. This will remove in release 2.6.
      new_shape.push_back(1);
    }
    decreased_dims = phi::make_ddim(new_shape);
  }
  return decreased_dims;
}

template <typename T, typename Context>
void SetTensorValueKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& value,
                          const phi::IntArray& starts,
                          const phi::IntArray& ends,
                          const phi::IntArray& steps,
                          const std::vector<int64_t>& axes,
                          const std::vector<int64_t>& decrease_axes,
                          const std::vector<int64_t>& none_axes,
                          phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA SetTensorValueKernel";

  dev_ctx.template Alloc<T>(out);

  if ((x.dims() == value.dims()) && (axes.size() == 0)) {
    phi::Copy(dev_ctx, value, out->place(), false, out);
    return;
  }

  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();

  auto in_dims = x.dims();
  CheckAndUpdateSliceAttrs(
      in_dims, axes, &starts_local, &ends_local, &steps_local);
  auto slice_dims =
      GetSliceDims(in_dims, axes, starts_local, ends_local, &steps_local);
  auto decrease_slice_dims =
      GetDecreasedDims<int64_t>(slice_dims, decrease_axes);

  auto slice_dims_for_assign = decrease_slice_dims;

  if (!none_axes.empty()) {
    std::vector<int64_t> slice_dims_with_none;
    size_t none_axes_cur = 0, decrease_axes_cur = 0;
    for (int i = 0; i < slice_dims.size(); i++) {
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

  for (int i = 0; i < in_dims.size(); i++) {
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

  phi::DenseTensor value_temp;
  if (slice_dims_for_assign == value.dims()) {
    value_temp = value;
  } else {
    // shape broadcast to slice_dims_for_assign
    value_temp.Resize(slice_dims_for_assign);
    dev_ctx.template Alloc<T>(&value_temp);
    custom_kernel::CheckIsDimsMatch(value, &value_temp);

    sdaa_ops::doExpandTensor(dev_ctx, value, &value_temp);
  }

  int64_t stride_step = x.numel();
  std::vector<T> index_indices(stride_step);
  std::iota(index_indices.begin(), index_indices.end(), 0);

  phi::DenseTensor in_temp, val_temp, index_out, index_temp;
  in_temp = x;
  val_temp = value_temp;
  index_temp.Resize(in_dims);
  TensorFromVector(dev_ctx, index_indices, dev_ctx, &index_temp);

  index_temp.Resize(in_dims);

  auto index_dims = in_dims;
  for (int i = 0; i < in_dims.size(); i++) {
    if (starts_indices[i] < 0 || ends_indices[i] < 0) {
      starts_indices[i] -= in_dims[i];
      ends_indices[i] -= in_dims[i];
    }
    if (strides_indices[i] > 0) {
      index_dims[i] =
          static_cast<int>((ends_indices[i] - starts_indices[i] - 1) /
                           strides_indices[i]) +
          1;
    } else {
      index_dims[i] =
          static_cast<int>((ends_indices[i] - starts_indices[i] + 1) /
                           strides_indices[i]) +
          1;
    }
  }

  auto new_in_dims = phi::make_ddim({x.numel()});
  auto new_val_dims = phi::make_ddim({val_temp.numel()});

  in_temp.Resize(new_in_dims);
  val_temp.Resize(new_val_dims);
  index_out.Resize(index_dims);
  dev_ctx.template Alloc<T>(&index_out);

  std::vector<int> strided_slice_axes(axes.begin(), axes.end());
  std::vector<int> strided_slice_start(starts_local.begin(),
                                       starts_local.end());
  std::vector<int> strided_slice_end(ends_local.begin(), ends_local.end());
  std::vector<int> strided_slice_step(steps_local.begin(), steps_local.end());

  sdaa_ops::doSliceTensor(dev_ctx,
                          index_temp,
                          strided_slice_axes,
                          strided_slice_start,
                          strided_slice_end,
                          strided_slice_step,
                          decrease_axes,
                          &index_out);

  PADDLE_ENFORCE_EQ(
      static_cast<int64_t>(phi::product(index_out.dims())),
      phi::product(slice_dims_for_assign),
      phi::errors::InvalidArgument(
          "OP(set_value) error index indices and value update not match "));

  phi::DenseTensor index_final(index_out);
  int64_t indices_numel = phi::product(index_dims);
  auto new_index_dims = phi::make_ddim({indices_numel});
  index_final.Resize(new_index_dims);

  phi::DenseTensor in_temp_non_int, val_temp_non_int, out_non_int,
      index_final_int32;
  if (x.dtype() == DataType::INT64) {
    index_final_int32 = index_final;

    in_temp_non_int.Resize(in_temp.dims());
    dev_ctx.template Alloc<float>(&in_temp_non_int);
    val_temp_non_int.Resize(val_temp.dims());
    dev_ctx.template Alloc<float>(&val_temp_non_int);
    out_non_int.Resize(out->dims());
    dev_ctx.template Alloc<float>(&out_non_int);
    sdaa_ops::doCastTensor(dev_ctx, in_temp, &in_temp_non_int);
    sdaa_ops::doCastTensor(dev_ctx, val_temp, &val_temp_non_int);
  } else {
    index_final_int32.Resize(index_final.dims());
    dev_ctx.template Alloc<int32_t>(&index_final_int32);
    sdaa_ops::doCastTensor(dev_ctx, index_final, &index_final_int32);

    in_temp_non_int = in_temp;
    val_temp_non_int = val_temp;
    out_non_int = *out;
  }

  sdaa_ops::doScatterTensor(dev_ctx,
                            in_temp_non_int,
                            index_final_int32,
                            val_temp_non_int,
                            true,
                            &out_non_int);

  if (x.dtype() == DataType::INT64) {
    sdaa_ops::doCastTensor(dev_ctx, out_non_int, out);
  }
}

}  // namespace custom_kernel

// TODO(zhanggq): unregistering set_value_with_tensor due to tecodnnScatter API
// has poor performance on big shape.
// PD_REGISTER_PLUGIN_KERNEL(set_value_with_tensor,
//                           sdaa,
//                           ALL_LAYOUT,
//                           custom_kernel::SetTensorValueKernel,
//                           int64_t,
//                           float) {}

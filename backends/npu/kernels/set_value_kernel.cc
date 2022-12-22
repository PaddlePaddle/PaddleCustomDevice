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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/slice_utils.h"

namespace custom_kernel {

template <typename T, typename Context>
void SetTensorValueNPUKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& value,
                             const phi::IntArray& starts,
                             const phi::IntArray& ends,
                             const phi::IntArray& steps,
                             const std::vector<int64_t>& axes,
                             const std::vector<int64_t>& decrease_axes,
                             const std::vector<int64_t>& none_axes,
                             phi::DenseTensor* out) {
  auto in_dims = x.dims();
  std::vector<int64_t> starts_local = starts.GetData();
  std::vector<int64_t> ends_local = ends.GetData();
  std::vector<int64_t> steps_local = steps.GetData();
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

  TensorCopy(dev_ctx, x, false, out);

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
    starts_indices[axis_index] = starts_local[i];
    ends_indices[axis_index] = ends_local[i];
    strides_indices[axis_index] = steps_local[i];
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

  PADDLE_ENFORCE_EQ(
      static_cast<int64_t>(index_indices.size()),
      phi::product(slice_dims_for_assign),
      phi::errors::InvalidArgument(
          "OP(set_value) error index indices and value update not match "));

  auto stream = dev_ctx.stream();

  phi::DenseTensor value_temp;
  if (slice_dims_for_assign == value.dims()) {
    value_temp = value;
  } else {
    value_temp.Resize(slice_dims_for_assign);
    dev_ctx.template Alloc<T>(&value_temp);
    NpuOpRunner runner_brd;
    runner_brd.SetType("BroadcastTo")
        .AddInput(value)
        .AddInput(dev_ctx, phi::vectorize(slice_dims_for_assign))
        .AddOutput(value_temp)
        .Run(stream);
  }

  int64_t input_numel = phi::product(in_dims);
  int64_t index_numel = index_indices.size();

  phi::DenseTensor in_temp(x), out_temp(*out), val_temp(value_temp);
  in_temp.Resize(phi::make_ddim({input_numel}));
  out_temp.Resize(phi::make_ddim({input_numel}));
  val_temp.Resize(phi::make_ddim({index_numel}));

  NpuOpRunner runner;
  runner.SetType("ScatterUpdate")
      .AddInput(in_temp)
      .AddInput(dev_ctx, std::move(index_indices))
      .AddInput(val_temp)
      .AddOutput(out_temp)
      .AddAttrs({{"use_locking", false}})
      .Run(stream);
}

template <typename T, typename Context>
void SetValueNPUKernel(const Context& dev_ctx,
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
  std::vector<T> assgin_values;
  assgin_values.reserve(values.size());
  for (const auto& val : values) {
    assgin_values.push_back(val.to<T>());
  }
  phi::DenseTensor value_tensor;
  value_tensor.Resize(phi::make_ddim(shape));
  custom_kernel::TensorFromVector(
      dev_ctx, assgin_values, dev_ctx, &value_tensor);
  value_tensor.Resize(phi::make_ddim(shape));

  custom_kernel::SetTensorValueNPUKernel<T, Context>(dev_ctx,
                                                     x,
                                                     value_tensor,
                                                     starts,
                                                     ends,
                                                     steps,
                                                     axes,
                                                     decrease_axes,
                                                     none_axes,
                                                     out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(set_value,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SetValueNPUKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(set_value_with_tensor,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::SetTensorValueNPUKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool) {}

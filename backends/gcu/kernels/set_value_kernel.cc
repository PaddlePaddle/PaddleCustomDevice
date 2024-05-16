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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace custom_kernel {
template <typename T, typename Context>
extern void ScatterKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& index,
                          const phi::DenseTensor& updates,
                          bool overwrite,
                          phi::DenseTensor* out);

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
  PADDLE_GCU_KERNEL_TRACE("set_value_with_tensor");
  dev_ctx.template Alloc<T>(out);

  if (LaunchAOTKernel()) {
    auto in_dims = x.dims();
    const int rank = in_dims.size();

    std::vector<int64_t> starts_local = starts.GetData();
    std::vector<int64_t> ends_local = ends.GetData();
    std::vector<int64_t> steps_local = steps.GetData();
    phi::funcs::CheckAndUpdateSliceAttrs(
        in_dims, axes, &starts_local, &ends_local, &steps_local);
    auto slice_dims = phi::funcs::GetSliceDims(
        in_dims, axes, starts_local, ends_local, &steps_local);
    auto decrease_slice_dims =
        phi::funcs::GetDecreasedDims(slice_dims, decrease_axes);

    auto slice_dims_for_assign = decrease_slice_dims;
    if (!none_axes.empty()) {
      std::vector<int64_t> slice_dims_with_none;

      size_t none_axes_cur = 0;
      size_t decrease_axes_cur = 0;
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

    // starts, ends, starides data processing.
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

    // get index shape.
    int64_t stride_step = phi::product(in_dims);
    std::vector<int32_t, PinnedAllocatorForSTL<int32_t>> index_indices(1, 0);
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

    // create index tensor.
    std::vector<int64_t> index_shape = {
        static_cast<int64_t>(index_indices.size())};
    phi::DenseTensor index_tensor;
    index_tensor.Resize(phi::make_ddim(index_shape));
    TensorFromVector(dev_ctx, index_indices, dev_ctx, &index_tensor);

    // check index size.
    if (index_indices.size() == 0) {
      TensorCopy(dev_ctx, x, false, out);
      return;
    }

    PADDLE_ENFORCE_EQ(
        static_cast<int64_t>(index_indices.size()),
        phi::product(slice_dims_for_assign),
        phi::errors::InvalidArgument(
            "OP(set_value) error index indices and value update not match "));

    auto slice_dims_for_assign_v = phi::vectorize(slice_dims_for_assign);
    PADDLE_ENFORCE_GE(
        slice_dims_for_assign_v.size(),
        value.dims().size(),
        phi::errors::InvalidArgument(
            "OP(set_value) error: the rank of slice_dims_for_assign_v must "
            "larger than or equal the rank of value shape. "));

    // Processing value_tensor data.
    phi::DenseTensor value_tensor(value);
    if (slice_dims_for_assign_v != phi::vectorize(value_tensor.dims()) &&
        value_tensor.numel() == 1) {
      std::vector<int64_t> broadcast_shape = {
          static_cast<int64_t>(index_tensor.numel())};
      value_tensor = Broadcast(dev_ctx, value_tensor, broadcast_shape);
    }

    std::vector<int64_t> reshape_shape = {static_cast<int64_t>(x.numel())};
    phi::DenseTensor reshape_x = ReshapeWithoutCopy(x, reshape_shape);
    phi::DenseTensor reshape_updates =
        ReshapeWithoutCopy(value_tensor, {value_tensor.numel()});
    phi::DenseTensor reshape_out = ReshapeWithoutCopy(*out, reshape_shape);

    custom_kernel::ScatterKernel<T, Context>(
        dev_ctx, reshape_x, index_tensor, reshape_updates, true, &reshape_out);
    *out = ReshapeWithoutCopy(reshape_out, phi::vectorize(in_dims));

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
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
  PADDLE_GCU_KERNEL_TRACE("set_value");
  std::vector<T, PinnedAllocatorForSTL<T>> assgin_values;
  assgin_values.reserve(values.size());
  for (const auto& val : values) {
    assgin_values.push_back(val.to<T>());
  }
  phi::DenseTensor value_tensor;
  value_tensor.Resize(phi::make_ddim(shape));
  TensorFromVector(dev_ctx, assgin_values, dev_ctx, &value_tensor);
  value_tensor.Resize(phi::make_ddim(shape));

  if (LaunchAOTKernel()) {
    custom_kernel::SetTensorValueKernel<T, Context>(dev_ctx,
                                                    x,
                                                    value_tensor,
                                                    starts,
                                                    ends,
                                                    steps,
                                                    axes,
                                                    decrease_axes,
                                                    none_axes,
                                                    out);

  } else {  // kernel impl base on JIT
    dev_ctx.template Alloc<T>(out);
    TensorNameMap input_names;
    input_names["Input"] = {"x"};
    input_names["ValueTensor"] = {"value_tensor"};

    TensorValueMap inputs;
    inputs["Input"] = {const_cast<DenseTensor*>(&x)};
    inputs["ValueTensor"] = {const_cast<DenseTensor*>(&value_tensor)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["starts"] = starts.GetData();
    attrs["ends"] = ends.GetData();
    attrs["steps"] = steps.GetData();
    attrs["axes"] = axes;
    attrs["decrease_axes"] = decrease_axes;
    attrs["none_axes"] = none_axes;
    attrs["shape"] = shape;

    attrs["bool_values"] = std::vector<int>();
    attrs["fp32_values"] = std::vector<float>();
    attrs["int32_values"] = std::vector<int>();
    attrs["int64_values"] = std::vector<int64_t>();
    attrs["fp64_values"] = std::vector<double>();

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "set_value",
              dev_ctx);
  }

  // out->Resize(in_dims);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(set_value,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SetValueKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(set_value_with_tensor,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::SetTensorValueKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool) {}

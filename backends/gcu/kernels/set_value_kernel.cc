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

#include "common/common.h"
#include "common/utils.h"
#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_name_list.h"
#include "kernels/funcs/gcu_op_runner.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"

namespace custom_kernel {

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
  std::vector<T> assgin_values;
  assgin_values.reserve(values.size());
  for (const auto& val : values) {
    assgin_values.push_back(val.to<T>());
  }
  phi::DenseTensor value_tensor;
  value_tensor.Resize(phi::make_ddim(shape));
  TensorFromVector(dev_ctx, assgin_values, dev_ctx, &value_tensor);
  value_tensor.Resize(phi::make_ddim(shape));

  // auto in_dims = x.dims();
  // auto out_dims_arr = phi::vectorize(in_dims);
  // out_dims_arr.push_back(1);
  // out->Resize(phi::make_ddim(out_dims_arr));
  // dev_ctx.template Alloc<T>(out);

  dev_ctx.template Alloc<T>(out);

  if (UseScatterMemory() && x.dtype() != phi::DataType::BOOL) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "set_value", set_value);
    const int rank = x.dims().size();

    auto in_dims = x.dims();
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
    std::vector<int32_t> index_indices(1, 0);
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
      TensorCopy(dev_ctx, x, true, out);
      PADDLE_GCU_KERNEL_END("set_value", set_value);
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
        shape.size(),
        phi::errors::InvalidArgument(
            "OP(set_value) error: the rank of slice_dims_for_assign_v must "
            "larger than or equal the rank of value shape. "));

    // Processing value_tensor data.
    if (slice_dims_for_assign_v != phi::vectorize(value_tensor.dims()) &&
        value_tensor.numel() == 1) {
      std::vector<int64_t> broadcast_shape = {
          static_cast<int64_t>(index_tensor.numel())};
      value_tensor = broadcast_to(dev_ctx, value_tensor, broadcast_shape);
    }

    std::vector<int64_t> reshape_shape = {static_cast<int64_t>(x.numel())};
    auto reshape_x = reshape(dev_ctx, x, reshape_shape);
    std::vector<int64_t> input_sizes = phi::vectorize(reshape_x.dims());
    std::vector<int64_t> updates_sizes = phi::vectorize(value_tensor.dims());
    std::vector<int64_t> index_sizes = phi::vectorize(index_tensor.dims());

    int64_t num_index_dims = 1;
    int64_t num_indices = index_sizes.size();

    // Degenerate case: nothing to update. Return the buffer unchanged.
    if (num_indices == 0) {
      PADDLE_ENFORCE(false,
                     phi::errors::InvalidArgument(
                         "There is an error in the data here !!!"));
    }

    // If any of the indexed dimensions are zero in the buffer,
    // the update cannotsucceed since it updates a slice of size 1.
    for (int64_t i = 0; i < num_index_dims; ++i) {
      PADDLE_ENFORCE(input_sizes.at(i) != 0,
                     phi::errors::InvalidArgument(
                         "Scatter dimension ", i, " is zero !!!"));
    }

    int64_t updates_rank = updates_sizes.size();
    int64_t buffer_rank = input_sizes.size();
    int64_t num_window_dims_in_updates = buffer_rank - num_index_dims;

    std::vector<int64_t> update_window_dims_vec = {};
    if (updates_rank > 0) {
      for (int64_t i = (updates_rank - num_window_dims_in_updates);
           i < updates_rank;
           ++i) {
        update_window_dims_vec.push_back(i);
      }
    }

    std::vector<int64_t> inserted_window_dims_vec = {};
    for (int64_t i = 0; i < num_index_dims; ++i) {
      inserted_window_dims_vec.push_back(i);
    }

    hlir::Metadata scatter_dimension_numbers;
    auto dims =
        HlirShape(inserted_window_dims_vec,
                  {static_cast<int64_t>(inserted_window_dims_vec.size())});
    auto update_window_dims =
        HlirShape(update_window_dims_vec,
                  {static_cast<int64_t>(update_window_dims_vec.size())});
    int64_t index_vector_dim = index_sizes.size();
    scatter_dimension_numbers.setValue("index_vector_dim", index_vector_dim);
    scatter_dimension_numbers.setValue("update_window_dims",
                                       update_window_dims);
    scatter_dimension_numbers.setValue("inserted_window_dims", dims);
    scatter_dimension_numbers.setValue("scatter_dims_to_operand_dims", dims);

    // hlir::DispatchParam params;
    hlir::DispatchParam params;
    params.metadata.setValue("scatter_dimension_numbers",
                             scatter_dimension_numbers);
    params.metadata.setValue("indices_are_sorted", false);
    params.metadata.setValue("unique_indices", true);
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());

    phi::DenseTensor tmp_out;
    tmp_out.Resize(reshape_x.dims());
    dev_ctx.Alloc(&tmp_out, out->dtype());

    // hlir::DispatchParam params;
    auto scatter_new_gcu = GetHlirTensor(reshape_x);
    auto indices_gcu = GetHlirTensor(index_tensor);
    auto data_gcu = GetHlirTensor(value_tensor);
    auto scatter_out_gcu = GetHlirTensor(tmp_out);
    params.inputs = {scatter_new_gcu, indices_gcu, data_gcu};
    params.outputs = {scatter_out_gcu};
    params.metadata.setValue("kScatterComputeKind", int64_t(0));
    AOTOPS_DEBUG(kScatter, params);
    GCUOPS_TRACE_START(scatter);
    auto func_ptr = GetOpFuncPtr(kScatter, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass, phi::errors::InvalidArgument("dispatch %s failed!", kScatter));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kScatter));
    }
    FreeDispatchParam(params);
    GCUOPS_TRACE_END(scatter);
    GcuOpStreamSync(dev_ctx);

    *out = reshape(dev_ctx, tmp_out, phi::vectorize(x.dims()));

    PADDLE_GCU_KERNEL_END("set_value", set_value);
  } else {
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

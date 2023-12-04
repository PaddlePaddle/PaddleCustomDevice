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

namespace custom_kernel {

template <typename T, typename Context>
void GatherNdKernel(const Context &dev_ctx,
                    const phi::DenseTensor &x,
                    const phi::DenseTensor &index,
                    phi::DenseTensor *out) {
  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0) return;

  bool empty_index = false;
  std::vector<int> result_dims;
  if (index.numel() == 0) {
    int diff = out->dims().size() - x.dims().size();
    if (diff == 0) {
      TensorCopy(dev_ctx, x, false, out);
      return;
    } else {
      empty_index = true;
      result_dims = phi::vectorize<int>(out->dims());
    }
  }

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "gather_nd", gather_nd);
    // transform gather attrs
    std::vector<int64_t> src_shape = phi::vectorize(x.dims());
    std::vector<int64_t> index_shape = phi::vectorize(index.dims());
    int64_t src_rank = src_shape.size();
    int64_t index_rank = index_shape.size();

    int64_t indices_last_dim_size = index_shape.back();
    std::vector<int64_t> start_index_map;
    int64_t index_vector_dim = index_rank - 1;
    for (auto i = 0; i < indices_last_dim_size; ++i)
      start_index_map.push_back(i);
    auto collapsed_slice_dims = start_index_map;
    int64_t slices_rank = src_rank - indices_last_dim_size;
    int64_t batch_dims_size = index_rank - 1;
    std::vector<int64_t> offset_dims;
    for (auto i = 0; i < slices_rank; ++i)
      offset_dims.push_back(batch_dims_size + i);
    std::vector<int64_t> slice_sizes(indices_last_dim_size, 1);
    for (int64_t i = indices_last_dim_size; i < src_rank; ++i)
      slice_sizes.push_back(src_shape.at(i));

    auto tmp_index = index;
    if (tmp_index.dtype() == phi::DataType::INT64) {
      tmp_index = cast(dev_ctx, tmp_index, phi::DataType::INT32);
    }
    // Run aot ops
    auto src_gcu = GetHlirTensor(x);
    auto index_gcu = GetHlirTensor(tmp_index);
    auto out_gcu = GetHlirTensor(*out);
    hlir::DispatchParam params;
    params.inputs = {src_gcu, index_gcu};
    params.outputs = {out_gcu};
    hlir::Metadata dimension_numbers;
    dimension_numbers.setValue("offset_dims", VectorToHlirShape(offset_dims));
    dimension_numbers.setValue("collapsed_slice_dims",
                               VectorToHlirShape(collapsed_slice_dims));
    dimension_numbers.setValue("start_index_map",
                               VectorToHlirShape(start_index_map));
    dimension_numbers.setValue("index_vector_dim", index_vector_dim);
    params.metadata.setValue("dimension_numbers", dimension_numbers);
    params.metadata.setValue("slice_sizes", VectorToHlirShape(slice_sizes));
    params.metadata.setValue("indices_are_sorted", false);
    params.metadata.setValue("unique_indices", false);
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(kGather, params);
    GCUOPS_TRACE_START(gather_nd);
    auto func_ptr = GetOpFuncPtr(kGather, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass, phi::errors::InvalidArgument("dispatch %s failed!", kGather));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kGather));
    }
    FreeDispatchParam(params);
    GCUOPS_TRACE_END(gather_nd);
    GcuOpStreamSync(dev_ctx);
    PADDLE_GCU_KERNEL_END("gather_nd", gather_nd);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Index"] = {"index"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor *>(&x)};
    inputs["Index"] = {const_cast<DenseTensor *>(&index)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["gather_nd_with_empty_index"] = empty_index;
    attrs["result_dims"] = result_dims;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "gather_nd",
              dev_ctx);
  }
}

template <typename T, typename Context>
void GatherNdGradKernel(const Context &dev_ctx,
                        const phi::DenseTensor &x,
                        const phi::DenseTensor &index,
                        const phi::DenseTensor &dout,
                        phi::DenseTensor *dx) {
  auto x_dims = dx->dims();
  dev_ctx.template Alloc<T>(dx);

  if (dx->numel() == 0) return;

  bool empty_index = false;
  std::vector<int> reduce_axes;
  if (index.numel() == 0) {
    int diff = dout.dims().size() - x.dims().size();
    if (diff == 0) {
      TensorCopy(dev_ctx, dout, false, dx);
      return;
    } else {
      for (size_t i = 0; i < diff; ++i) {
        reduce_axes.emplace_back(i);
      }
      empty_index = true;
    }
  }

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "gather_nd_grad", gather_nd_grad);

    hlir::Metadata dimension_numbers;
    int64_t x_rank = x.dims().size();
    std::vector<int64_t> index_shape = phi::vectorize(index.dims());
    int64_t index_rank = index_shape.size();
    int64_t updates_rank = dout.dims().size();
    int64_t num_index_dims = index_shape.back();
    int64_t window_dims = x_rank - num_index_dims;

    auto tmp_index = index;
    if (tmp_index.dtype() == phi::DataType::INT64) {
      tmp_index = cast(dev_ctx, tmp_index, phi::DataType::INT32);
    }

    // unsqueeze index when dout is scalar
    phi::DenseTensor new_index = tmp_index;
    if (window_dims == 0 && updates_rank == 0) {
      std::vector<int64_t> new_idx_shape = index_shape;
      new_idx_shape.insert(new_idx_shape.begin(), 1);
      index_rank = new_idx_shape.size();
      new_index = reshape(dev_ctx, index, new_idx_shape);
    }

    std::vector<int64_t> update_window_dims;
    for (int64_t i = updates_rank - window_dims; i < updates_rank; ++i)
      update_window_dims.push_back(i);
    std::vector<int64_t> dims(num_index_dims);
    std::iota(dims.begin(), dims.end(), 0);
    dimension_numbers.setValue("index_vector_dim", index_rank - 1);
    dimension_numbers.setValue("update_window_dims",
                               VectorToHlirShape(update_window_dims));
    dimension_numbers.setValue("inserted_window_dims", VectorToHlirShape(dims));
    dimension_numbers.setValue("scatter_dims_to_operand_dims",
                               VectorToHlirShape(dims));

    auto zero = zeros_like(dev_ctx, x);
    auto zero_gcu = GetHlirTensor(zero);
    auto index_gcu = GetHlirTensor(new_index);
    auto src_gcu = GetHlirTensor(dout);
    auto out_gcu = GetHlirTensor(*dx);
    hlir::DispatchParam params;
    params.inputs = {zero_gcu, index_gcu, src_gcu};
    params.outputs = {out_gcu};
    params.metadata.setValue("scatter_dimension_numbers", dimension_numbers);
    params.metadata.setValue(hlir::kScatterComputeKind, /*add*/ int64_t(1));
    params.metadata.setValue("indices_are_sorted", false);
    params.metadata.setValue("unique_indices", false);
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
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

    PADDLE_GCU_KERNEL_END("gather_nd_grad", gather_nd_grad);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Index"] = {"index"};
    input_names[GradVarName("Out")] = {"dout"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor *>(&x)};
    inputs["Index"] = {const_cast<DenseTensor *>(&index)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor *>(&dout)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"dx"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {dx};

    GcuAttributeMap attrs;
    attrs["gather_nd_grad_with_empty_index"] = empty_index;
    attrs["reduce_axes"] = reduce_axes;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "gather_nd_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gather_nd,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GatherNdKernel,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          float,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(gather_nd_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GatherNdGradKernel,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          float) {}


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
void ScatterKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& index,
                   const phi::DenseTensor& updates,
                   bool overwrite,
                   phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "scatter", scatter);
    std::vector<int64_t> input_sizes = phi::vectorize(x.dims());
    std::vector<int64_t> updates_sizes = phi::vectorize(updates.dims());
    std::vector<int64_t> index_sizes = phi::vectorize(index.dims());

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

    phi::DenseTensor tmp_copy_x = x;

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

    int64_t compute_kind = /*update*/ int64_t(0);
    if (!overwrite) {
      compute_kind = /*add*/ int64_t(1);

      auto zero_updates = zeros_like(dev_ctx, updates);
      phi::DenseTensor tmp_out;
      tmp_out.Resize(out->dims());
      dev_ctx.Alloc(&tmp_out, out->dtype());

      auto scatter_ref_gcu = GetHlirTensor(x);
      auto indices_gcu = GetHlirTensor(index);
      auto zero_updates_gcu = GetHlirTensor(zero_updates);
      auto tmp_out_gcu = GetHlirTensor(tmp_out);

      // hlir::DispatchParam params;
      params.inputs = {scatter_ref_gcu, indices_gcu, zero_updates_gcu};
      params.outputs = {tmp_out_gcu};
      params.metadata.setValue("kScatterComputeKind", /*update*/ int64_t(0));
      AOTOPS_DEBUG(kScatter, params);
      GCUOPS_TRACE_START(scatter);
      auto func_ptr = GetOpFuncPtr(kScatter, params);
      if (func_ptr) {
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
        PADDLE_ENFORCE(
            pass,
            phi::errors::InvalidArgument("dispatch %s failed!", kScatter));
      } else {
        PADDLE_ENFORCE(
            false,
            phi::errors::InvalidArgument("not find aot func for %s", kScatter));
      }
      GCUOPS_TRACE_END(scatter);
      GcuOpStreamSync(dev_ctx);
      tmp_copy_x = tmp_out;
    }

    // hlir::DispatchParam params;
    auto scatter_new_gcu = GetHlirTensor(tmp_copy_x);
    auto indices_gcu = GetHlirTensor(index);
    auto data_gcu = GetHlirTensor(updates);
    auto scatter_out_gcu = GetHlirTensor(*out);
    params.inputs = {scatter_new_gcu, indices_gcu, data_gcu};
    params.outputs = {scatter_out_gcu};
    params.metadata.setValue("kScatterComputeKind", compute_kind);
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

    PADDLE_GCU_KERNEL_END("scatter", scatter);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Ids"] = {"index"};
    input_names["Updates"] = {"updates"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Ids"] = {const_cast<DenseTensor*>(&index)};
    inputs["Updates"] = {const_cast<DenseTensor*>(&updates)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["overwrite"] = overwrite;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "scatter", dev_ctx);
  }
}

template <typename T, typename Context>
void ScatterGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& index,
                       const phi::DenseTensor& updates,
                       const phi::DenseTensor& out_grad,
                       bool overwrite,
                       phi::DenseTensor* x_grad,
                       phi::DenseTensor* updates_grad) {}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scatter,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ScatterKernel,
                          float,
                          int64_t,
                          int,
                          phi::dtype::float16) {}

// PD_REGISTER_PLUGIN_KERNEL(scatter_grad,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::ScatterGradKernel,
//                           float,
//                           int64_t,
//                           int,
//                           phi::dtype::float16) {}

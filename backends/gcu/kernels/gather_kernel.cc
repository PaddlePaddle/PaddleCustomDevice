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
void GatherKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& index,
                  const phi::Scalar& axis,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "gather", gather);

    auto rank = x.dims().size();
    auto axis_ = axis.to<int64_t>();
    if (axis_ < 0) axis_ += rank;
    std::vector<int64_t> slice_sizes = phi::vectorize(x.dims());
    slice_sizes.at(axis_) = 1;

    phi::DenseTensor tmp_index = index;
    if (index.dtype() == phi::DataType::INT64) {
      tmp_index = cast(dev_ctx, index, phi::DataType::INT32);
    }

    phi::DenseTensor tmp_x = x;
    if (x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, x, phi::DataType::INT32);
    }

    auto tmp_out = *out;
    if (tmp_out.dtype() == phi::DataType::INT64) {
      auto tmp = EmptyTensor(dev_ctx, phi::DataType::INT32, tmp_out.dims());
      dev_ctx.template Alloc(&tmp, tmp.dtype());
      tmp_out = tmp;
    }

    auto src_gcu = GetHlirTensor(tmp_x);
    auto index_gcu = GetHlirTensor(tmp_index);
    auto out_gcu = GetHlirTensor(tmp_out);
    hlir::DispatchParam params;
    params.inputs = {src_gcu, index_gcu};
    params.outputs = {out_gcu};
    hlir::Metadata dimension_numbers;
    std::vector<int64_t> offset_dims;
    for (auto i = 0; i < axis_; ++i) {
      offset_dims.push_back(i);
    }
    for (auto i = axis_ + 1; i < rank; ++i) {
      offset_dims.push_back(i);
    }
    std::vector<int64_t> dims = {axis_};
    dimension_numbers.setValue("offset_dims", VectorToHlirShape(offset_dims));
    dimension_numbers.setValue("collapsed_slice_dims", VectorToHlirShape(dims));
    dimension_numbers.setValue("start_index_map", VectorToHlirShape(dims));
    dimension_numbers.setValue("index_vector_dim", int64_t(1));
    params.metadata.setValue("dimension_numbers", dimension_numbers);
    params.metadata.setValue("slice_sizes", VectorToHlirShape(slice_sizes));
    params.metadata.setValue("indices_are_sorted", false);
    params.metadata.setValue("unique_indices", false);
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(kGather, params);
    GCUOPS_TRACE_START(gather);
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
    GCUOPS_TRACE_END(gather);
    GcuOpStreamSync(dev_ctx);

    if (out->dtype() == phi::DataType::INT64) {
      *out = cast(dev_ctx, tmp_out, phi::DataType::INT64);
    }

    PADDLE_GCU_KERNEL_END("gather", gather);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Index"] = {"index"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Index"] = {const_cast<DenseTensor*>(&index)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    int gather_axis = axis.to<int>();

    GcuAttributeMap attrs;
    attrs["axis"] = gather_axis;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "gather", dev_ctx);
  }
}

template <typename T, typename Context>
void GatherGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& index,
                      const phi::DenseTensor& out_grad,
                      const phi::Scalar& axis,
                      phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "gather_grad", gather_grad);

    auto rank = x.dims().size();
    auto axis_ = axis.to<int64_t>();
    if (axis_ < 0) axis_ += rank;

    auto zero = zeros_like(dev_ctx, x);

    hlir::Metadata dimension_numbers;
    std::vector<int64_t> update_window_dims;
    for (int64_t i = 1; i < rank; ++i) {
      update_window_dims.push_back(i);
    }
    std::vector<int64_t> dims = {0};
    dimension_numbers.setValue("index_vector_dim", int64_t(1));
    dimension_numbers.setValue("update_window_dims",
                               VectorToHlirShape(update_window_dims));
    dimension_numbers.setValue("inserted_window_dims", VectorToHlirShape(dims));
    dimension_numbers.setValue("scatter_dims_to_operand_dims",
                               VectorToHlirShape(dims));

    auto tmp_index = index;
    if (tmp_index.dtype() == phi::DataType::INT64) {
      tmp_index = cast(dev_ctx, tmp_index, phi::DataType::INT32);
    }
    if (axis_ == 0) {
      auto zero_gcu = GetHlirTensor(zero);
      auto index_gcu = GetHlirTensor(tmp_index);
      auto src_gcu = GetHlirTensor(out_grad);
      auto out_gcu = GetHlirTensor(*x_grad);
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
            pass,
            phi::errors::InvalidArgument("dispatch %s failed!", kScatter));
      } else {
        PADDLE_ENFORCE(
            false,
            phi::errors::InvalidArgument("not find aot func for %s", kScatter));
      }
      FreeDispatchParam(params);
      GCUOPS_TRACE_END(scatter);
      GcuOpStreamSync(dev_ctx);
    } else {
      // transpose scatter axis to 0
      std::vector<int64_t> scatter_perm;
      std::vector<int64_t> out_perm;
      for (int64_t i = 0; i < rank; ++i) {
        if (i == axis_) {
          scatter_perm.insert(scatter_perm.begin(), i);
          out_perm.push_back(0);
        } else {
          scatter_perm.push_back(i);
          if (i < axis_) {
            out_perm.push_back(i + 1);
          } else {
            out_perm.push_back(i);
          }
        }
      }
      auto zero_trans = transpose(dev_ctx, zero, scatter_perm);
      auto src_trans = transpose(dev_ctx, out_grad, scatter_perm);

      // scatter
      std::vector<int64_t> out_dims = phi::vectorize(x_grad->dims());
      std::vector<int64_t> trans_dims = reorder_vector(out_dims, scatter_perm);

      phi::DenseTensor out_trans;
      phi::DenseTensorMeta meta(x_grad->dtype(), phi::make_ddim(trans_dims));
      out_trans.set_meta(meta);
      dev_ctx.Alloc(&out_trans, out_trans.dtype());

      auto zero_gcu = GetHlirTensor(zero_trans);
      auto index_gcu = GetHlirTensor(tmp_index);
      auto src_gcu = GetHlirTensor(src_trans);
      auto out_gcu = GetHlirTensor(out_trans);
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
            pass,
            phi::errors::InvalidArgument("dispatch %s failed!", kScatter));
      } else {
        PADDLE_ENFORCE(
            false,
            phi::errors::InvalidArgument("not find aot func for %s", kScatter));
      }
      FreeDispatchParam(params);
      GCUOPS_TRACE_END(scatter);
      GcuOpStreamSync(dev_ctx);

      // transpose result back
      transpose(dev_ctx, out_trans, *x_grad, out_perm);
    }

    PADDLE_GCU_KERNEL_END("gather_grad", gather_grad);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Index"] = {"index"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Index"] = {const_cast<DenseTensor*>(&index)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"x_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {x_grad};

    int gather_axis = axis.to<int>();

    GcuAttributeMap attrs;
    attrs["axis"] = gather_axis;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "gather_grad",
              dev_ctx);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gather,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GatherKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gather_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GatherGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

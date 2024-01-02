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

#include "kernels/common_ops/reduce_x_ops.h"

#include <vector>

#include "kernels/common_ops/common_ops.h"

namespace custom_kernel {

#define DEFINE_REDUCTION_OP(op, op_name, aot_op)                              \
  void op##_compute(const phi::CustomContext& dev_ctx,                        \
                    const phi::DenseTensor& data,                             \
                    bool keep_dims,                                           \
                    std::vector<int64_t> axes,                                \
                    phi::DenseTensor& output) {                               \
    auto input_rank = data.dims().size();                                     \
    for (auto& dim : axes) {                                                  \
      if (dim < 0) {                                                          \
        dim += input_rank;                                                    \
      }                                                                       \
    }                                                                         \
    if (output.numel() > 0) {                                                 \
      auto data_gcu = GetHlirTensor(data);                                    \
      auto out_gcu = GetHlirTensor(output);                                   \
      hlir::DispatchParam params;                                             \
      params.inputs = {data_gcu};                                             \
      params.outputs = {out_gcu};                                             \
      params.metadata.setValue("keep_dims", keep_dims);                       \
      params.metadata.setValue(                                               \
          "dimensions",                                                       \
          hlir::ShapeMetaData<int64_t>(axes,                                  \
                                       {static_cast<int64_t>(axes.size())})); \
      params.metadata.setValue("nanOpt", static_cast<int64_t>(0));            \
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());            \
      AOTOPS_DEBUG(op_name, params);                                          \
      GCUOPS_TRACE_START(aot_op);                                             \
      auto func_ptr = GetOpFuncPtr(op_name, params);                          \
      if (func_ptr) {                                                         \
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);           \
        PADDLE_ENFORCE(                                                       \
            pass,                                                             \
            phi::errors::InvalidArgument("dispatch %s failed!", op_name));    \
      } else {                                                                \
        PADDLE_ENFORCE(false,                                                 \
                       phi::errors::InvalidArgument(                          \
                           "not find aot func for %s", op_name));             \
      }                                                                       \
      FreeDispatchParam(params);                                              \
      GcuOpStreamSync(dev_ctx);                                               \
      GCUOPS_TRACE_END(aot_op);                                               \
    }                                                                         \
  }                                                                           \
                                                                              \
  phi::DenseTensor op##_compute(const phi::CustomContext& dev_ctx,            \
                                const phi::DenseTensor& data,                 \
                                bool keep_dims,                               \
                                std::vector<int64_t> axes) {                  \
    auto input_rank = data.dims().size();                                     \
    for (auto& dim : axes) {                                                  \
      if (dim < 0) {                                                          \
        dim += input_rank;                                                    \
      }                                                                       \
    }                                                                         \
    std::vector<int64_t> data_dims = phi::vectorize(data.dims());             \
    std::vector<int64_t> out_dims;                                            \
    for (int64_t i = 0; i < data_dims.size(); ++i) {                          \
      if (vector_contains(axes, i)) {                                         \
        if (keep_dims) out_dims.push_back(1);                                 \
      } else {                                                                \
        out_dims.push_back(data_dims[i]);                                     \
      }                                                                       \
    }                                                                         \
    phi::DenseTensor output =                                                 \
        EmptyTensor(dev_ctx, data.dtype(), phi::make_ddim(out_dims));         \
    if (output.numel() > 0) {                                                 \
      auto data_gcu = GetHlirTensor(data);                                    \
      auto out_gcu = GetHlirTensor(output);                                   \
      hlir::DispatchParam params;                                             \
      params.inputs = {data_gcu};                                             \
      params.outputs = {out_gcu};                                             \
      params.metadata.setValue("keep_dims", keep_dims);                       \
      params.metadata.setValue(                                               \
          "dimensions",                                                       \
          hlir::ShapeMetaData<int64_t>(axes,                                  \
                                       {static_cast<int64_t>(axes.size())})); \
      params.metadata.setValue("nanOpt", static_cast<int64_t>(0));            \
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());            \
      AOTOPS_DEBUG(op_name, params);                                          \
      GCUOPS_TRACE_START(aot_op);                                             \
      auto func_ptr = GetOpFuncPtr(op_name, params);                          \
      if (func_ptr) {                                                         \
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);           \
        PADDLE_ENFORCE(                                                       \
            pass,                                                             \
            phi::errors::InvalidArgument("dispatch %s failed!", op_name));    \
      } else {                                                                \
        PADDLE_ENFORCE(false,                                                 \
                       phi::errors::InvalidArgument(                          \
                           "not find aot func for %s", op_name));             \
      }                                                                       \
      FreeDispatchParam(params);                                              \
      GcuOpStreamSync(dev_ctx);                                               \
      GCUOPS_TRACE_END(aot_op);                                               \
    }                                                                         \
    return output;                                                            \
  }

DEFINE_REDUCTION_OP(reduce_sum, kSum, Sum)
DEFINE_REDUCTION_OP(reduce_mean, kMean, Mean)
DEFINE_REDUCTION_OP(reduce_max, kMax, Max)
DEFINE_REDUCTION_OP(reduce_min, kMin, Min)
DEFINE_REDUCTION_OP(reduce_prod, kProd, Prod)

#undef DEFINE_REDUCTION_OP
}  // namespace custom_kernel

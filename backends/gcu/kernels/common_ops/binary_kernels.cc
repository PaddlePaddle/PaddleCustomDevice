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

#include "kernels/common_ops/binary_kernels.h"

namespace custom_kernel {

#define DEFINE_BINARY_OP(op, op_name, aot_op)                              \
  void op##_compute(const phi::CustomContext& dev_ctx,                     \
                    const phi::DenseTensor& x,                             \
                    const phi::DenseTensor& y,                             \
                    phi::DenseTensor* output) {                            \
    VLOG(6) << "start run aot op: " << op_name;                            \
    if (output->capacity() > 0) {                                          \
      auto x_gcu = GetHlirTensor(x);                                       \
      auto y_gcu = GetHlirTensor(y);                                       \
      auto out_gcu = GetHlirTensor(*output);                               \
      hlir::DispatchParam params;                                          \
      params.inputs = {x_gcu, y_gcu};                                      \
      params.outputs = {out_gcu};                                          \
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());         \
      AOTOPS_DEBUG(op_name, params);                                       \
      GCUOPS_TRACE_START(aot_op);                                          \
      auto func_ptr = GetOpFuncPtr(op_name, params);                       \
      if (func_ptr) {                                                      \
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);        \
        PADDLE_ENFORCE(                                                    \
            pass,                                                          \
            phi::errors::InvalidArgument("dispatch %s failed!", op_name)); \
      } else {                                                             \
        PADDLE_ENFORCE(false,                                              \
                       phi::errors::InvalidArgument(                       \
                           "not find aot func for %s", op_name));          \
      }                                                                    \
      FreeDispatchParam(params);                                           \
      GCUOPS_TRACE_END(aot_op);                                            \
      GcuOpStreamSync(params.stream);                                      \
    }                                                                      \
  }                                                                        \
  phi::DenseTensor op##_compute(const phi::CustomContext& dev_ctx,         \
                                const phi::DenseTensor& x,                 \
                                const phi::DenseTensor& y) {               \
    VLOG(6) << "start run aot op: " << op_name;                            \
    phi::DenseTensor output;                                               \
    output.set_meta((x.numel() > y.numel() ? x.meta() : y.meta()));        \
    dev_ctx.Alloc(&output, output.dtype());                                \
    if (output.capacity() > 0) {                                           \
      auto x_gcu = GetHlirTensor(x);                                       \
      auto y_gcu = GetHlirTensor(y);                                       \
      auto out_gcu = GetHlirTensor(output);                                \
      hlir::DispatchParam params;                                          \
      params.inputs = {x_gcu, y_gcu};                                      \
      params.outputs = {out_gcu};                                          \
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());         \
      AOTOPS_DEBUG(op_name, params);                                       \
      GCUOPS_TRACE_START(aot_op);                                          \
      auto func_ptr = GetOpFuncPtr(op_name, params);                       \
      if (func_ptr) {                                                      \
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);        \
        PADDLE_ENFORCE(                                                    \
            pass,                                                          \
            phi::errors::InvalidArgument("dispatch %s failed!", op_name)); \
      } else {                                                             \
        PADDLE_ENFORCE(false,                                              \
                       phi::errors::InvalidArgument(                       \
                           "not find aot func for %s", op_name));          \
      }                                                                    \
      FreeDispatchParam(params);                                           \
      GCUOPS_TRACE_END(aot_op);                                            \
      GcuOpStreamSync(params.stream);                                      \
    }                                                                      \
    return output;                                                         \
  }

#define DEFINE_COMPARE_OP(op, op_name, aot_op)                             \
  void op##_compute(const phi::CustomContext& dev_ctx,                     \
                    const phi::DenseTensor& x,                             \
                    const phi::DenseTensor& y,                             \
                    phi::DenseTensor* output) {                            \
    VLOG(6) << "start run aot op: " << op_name;                            \
    if (output->capacity() > 0) {                                          \
      auto x_gcu = GetHlirTensor(x);                                       \
      auto y_gcu = GetHlirTensor(y);                                       \
      auto out_gcu = GetHlirTensor(*output);                               \
      hlir::DispatchParam params;                                          \
      params.inputs = {x_gcu, y_gcu};                                      \
      params.outputs = {out_gcu};                                          \
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());         \
      AOTOPS_DEBUG(op_name, params);                                       \
      GCUOPS_TRACE_START(aot_op);                                          \
      auto func_ptr = GetOpFuncPtr(op_name, params);                       \
      if (func_ptr) {                                                      \
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);        \
        PADDLE_ENFORCE(                                                    \
            pass,                                                          \
            phi::errors::InvalidArgument("dispatch %s failed!", op_name)); \
      } else {                                                             \
        PADDLE_ENFORCE(false,                                              \
                       phi::errors::InvalidArgument(                       \
                           "not find aot func for %s", op_name));          \
      }                                                                    \
      FreeDispatchParam(params);                                           \
      GCUOPS_TRACE_END(aot_op);                                            \
      GcuOpStreamSync(params.stream);                                      \
    }                                                                      \
  }                                                                        \
  phi::DenseTensor op##_compute(const phi::CustomContext& dev_ctx,         \
                                const phi::DenseTensor& x,                 \
                                const phi::DenseTensor& y) {               \
    VLOG(6) << "start run aot op: " << op_name;                            \
    phi::DenseTensor output;                                               \
    auto meta = (x.numel() > y.numel() ? x.meta() : y.meta());             \
    meta.dtype = phi::DataType::BOOL;                                      \
    output.set_meta(meta);                                                 \
    dev_ctx.Alloc(&output, output.dtype());                                \
    if (output.capacity() > 0) {                                           \
      auto x_gcu = GetHlirTensor(x);                                       \
      auto y_gcu = GetHlirTensor(y);                                       \
      auto out_gcu = GetHlirTensor(output);                                \
      hlir::DispatchParam params;                                          \
      params.inputs = {x_gcu, y_gcu};                                      \
      params.outputs = {out_gcu};                                          \
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());         \
      AOTOPS_DEBUG(op_name, params);                                       \
      GCUOPS_TRACE_START(aot_op);                                          \
      auto func_ptr = GetOpFuncPtr(op_name, params);                       \
      if (func_ptr) {                                                      \
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);        \
        PADDLE_ENFORCE(                                                    \
            pass,                                                          \
            phi::errors::InvalidArgument("dispatch %s failed!", op_name)); \
      } else {                                                             \
        PADDLE_ENFORCE(false,                                              \
                       phi::errors::InvalidArgument(                       \
                           "not find aot func for %s", op_name));          \
      }                                                                    \
      FreeDispatchParam(params);                                           \
      GCUOPS_TRACE_END(aot_op);                                            \
      GcuOpStreamSync(params.stream);                                      \
    }                                                                      \
    return output;                                                         \
  }

DEFINE_BINARY_OP(add, kAdd, add)
DEFINE_BINARY_OP(mul, kMul, mul)
DEFINE_BINARY_OP(div, kDiv, div)
DEFINE_BINARY_OP(sub, kSub, sub)
DEFINE_BINARY_OP(minimum, kMinimum, minimum)
DEFINE_BINARY_OP(maximum, kMaximum, maximum)
DEFINE_BINARY_OP(and, kBitwiseAnd, and)

DEFINE_COMPARE_OP(equal, kEq, equal)
DEFINE_COMPARE_OP(not_equal, kNe, not_equal)
DEFINE_COMPARE_OP(less_than, kLt, less_than)
DEFINE_COMPARE_OP(less_equal, kLe, less_equal)
DEFINE_COMPARE_OP(greater_than, kGt, greater_than)
DEFINE_COMPARE_OP(greater_equal, kGe, greater_equal)

#undef DEFINE_BINARY_OP
#undef DEFINE_COMPARE_OP
}  // namespace custom_kernel

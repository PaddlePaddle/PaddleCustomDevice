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

#include "kernels/common_ops/elementwise_ops.h"

#include <map>
#include <string>
#include <vector>

#include "kernels/common_ops/common_ops.h"

namespace custom_kernel {

#define DEFINE_BINARY_OP(op, op_name, aot_op)                              \
  void op##_compute(const phi::CustomContext& dev_ctx,                     \
                    const phi::DenseTensor& x,                             \
                    const phi::DenseTensor& y,                             \
                    phi::DenseTensor* output) {                            \
    VLOG(6) << "start run aot op: " << op_name;                            \
    int axis = -1;                                                         \
    auto lhs_shape = phi::vectorize(x.dims());                             \
    auto rhs_shape = phi::vectorize(y.dims());                             \
    auto lhs_rank = lhs_shape.size();                                      \
    auto rhs_rank = rhs_shape.size();                                      \
    std::map<std::string, phi::DenseTensor> op_map{{"X", x}, {"Y", y}};    \
    auto low = lhs_rank < rhs_rank ? "X" : "Y";                            \
    std::vector<int64_t> new_shape;                                        \
    int64_t iter = 0;                                                      \
    if (lhs_rank < rhs_rank) {                                             \
      new_shape.assign(rhs_rank, 1);                                       \
      axis = axis > 0 ? axis : rhs_rank - lhs_rank;                        \
      for (int64_t i = axis; i < axis + lhs_rank; ++i) {                   \
        new_shape[i] = lhs_shape[iter++];                                  \
      }                                                                    \
    } else {                                                               \
      new_shape.assign(lhs_rank, 1);                                       \
      axis = axis > 0 ? axis : lhs_rank - rhs_rank;                        \
      for (int64_t i = axis; i < axis + rhs_rank; ++i) {                   \
        new_shape[i] = rhs_shape[iter++];                                  \
      }                                                                    \
    }                                                                      \
    if (phi::vectorize(op_map[low].dims()) != new_shape) {                 \
      op_map[low] = reshape(dev_ctx, op_map[low], new_shape);              \
    }                                                                      \
    if ((op_map["X"].dims() != output->dims()) &&                          \
        (op_map["Y"].dims() != output->dims())) {                          \
      auto out_shape = phi::vectorize(output->dims());                     \
      op_map["X"] = broadcast_to(dev_ctx, op_map["X"], out_shape);         \
      op_map["Y"] = broadcast_to(dev_ctx, op_map["Y"], out_shape);         \
    }                                                                      \
    if (output->capacity() > 0) {                                          \
      auto x_gcu = GetHlirTensor(op_map["X"]);                             \
      auto y_gcu = GetHlirTensor(op_map["Y"]);                             \
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
      GcuOpStreamSync(dev_ctx);                                            \
      GCUOPS_TRACE_END(aot_op);                                            \
    }                                                                      \
  }                                                                        \
                                                                           \
  phi::DenseTensor op##_compute(const phi::CustomContext& dev_ctx,         \
                                const phi::DenseTensor& x,                 \
                                const phi::DenseTensor& y) {               \
    std::map<std::string, phi::DenseTensor> op_map{{"X", x}, {"Y", y}};    \
    int axis = -1;                                                         \
    auto lhs_shape = phi::vectorize(x.dims());                             \
    auto rhs_shape = phi::vectorize(y.dims());                             \
    auto lhs_rank = lhs_shape.size();                                      \
    auto rhs_rank = rhs_shape.size();                                      \
    auto low = lhs_rank < rhs_rank ? "X" : "Y";                            \
    std::vector<int64_t> new_shape;                                        \
    int64_t iter = 0;                                                      \
    if (lhs_rank < rhs_rank) {                                             \
      new_shape.assign(rhs_rank, 1);                                       \
      axis = axis > 0 ? axis : rhs_rank - lhs_rank;                        \
      for (int64_t i = axis; i < axis + lhs_rank; ++i) {                   \
        new_shape[i] = lhs_shape[iter++];                                  \
      }                                                                    \
    } else {                                                               \
      new_shape.assign(lhs_rank, 1);                                       \
      axis = axis > 0 ? axis : lhs_rank - rhs_rank;                        \
      for (int64_t i = axis; i < axis + rhs_rank; ++i) {                   \
        new_shape[i] = rhs_shape[iter++];                                  \
      }                                                                    \
    }                                                                      \
    if (phi::vectorize(op_map[low].dims()) != new_shape) {                 \
      op_map[low] = reshape(dev_ctx, op_map[low], new_shape);              \
    }                                                                      \
    phi::DenseTensor output;                                               \
    output.set_meta((op_map["X"].numel() > op_map["Y"].numel()             \
                         ? op_map["X"].meta()                              \
                         : op_map["Y"].meta()));                           \
    dev_ctx.Alloc(&output, output.dtype());                                \
    if (output.capacity() > 0) {                                           \
      auto x_gcu = GetHlirTensor(op_map["X"]);                             \
      auto y_gcu = GetHlirTensor(op_map["Y"]);                             \
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
      GcuOpStreamSync(dev_ctx);                                            \
      GCUOPS_TRACE_END(aot_op);                                            \
    }                                                                      \
    return output;                                                         \
  }

#define DEFINE_COMPARE_OP(op, op_name, aot_op)                             \
  void op##_compute(const phi::CustomContext& dev_ctx,                     \
                    const phi::DenseTensor& x,                             \
                    const phi::DenseTensor& y,                             \
                    phi::DenseTensor* output) {                            \
    VLOG(6) << "start run aot op: " << op_name;                            \
    int axis = -1;                                                         \
    auto lhs_shape = phi::vectorize(x.dims());                             \
    auto rhs_shape = phi::vectorize(y.dims());                             \
    auto lhs_rank = lhs_shape.size();                                      \
    auto rhs_rank = rhs_shape.size();                                      \
    std::map<std::string, phi::DenseTensor> op_map{{"X", x}, {"Y", y}};    \
    auto low = lhs_rank < rhs_rank ? "X" : "Y";                            \
    std::vector<int64_t> new_shape;                                        \
    int64_t iter = 0;                                                      \
    if (lhs_rank < rhs_rank) {                                             \
      new_shape.assign(rhs_rank, 1);                                       \
      axis = axis > 0 ? axis : rhs_rank - lhs_rank;                        \
      for (int64_t i = axis; i < axis + lhs_rank; ++i) {                   \
        new_shape[i] = lhs_shape[iter++];                                  \
      }                                                                    \
    } else {                                                               \
      new_shape.assign(lhs_rank, 1);                                       \
      axis = axis > 0 ? axis : lhs_rank - rhs_rank;                        \
      for (int64_t i = axis; i < axis + rhs_rank; ++i) {                   \
        new_shape[i] = rhs_shape[iter++];                                  \
      }                                                                    \
    }                                                                      \
    if (phi::vectorize(op_map[low].dims()) != new_shape) {                 \
      op_map[low] = reshape(dev_ctx, op_map[low], new_shape);              \
    }                                                                      \
    if ((op_map["X"].dims() != output->dims()) &&                          \
        (op_map["Y"].dims() != output->dims())) {                          \
      auto out_shape = phi::vectorize(output->dims());                     \
      op_map["X"] = broadcast_to(dev_ctx, op_map["X"], out_shape);         \
      op_map["Y"] = broadcast_to(dev_ctx, op_map["Y"], out_shape);         \
    }                                                                      \
    if (output->capacity() > 0) {                                          \
      auto x_gcu = GetHlirTensor(op_map["X"]);                             \
      auto y_gcu = GetHlirTensor(op_map["Y"]);                             \
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
      GcuOpStreamSync(dev_ctx);                                            \
      GCUOPS_TRACE_END(aot_op);                                            \
    }                                                                      \
  }                                                                        \
                                                                           \
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
      GcuOpStreamSync(dev_ctx);                                            \
      GCUOPS_TRACE_END(aot_op);                                            \
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

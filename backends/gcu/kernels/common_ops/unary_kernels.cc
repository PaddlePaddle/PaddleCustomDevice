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

#include "kernels/common_ops/unary_kernels.h"

namespace custom_kernel {

#define DEFINE_UNARY_OP(op, op_name, aot_op)                               \
  void op##_compute(const phi::CustomContext& dev_ctx,                     \
                    const phi::DenseTensor& input,                         \
                    phi::DenseTensor* output) {                            \
    if (output->capacity() > 0) {                                          \
      if (output->data() == input.data()) {                                \
        phi::DenseTensor tmp_tensor;                                       \
        tmp_tensor.Resize(output->dims());                                 \
        dev_ctx.template Alloc(&tmp_tensor, output->dtype());              \
        *output = tmp_tensor;                                              \
      }                                                                    \
      auto in_gcu = GetHlirTensor(input);                                  \
      auto out_gcu = GetHlirTensor(*output);                               \
      hlir::DispatchParam params;                                          \
      params.inputs = {in_gcu};                                            \
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
                                const phi::DenseTensor& input) {           \
    phi::DenseTensor output = EmptyTensor(dev_ctx, input.meta());          \
    if (output.capacity() > 0) {                                           \
      if (output.data() == input.data()) {                                 \
        phi::DenseTensor tmp_tensor;                                       \
        tmp_tensor.Resize(output.dims());                                  \
        dev_ctx.template Alloc(&tmp_tensor, output.dtype());               \
        output = tmp_tensor;                                               \
      }                                                                    \
      auto in_gcu = GetHlirTensor(input);                                  \
      auto out_gcu = GetHlirTensor(output);                                \
      hlir::DispatchParam params;                                          \
      params.inputs = {in_gcu};                                            \
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

DEFINE_UNARY_OP(abs, kAbs, abs)
DEFINE_UNARY_OP(bitwise_not, kBitwiseNot, bitwise_not)
DEFINE_UNARY_OP(exp, kExp, exp)
DEFINE_UNARY_OP(floor, kFloor, floor)
DEFINE_UNARY_OP(log, kLog, log)
DEFINE_UNARY_OP(relu, kRelu, relu)
DEFINE_UNARY_OP(sigmoid, kSigmoid, sigmoid)
DEFINE_UNARY_OP(neg, kNeg, neg)

#undef DEFINE_UNARY_OP
}  // namespace custom_kernel

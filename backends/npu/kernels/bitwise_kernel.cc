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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {
struct And;
struct Or;
struct Xor;
template <typename P, typename T>
struct compute {
  static void run(T var1, T var2, T* var3) {
    compute<P, T>::run(var1, var2, var3);
  }
};
// bitwise and
template <typename T>
struct compute<And, T> {
  static void run(T var1, T var2, T* var3) { *var3 = var1 & var2; }
};

// bitwise or
template <typename T>
struct compute<Or, T> {
  static void run(T var1, T var2, T* var3) { *var3 = var1 | var2; }
};

// bitwise xor
template <typename T>
struct compute<Xor, T> {
  static void run(T var1, T var2, T* var3) { *var3 = var1 ^ var2; }
};

template <typename T, typename P>
void FallbackToCPU(const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  T val1;
  T val2;
  T res;
  MemCpyD2H(nullptr, &val1, x.data(), sizeof(T));
  MemCpyD2H(nullptr, &val2, y.data(), sizeof(T));
  compute<P, T>::run(val1, val2, &res);
  MemCpyH2D(nullptr, out->data(), &res, sizeof(T));
}

template <typename T, typename Context>
void BitwiseAndKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  // TODO(liyuhang06):
  // Current version CANN bitwise op cann't get accurate result when both input
  // shape is 1-dim. https://gitee.com/ascend/modelzoo/issues/I6AODW
  if (x.numel() == 1 && y.numel() == 1) {
    FallbackToCPU<T, And>(x, y, out);
    return;
  }
  if (x.dtype() == phi::DataType::BOOL) {
    const auto& runner = NpuOpRunner("LogicalAnd", {x, y}, {*out});
    runner.Run(stream);
  } else {
    const auto& runner = NpuOpRunner("BitwiseAnd", {x, y}, {*out});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void BitwiseOrKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  // TODO(liyuhang06):
  // like BitwiseAnd
  if (x.numel() == 1 && y.numel() == 1) {
    FallbackToCPU<T, Or>(x, y, out);
    return;
  }
  if (x.dtype() == phi::DataType::BOOL) {
    const auto& runner = NpuOpRunner("LogicalOr", {x, y}, {*out});
    runner.Run(stream);
  } else {
    const auto& runner = NpuOpRunner("BitwiseOr", {x, y}, {*out});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void BitwiseXorKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  // TODO(liyuhang06):
  // like BitwiseAnd
  if (x.numel() == 1 && y.numel() == 1) {
    FallbackToCPU<T, Xor>(x, y, out);
    return;
  }
  if (x.dtype() == phi::DataType::BOOL) {
    phi::DenseTensor transformed_x, transformed_y, transformed_out;
    transformed_x.Resize(x.dims());
    transformed_y.Resize(y.dims());
    transformed_out.Resize(out->dims());
    dev_ctx.template Alloc<int32_t>(&transformed_x);
    dev_ctx.template Alloc<int32_t>(&transformed_y);
    dev_ctx.template Alloc<int32_t>(&transformed_out);

    const auto& cast_runner1 =
        NpuOpRunner("Cast", {x}, {transformed_x}, {{"dst_type", ACL_INT32}});
    cast_runner1.Run(stream);

    const auto& cast_runner2 =
        NpuOpRunner("Cast", {y}, {transformed_y}, {{"dst_type", ACL_INT32}});
    cast_runner2.Run(stream);

    const auto& runner = NpuOpRunner(
        "BitwiseXor", {transformed_x, transformed_y}, {transformed_out});
    runner.Run(stream);

    const auto& cast_runner3 = NpuOpRunner(
        "Cast", {transformed_out}, {*out}, {{"dst_type", ACL_BOOL}});
    cast_runner3.Run(stream);
  } else {
    const auto& runner = NpuOpRunner("BitwiseXor", {x, y}, {*out});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void BitwiseNotKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  phi::DenseTensor x_tensor(x);
  if (x.dims().size() == 0) {
    x_tensor.Resize({1});
  }

  if (x.dtype() == phi::DataType::BOOL) {
    const auto& runner = NpuOpRunner("LogicalNot", {x}, {*out});
    runner.Run(stream);
  } else {
    const auto& runner = NpuOpRunner("Invert", {x_tensor}, {*out});
    runner.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(bitwise_and,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseAndKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(bitwise_or,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseOrKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(bitwise_xor,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseXorKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(bitwise_not,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseNotKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t) {}

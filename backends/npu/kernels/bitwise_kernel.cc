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

template <typename T, typename Context>
void BitwiseAndKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  phi::DenseTensor x_tensor(x), y_tensor(y), out_tensor(*out);
  if (x.dims().size() == 0 && y.dims().size() == 0) {
    x_tensor.Resize({1});
    y_tensor.Resize({1});
    out_tensor.Resize({1});
  }

  if (x.dtype() == phi::DataType::BOOL) {
    const auto& runner =
        NpuOpRunner("LogicalAnd", {x_tensor, y_tensor}, {out_tensor});
    runner.Run(stream);
  } else {
    const auto& runner =
        NpuOpRunner("BitwiseAnd", {x_tensor, y_tensor}, {out_tensor});
    runner.Run(stream);
  }

  if (x.dims().size() == 0 && y.dims().size() == 0) {
    out->Resize({});
  }
}

template <typename T, typename Context>
void BitwiseOrKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  phi::DenseTensor x_tensor(x), y_tensor(y), out_tensor(*out);
  if (x.dims().size() == 0 && y.dims().size() == 0) {
    x_tensor.Resize({1});
    y_tensor.Resize({1});
    out_tensor.Resize({1});
  }

  if (x.dtype() == phi::DataType::BOOL) {
    const auto& runner =
        NpuOpRunner("LogicalOr", {x_tensor, y_tensor}, {out_tensor});
    runner.Run(stream);
  } else {
    const auto& runner =
        NpuOpRunner("BitwiseOr", {x_tensor, y_tensor}, {out_tensor});
    runner.Run(stream);
  }

  if (x.dims().size() == 0 && y.dims().size() == 0) {
    out->Resize({});
  }
}

template <typename T, typename Context>
void BitwiseXorKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  phi::DenseTensor x_tensor(x), y_tensor(y), out_tensor(*out);
  if (x.dims().size() == 0 && y.dims().size() == 0) {
    x_tensor.Resize({1});
    y_tensor.Resize({1});
    out_tensor.Resize({1});
  }

  if (x.dtype() == phi::DataType::BOOL) {
    // NotEqual op do not support bool data type
    phi::DenseTensor transformed_x, transformed_y;
    transformed_x.Resize(x_tensor.dims());
    transformed_y.Resize(y_tensor.dims());
    dev_ctx.template Alloc<int32_t>(&transformed_x);
    dev_ctx.template Alloc<int32_t>(&transformed_y);

    const auto& cast_runner1 = NpuOpRunner(
        "Cast", {x_tensor}, {transformed_x}, {{"dst_type", ACL_INT32}});
    cast_runner1.Run(stream);

    const auto& cast_runner2 = NpuOpRunner(
        "Cast", {y_tensor}, {transformed_y}, {{"dst_type", ACL_INT32}});
    cast_runner2.Run(stream);

    const auto& runner =
        NpuOpRunner("NotEqual", {transformed_x, transformed_y}, {out_tensor});
    runner.Run(stream);
  } else {
    const auto& runner =
        NpuOpRunner("BitwiseXor", {x_tensor, y_tensor}, {out_tensor});
    runner.Run(stream);
  }

  if (x.dims().size() == 0 && y.dims().size() == 0) {
    out->Resize({});
  }
}

template <typename T, typename Context>
void BitwiseNotKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  phi::DenseTensor x_tensor(x), out_tensor(*out);
  if (x.dims().size() == 0) {
    x_tensor.Resize({1});
    out_tensor.Resize({1});
  }

  if (x.dtype() == phi::DataType::BOOL) {
    const auto& runner = NpuOpRunner("LogicalNot", {x}, {*out});
    runner.Run(stream);
  } else {
    phi::DenseTensor all_f;
    all_f.Resize(x_tensor.dims());

    if (x.dtype() == phi::DataType::INT8 || x.dtype() == phi::DataType::UINT8) {
      FillNpuTensorWithConstant<T>(&all_f, dev_ctx, static_cast<T>(0xff));
    } else if (x.dtype() == phi::DataType::INT16) {
      FillNpuTensorWithConstant<T>(&all_f, dev_ctx, static_cast<T>(0xffff));
    } else if (x.dtype() == phi::DataType::INT32) {
      FillNpuTensorWithConstant<T>(&all_f, dev_ctx, static_cast<T>(0xffffffff));
    } else if (x.dtype() == phi::DataType::INT64) {
      FillNpuTensorWithConstant<T>(
          &all_f, dev_ctx, static_cast<T>(0xffffffffffffffff));
    } else {
      phi::errors::InvalidArgument(
          "Supported data type for BitwiseNot is bool, int8, uint8, int16, int "
          "and int64, but received %s",
          x.dtype());
    }
    all_f.Resize(x.dims());

    const auto& runner =
        NpuOpRunner("BitwiseXor", {x_tensor, all_f}, {out_tensor});
    runner.Run(stream);
  }

  if (x.dims().size() == 0) {
    out->Resize({});
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

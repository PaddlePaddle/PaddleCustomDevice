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

#include <iostream>
#include <vector>

#include "ascendc/elementwise_add.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"
#include "runtime/runtime.h"

#ifdef PADDLE_WITH_ASCENDC
template <typename T>
struct elementwise_add {
  void operator()(const phi::CustomContext& dev_ctx,
                  const void* x,
                  const void* y,
                  void* z,
                  size_t length) {
    PADDLE_THROW(phi::errors::Unimplemented("Not implemented yet."));
  }
};

template <>
struct elementwise_add<float> {
  void operator()(const phi::CustomContext& dev_ctx,
                  size_t blockDim,
                  const void* x,
                  const void* y,
                  void* z,
                  size_t length) {
    elementwise_add_f32(
        blockDim, nullptr, dev_ctx.stream(), x, y, z, length, 1);
  }
};

template <>
struct elementwise_add<phi::dtype::float16> {
  void operator()(const phi::CustomContext& dev_ctx,
                  size_t blockDim,
                  const void* x,
                  const void* y,
                  void* z,
                  size_t length) {
    elementwise_add_f16(
        blockDim, nullptr, dev_ctx.stream(), x, y, z, length, 1);
  }
};

template <>
struct elementwise_add<int32_t> {
  void operator()(const phi::CustomContext& dev_ctx,
                  size_t blockDim,
                  const void* x,
                  const void* y,
                  void* z,
                  size_t length) {
    elementwise_add_s32(
        blockDim, nullptr, dev_ctx.stream(), x, y, z, length, 1);
  }
};

template <>
struct elementwise_add<int16_t> {
  void operator()(const phi::CustomContext& dev_ctx,
                  size_t blockDim,
                  const void* x,
                  const void* y,
                  void* z,
                  size_t length) {
    elementwise_add_s16(
        blockDim, nullptr, dev_ctx.stream(), x, y, z, length, 1);
  }
};

#endif

std::vector<paddle::Tensor> MyAddOp(const paddle::Tensor& x,
                                    const paddle::Tensor& y) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto x_dtype = x.dtype();
  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim(x.shape()));
  dev_ctx->Alloc(out_tensor.get(), x_dtype);

#if PADDLE_WITH_ASCENDC
  auto nBlocks = custom_device::ascendc::GetBlockNum();
  if (x_dtype == phi::DataType::FLOAT32) {
    elementwise_add<float>()(
        *dev_ctx, nBlocks, x.data(), y.data(), out_tensor->data(), x.numel());
  } else if (x_dtype == phi::DataType::FLOAT16) {
    elementwise_add<phi::dtype::float16>()(
        *dev_ctx, nBlocks, x.data(), y.data(), out_tensor->data(), x.numel());
  } else if (x_dtype == phi::DataType::INT32) {
    elementwise_add<int32_t>()(
        *dev_ctx, nBlocks, x.data(), y.data(), out_tensor->data(), x.numel());
  } else {
    PADDLE_THROW(phi::errors::Unimplemented("Only support fp16/fp32/int32."));
  }
#else
  auto x_tensor = static_cast<const phi::DenseTensor*>(x.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor*>(y.impl().get());
  const auto& add_runner1 =
      NpuOpRunner("Add", {*x_tensor, *y_tensor}, {*out_tensor}, {});
  add_runner1.Run(stream);
#endif

  return {paddle::Tensor(out_tensor)};
}

std::vector<std::vector<int64_t>> MyAddOpInferShape(
    const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape) {
  return {x_shape};
}

PD_BUILD_OP(my_add)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(MyAddOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        MyAddOpInferShape));  // neccessary if the op has muti_inputs

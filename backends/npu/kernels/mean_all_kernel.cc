// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
void AclopMeanAllKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        phi::DenseTensor* out) {
  auto rank = x.dims().size();
  auto out_dims = out->dims();
  dev_ctx.template Alloc<T>(out);
  if (rank == 0) {  // scalar
    TensorCopy(dev_ctx, x, false, out);
    out->Resize(out_dims);  // copy will reset the dims.
    return;
  }

  auto stream = dev_ctx.stream();

  std::vector<int64_t> reduce_dims;
  reduce_dims.reserve(rank);
  for (decltype(rank) i = 0; i < rank; ++i) {
    reduce_dims.push_back(i);
  }

  NpuOpRunner runner;
  runner.SetType("ReduceMean")
      .AddInput(x)
      .AddInput(dev_ctx, std::move(reduce_dims))
      .AddOutput(*out)
      .AddAttr("keep_dims", false)
      .AddAttr("noop_with_empty_axes", true)
      .Run(stream);
}

template <typename T, typename Context>
void MeanAllKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  DO_COMPATIBILITY(
      aclnnMeanV2,
      (custom_kernel::AclopMeanAllKernel<T, Context>(dev_ctx, x, out)));

  auto rank = x.dims().size();
  auto out_dims = out->dims();
  dev_ctx.template Alloc<T>(out);
  if (rank == 0) {  // scalar
    TensorCopy(dev_ctx, x, false, out);
    out->Resize(out_dims);  // copy will reset the dims.
    return;
  }

  std::vector<int64_t> reduce_dims;
  reduce_dims.reserve(rank);
  for (decltype(rank) i = 0; i < rank; ++i) {
    reduce_dims.push_back(i);
  }

  bool keep_dim = false;
  bool noop_with_empty_dims = true;
  EXEC_NPU_CMD(aclnnMeanV2,
               dev_ctx,
               x,
               reduce_dims,
               keep_dim,
               noop_with_empty_dims,
               *out);
}

template <typename T, typename Context>
void MeanAllGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& grad,
                       phi::DenseTensor* x_grad) {
  auto stream = dev_ctx.stream();

  PADDLE_ENFORCE_EQ(
      grad.numel(),
      1,
      phi::errors::InvalidArgument(
          "Mean Gradient Input phi::DenseTensor len should be 1. But "
          "received Out@Grad's elements num is %d.",
          grad.numel()));

  dev_ctx.template Alloc<T>(x_grad);

  // ones
  phi::DenseTensor ones;
  phi::DenseTensorMeta ones_meta = {grad.dtype(), x_grad->dims()};
  ones.set_meta(ones_meta);
  dev_ctx.template Alloc<T>(&ones);
  const auto& runner_ones = NpuOpRunner("OnesLike", {*x_grad}, {ones}, {});
  runner_ones.Run(stream);

  // means
  phi::DenseTensor mean_tensor;
  phi::DenseTensorMeta mean_meta = {grad.dtype(), {1}};
  mean_tensor.set_meta(mean_meta);
  dev_ctx.template Alloc<T>(&mean_tensor);
  FillNpuTensorWithConstant<T>(
      &mean_tensor,
      dev_ctx,
      static_cast<T>(1.0 / static_cast<float>(x_grad->numel())));

  // means mul ones
  phi::DenseTensor mean_ma;
  phi::DenseTensorMeta mean_ma_meta = {grad.dtype(), x_grad->dims()};
  mean_ma.set_meta(mean_ma_meta);
  dev_ctx.template Alloc<T>(&mean_ma);

  const auto& runner_mul_1 =
      NpuOpRunner("Mul", {mean_tensor, ones}, {mean_ma}, {});
  runner_mul_1.Run(stream);

  // and mul grad
  const auto& runner_mul_2 = NpuOpRunner("Mul", {mean_ma, grad}, {*x_grad}, {});
  runner_mul_2.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(mean_all,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(mean_all_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

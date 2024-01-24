// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
void AclopCheckFiniteAndUnscale(const Context& dev_ctx,
                                const std::vector<const phi::DenseTensor*>& xs,
                                const phi::DenseTensor& t_scale,
                                std::vector<phi::DenseTensor*> outs,
                                phi::DenseTensor* found_inf) {
  auto stream = dev_ctx.stream();
  phi::DenseTensor scale;
  if (std::is_same<T, phi::dtype::float16>::value) {
    scale.Resize(t_scale.dims());
    dev_ctx.template Alloc<T>(&scale);
    const auto& cast_runner =
        NpuOpRunner("Cast",
                    {t_scale},
                    {scale},
                    {{"dst_type", static_cast<int>(ACL_FLOAT16)}});
    cast_runner.Run(stream);
  } else {
    scale = t_scale;
  }

  dev_ctx.template Alloc<bool>(found_inf);

  // step1: inverse scale
  phi::DenseTensor const_tensor;
  const_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&const_tensor);
  FillNpuTensorWithConstant<T>(&const_tensor, dev_ctx, static_cast<T>(1.0));

  // Inverse(1.0/scale)
  phi::DenseTensor* tmp_inverse_out = const_cast<phi::DenseTensor*>(&scale);
  phi::DenseTensor inverse_out;
  inverse_out.Resize(scale.dims());
  dev_ctx.template Alloc<T>(&inverse_out);
  const auto& runner_inverse =
      NpuOpRunner("Div", {const_tensor, scale}, {inverse_out}, {});
  runner_inverse.Run(stream);
  tmp_inverse_out = &inverse_out;

  bool found_inf_cpu = NpuOpRunner::GetFloatStatus(stream);
  FillNpuTensorWithConstant<bool>(found_inf, dev_ctx, found_inf_cpu);

  // NOTE(zhiqiu): The normal logic is :
  // out = in, if found_inf = true
  // out = in/scale, if found_inf = false
  // However, on NPU, in order to avoid stream sync, we do not copy the
  // found_inf data to cpu to check whether to unscale or not.
  // Instead, we do the Mul no matter found_inf or not.
  // And, a fact is, only few steps contains nan/inf during training.
  for (size_t i = 0; i < xs.size(); ++i) {
    const auto* x = xs[i];
    auto* out = outs[i];
    dev_ctx.template Alloc<T>(out);
    const auto& runner_mul = NpuOpRunner("Mul", {*x, inverse_out}, {*out}, {});
    runner_mul.Run(stream);
  }

  NpuOpRunner::ClearFloatStatus(stream);
}

template <typename T, typename Context>
void CastKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void CheckFiniteAndUnscale(const Context& dev_ctx,
                           const std::vector<const phi::DenseTensor*>& xs,
                           const phi::DenseTensor& t_scale,
                           std::vector<phi::DenseTensor*> outs,
                           phi::DenseTensor* found_inf) {
  DO_COMPATIBILITY(aclnnForeachNonFiniteCheckAndUnscale,
                   (custom_kernel::AclopCheckFiniteAndUnscale<T, Context>(
                       dev_ctx, xs, t_scale, outs, found_inf)));
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<bool>(found_inf);

  for (int i = 0; i < xs.size(); i++) {
    dev_ctx.template Alloc<T>(outs[i]);
    TensorCopy(dev_ctx, *xs[i], true, outs[i]);
  }

  // step1: inverse scale
  phi::DenseTensor const_tensor;
  phi::DenseTensor found_inf_tmp;
  found_inf_tmp.Resize({1});
  dev_ctx.template Alloc<float>(&found_inf_tmp);

  const_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&const_tensor);
  FillNpuTensorWithConstant<float>(&const_tensor, dev_ctx, static_cast<T>(1.0));
  FillNpuTensorWithConstant<float>(
      &found_inf_tmp, dev_ctx, static_cast<float>(0));

  // Inverse(1.0/scale)
  phi::DenseTensor* tmp_inverse_out = const_cast<phi::DenseTensor*>(&t_scale);
  phi::DenseTensor inverse_out;
  inverse_out.Resize(t_scale.dims());
  dev_ctx.template Alloc<float>(&inverse_out);
  const auto& runner_inverse =
      NpuOpRunner("Div", {const_tensor, t_scale}, {inverse_out}, {});
  runner_inverse.Run(stream);

  size_t tensor_count = outs.size();
  size_t loop_time = tensor_count / 256;  // Upward division

  for (size_t i = 0; i < loop_time; i++) {
    std::vector<phi::DenseTensor*> tmp(outs.begin() + i * 256,
                                       outs.begin() + i * 256 + 256);
    EXEC_NPU_CMD(aclnnForeachNonFiniteCheckAndUnscale,
                 dev_ctx,
                 tmp,
                 found_inf_tmp,
                 inverse_out);
  }
  size_t remaining_count = tensor_count % 256;
  if (remaining_count) {
    std::vector<phi::DenseTensor*> tmp(
        outs.begin() + loop_time * 256,
        outs.begin() + loop_time * 256 + remaining_count);
    EXEC_NPU_CMD(aclnnForeachNonFiniteCheckAndUnscale,
                 dev_ctx,
                 tmp,
                 found_inf_tmp,
                 inverse_out);
  }
  custom_kernel::CastKernel<T, Context>(
      dev_ctx, found_inf_tmp, phi::DataType::BOOL, found_inf);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(check_finite_and_unscale,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CheckFiniteAndUnscale,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

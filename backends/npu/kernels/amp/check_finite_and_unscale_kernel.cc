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
void CheckFiniteAndUnscale(const Context& dev_ctx,
                           const std::vector<const phi::DenseTensor*>& xs,
                           const phi::DenseTensor& t_scale,
                           std::vector<phi::DenseTensor*> outs,
                           phi::DenseTensor* found_inf) {
  auto stream = dev_ctx.stream();
  auto scale = &t_scale;

  dev_ctx.template Alloc<bool>(found_inf);

  // step1: inverse scale
  phi::DenseTensor const_tensor;
  const_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&const_tensor);
  FillNpuTensorWithConstant<T>(&const_tensor, dev_ctx, static_cast<T>(1.0));

  // Inverse(1.0/scale)
  phi::DenseTensor* tmp_inverse_out = const_cast<phi::DenseTensor*>(scale);
  phi::DenseTensor inverse_out;
  inverse_out.Resize(scale->dims());
  dev_ctx.template Alloc<T>(&inverse_out);
  const auto& runner_inverse =
      NpuOpRunner("Div", {const_tensor, *scale}, {inverse_out}, {});
  runner_inverse.Run(stream);
  tmp_inverse_out = &inverse_out;

  // bool found_inf_cpu = NpuOpRunner::GetFloatStatus(stream);
  // FillNpuTensorWithConstant<bool>(found_inf, dev_ctx, found_inf_cpu);

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
    const auto& runner_mul =
        NpuOpRunner("Mul", {*x, *tmp_inverse_out}, {*out}, {});
    runner_mul.Run(stream);
  }

  NpuOpRunner::ClearFloatStatus(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(check_finite_and_unscale,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::CheckFiniteAndUnscale,
                          float,
                          double) {}

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

  phi::DenseTensor sum;
  sum.Resize({1});
  dev_ctx.template Alloc<T>(&sum);
  FillNpuTensorWithConstant<T>(&sum, dev_ctx, static_cast<T>(0.0));

  phi::DenseTensor tmp;
  tmp.Resize({1});
  dev_ctx.template Alloc<T>(&tmp);

  for (const auto& xs_item : xs) {
    std::vector<int> axes;
    phi::DenseTensor xs_is_finite, xs_is_finite_f, xs_is_inf, xs_is_nan;
    xs_is_finite.Resize(xs_item->dims());
    dev_ctx.template Alloc<bool>(&xs_is_finite);
    xs_is_inf.Resize(xs_item->dims());
    dev_ctx.template Alloc<bool>(&xs_is_inf);
    xs_is_nan.Resize(xs_item->dims());
    dev_ctx.template Alloc<bool>(&xs_is_nan);
    xs_is_finite_f.Resize(xs_item->dims());
    dev_ctx.template Alloc<T>(&xs_is_finite_f);

    for (auto i = 0; i < xs_item->dims().size(); ++i) {
      axes.push_back(i);
    }

    const auto& runner_check_inf =
        NpuOpRunner("IsInf", {*xs_item}, {xs_is_inf}, {});
    runner_check_inf.Run(stream);

    const auto& runner_check_nan =
        NpuOpRunner("IsNan", {*xs_item}, {xs_is_nan}, {});
    runner_check_nan.Run(stream);

    const auto& runner_logical_and =
        NpuOpRunner("LogicalOr", {xs_is_inf, xs_is_nan}, {xs_is_finite}, {});
    runner_logical_and.Run(stream);

    const auto& runner_cast = NpuOpRunner(
        "Cast",
        {xs_is_finite},
        {xs_is_finite_f},
        {{"dst_type", static_cast<int>(cpp_type_to_acl_dtype<T>::value())}});
    runner_cast.Run(stream);

    const auto& runner_reduce_sum =
        NpuOpRunner("ReduceSumD",
                    {xs_is_finite_f},
                    {tmp},
                    {{"axes", axes}, {"keep_dims", false}});
    runner_reduce_sum.Run(stream);

    const auto& runner_add = NpuOpRunner("Add", {tmp, sum}, {sum}, {});
    runner_add.Run(stream);
  }

  const auto& runner_greater =
      NpuOpRunner("GreaterEqual", {sum, const_tensor}, {*found_inf}, {});
  runner_greater.Run(stream);

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
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(check_finite_and_unscale,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::CheckFiniteAndUnscale,
                          float,
                          double) {}

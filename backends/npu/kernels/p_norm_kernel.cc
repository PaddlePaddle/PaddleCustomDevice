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

#include <limits.h>

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void PnormKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 float porder,
                 int axis,
                 float epsilon,
                 bool keepdim,
                 bool asvector,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto xdim = x.dims();
  axis = axis < 0 ? xdim.size() + axis : axis;

  auto stream = dev_ctx.stream();

  int p = 0;
  bool combine_op = !(porder == 0 || porder == INFINITY || porder == -INFINITY);
  if (porder == INFINITY) {
    p = INT_MAX;
  } else if (porder == -INFINITY) {
    p = INT_MIN;
  } else {
    p = static_cast<int>(porder);
    float t = 0;
    float diff = abs(std::modf(porder, &t));
    if (diff < 1e-5) {
      combine_op = false;
    }
  }

  if (!combine_op) {
    const auto& runner = NpuOpRunner("LpNorm",
                                     {x},
                                     {*out},
                                     {{"p", p},
                                      {"axes", std::vector<int32_t>({axis})},
                                      {"keep_dims", keepdim}});
    runner.Run(stream);
  } else {
    phi::DenseTensor tmp_x;
    tmp_x.Resize(xdim);
    dev_ctx.template Alloc<T>(&tmp_x);

    const auto& power_runner1 =
        NpuOpRunner("Power",
                    {x},
                    {tmp_x},
                    {{"power", porder}, {"scale", 1.0f}, {"shift", 0.0f}});
    power_runner1.Run(stream);

    const auto& reduce_runner = NpuOpRunner(
        "ReduceSumD",
        {tmp_x},
        {*out},
        {{"axes", std::vector<int32_t>({axis})}, {"keep_dims", keepdim}});
    reduce_runner.Run(stream);

    const auto& power_runner2 =
        NpuOpRunner("Power",
                    {*out},
                    {*out},
                    {{"power", 1 / porder}, {"scale", 1.0f}, {"shift", 0.0f}});
    power_runner2.Run(stream);
  }
}

template <typename T, typename Context>
void PnormGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     const phi::DenseTensor& dy,
                     float porder,
                     int axis,
                     float epsilon,
                     bool keepdim,
                     bool asvector,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto xdim = x.dims();
  axis = axis < 0 ? xdim.size() + axis : axis;

  auto stream = dev_ctx.stream();

  phi::DenseTensor y_share(y);
  phi::DenseTensor dy_share(dy);
  auto ydim = xdim;
  if (!keepdim) {
    ydim[axis] = 1;
  } else {
    ydim = y.dims();
  }
  y_share.Resize(ydim);
  dy_share.Resize(ydim);
  if (porder == 0) {
    FillNpuTensorWithConstant(out, dev_ctx, static_cast<T>(0));
    out->Resize(xdim);
  } else if (porder == INFINITY || porder == -INFINITY) {
    phi::DenseTensor x_abs;
    x_abs.Resize(xdim);
    dev_ctx.template Alloc<T>(&x_abs);
    const auto& r_abs = NpuOpRunner("Abs", {x}, {x_abs}, {});
    r_abs.Run(stream);

    phi::DenseTensor t_cond;
    t_cond.Resize(xdim);
    dev_ctx.template Alloc<bool>(&t_cond);
    const auto& r_equal = NpuOpRunner("Equal", {x_abs, y_share}, {t_cond}, {});
    r_equal.Run(stream);

    phi::DenseTensor t_zero;
    t_zero.Resize({1});
    dev_ctx.template Alloc<T>(&t_zero);
    FillNpuTensorWithConstant(&t_zero, dev_ctx, static_cast<T>(0));

    phi::DenseTensor x_sign;
    x_sign.Resize(xdim);
    dev_ctx.template Alloc<T>(&x_sign);
    const auto& r_sign = NpuOpRunner("Sign", {x}, {x_sign}, {});
    r_sign.Run(stream);

    const auto& r_mul = NpuOpRunner("Mul", {x_sign, dy_share}, {*out}, {});
    r_mul.Run(stream);

    const auto& r_sel =
        NpuOpRunner("SelectV2", {t_cond, *out, t_zero}, {*out}, {});
    r_sel.Run(stream);
  } else {
    phi::DenseTensor x_abs;
    x_abs.Resize(xdim);
    dev_ctx.template Alloc<T>(&x_abs);
    const auto& r_abs = NpuOpRunner("Abs", {x}, {x_abs}, {});
    r_abs.Run(stream);

    phi::DenseTensor x_sign;
    x_sign.Resize(xdim);
    dev_ctx.template Alloc<T>(&x_sign);
    const auto& r_sign = NpuOpRunner("Sign", {x}, {x_sign}, {});
    r_sign.Run(stream);

    phi::DenseTensor y_pow;
    y_pow.Resize(ydim);
    dev_ctx.template Alloc<T>(&y_pow);
    if (porder >= 1) {
      const auto& r_pow1 = NpuOpRunner(
          "Power",
          {x_abs},
          {x_abs},
          {{"power", (porder - 1)}, {"scale", 1.0f}, {"shift", 0.0f}});
      r_pow1.Run(stream);

      const auto& r_pow2 = NpuOpRunner(
          "Power",
          {y_share},
          {y_pow},
          {{"power", (porder - 1)}, {"scale", 1.0f}, {"shift", 0.0f}});
      r_pow2.Run(stream);

      const auto& r_div = NpuOpRunner("DivNoNan", {x_abs, y_pow}, {*out}, {});
      r_div.Run(stream);
    } else {
      const auto& r_pow1 = NpuOpRunner(
          "Power",
          {x_abs},
          {x_abs},
          {{"power", (1 - porder)}, {"scale", 1.0f}, {"shift", 0.0f}});
      r_pow1.Run(stream);

      const auto& r_pow2 = NpuOpRunner(
          "Power",
          {y_share},
          {y_pow},
          {{"power", (1 - porder)}, {"scale", 1.0f}, {"shift", 0.0f}});
      r_pow2.Run(stream);

      const auto& r_div = NpuOpRunner("DivNoNan", {y_pow, x_abs}, {*out}, {});
      r_div.Run(stream);
    }

    const auto& r_mul1 = NpuOpRunner("Mul", {*out, x_sign}, {*out}, {});
    r_mul1.Run(stream);

    const auto& r_mul2 = NpuOpRunner("Mul", {*out, dy_share}, {*out}, {});
    r_mul2.Run(stream);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(p_norm,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::PnormKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(p_norm_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::PnormGradKernel,
                          float,
                          phi::dtype::float16) {}

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
static void ReduceDims(const Context& dev_ctx,
                       const aclrtStream& stream,
                       const int axis,
                       const phi::DDim& ddims,
                       const phi::DDim& brd_ddims,
                       const phi::DenseTensor& in,
                       phi::DenseTensor* out) {
  std::vector<int64_t> axes;
  int64_t brd_size = brd_ddims.size();
  int64_t org_size = ddims.size();
  // int64_t diff = brd_dims.size() - dims.size();
  for (int64_t i = 0; i < brd_size; ++i) {
    if (i < axis || i >= org_size + axis) {
      axes.push_back(i);
      continue;
    }
    if (brd_ddims[i] > ddims[i - axis]) {
      axes.push_back(i);
    }
  }
  dev_ctx.template Alloc<T>(out);
  const auto& runner = NpuOpRunner(
      "ReduceSumD", {in}, {*out}, {{"axes", axes}, {"keep_dims", false}});
  runner.Run(stream);
}

template <typename T, typename Context>
void DivideRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     int axis,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  const auto& runner = NpuOpRunner("Div", {x, y}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::DivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void DivideGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      const phi::DenseTensor& out,
                      const phi::DenseTensor& dout,
                      int axis,
                      phi::DenseTensor* dx,
                      phi::DenseTensor* dy) {
  auto stream = dev_ctx.stream();
  axis = (axis == -1 ? std::abs(x.dims().size() - y.dims().size()) : axis);
  phi::DenseTensor trans_x, trans_y;
  NpuElementWiseOpBroadcast<T>(dev_ctx, &x, &y, axis, &trans_x, &trans_y);
  // compute dout/y == 1/y * dout
  phi::DenseTensor dout_div_y;
  phi::DenseTensorMeta dout_div_y_meta = {dout.dtype(), dout.dims()};
  dout_div_y.set_meta(dout_div_y_meta);
  dev_ctx.template Alloc<T>(&dout_div_y);
  const auto& runner = NpuOpRunner("Div", {dout, trans_y}, {dout_div_y}, {});
  runner.Run(stream);
  if (dx) {
    // compute dx = dout/y = 1/y * dout
    if (dx->dims() == dout.dims()) {
      *dx = dout_div_y;
    } else {
      dev_ctx.template Alloc<T>(dx);
      ReduceDims<T>(
          dev_ctx, stream, axis, dx->dims(), trans_x.dims(), dout_div_y, dx);
    }
  }
  if (dy) {
    // compute dy = -out * (dout/y) = -out/y * dout
    phi::DenseTensor neg_out;
    phi::DenseTensorMeta neg_out_meta = {out.dtype(), out.dims()};
    neg_out.set_meta(neg_out_meta);
    dev_ctx.template Alloc<T>(&neg_out);
    const auto& runner_neg_out = NpuOpRunner("Neg", {out}, {neg_out}, {});
    runner_neg_out.Run(stream);

    phi::DenseTensor dy_tmp;
    phi::DenseTensorMeta dy_tmp_meta = {dout.dtype(), dout.dims()};
    dy_tmp.set_meta(dy_tmp_meta);
    dev_ctx.template Alloc<T>(&dy_tmp);
    const auto& runner_mul =
        NpuOpRunner("Mul", {neg_out, dout_div_y}, {dy_tmp}, {});
    runner_mul.Run(stream);

    if (dy->dims() == dout.dims()) {
      *dy = dy_tmp;
    } else {
      dev_ctx.template Alloc<T>(dy);
      ReduceDims<T>(
          dev_ctx, stream, axis, dy->dims(), trans_y.dims(), dy_tmp, dy);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(divide_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DivideRawKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(divide,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DivideKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(divide_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::DivideGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          double) {}

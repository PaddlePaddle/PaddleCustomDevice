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
#include "kernels/funcs/op_command.h"

namespace custom_kernel {

// template <typename T, typename Context>
// static void ReduceDims(const Context& dev_ctx,
//                        const aclrtStream& stream,
//                        const int axis,
//                        const phi::DDim& ddims,
//                        const phi::DDim& brd_ddims,
//                        const phi::DenseTensor& in,
//                        phi::DenseTensor* out) {
//   std::vector<int64_t> axes;
//   int64_t brd_size = brd_ddims.size();
//   int64_t org_size = ddims.size();
//   // int64_t diff = brd_dims.size() - dims.size();
//   for (int64_t i = 0; i < brd_size; ++i) {
//     if (i < axis || i >= org_size + axis) {
//       axes.push_back(i);
//       continue;
//     }
//     if (brd_ddims[i] > ddims[i - axis]) {
//       axes.push_back(i);
//     }
//   }
//   dev_ctx.template Alloc<T>(out);
//   const auto& runner = NpuOpRunner(
//       "ReduceSumD", {in}, {*out}, {{"axes", axes}, {"keep_dims", false}});
//   runner.Run(stream);
// }

template <typename T, typename Context>
void MultiplyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();

  bool direct_compute = false;
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  if (x_dims.size() >= y_dims.size()) {
    direct_compute = x_dims.size() == (y_dims.size() + axis);
  } else {
    direct_compute = y_dims.size() == (x_dims.size() + axis);
  }

  if (direct_compute) {
    experimental::OpCommand("Mul").Input(x).Input(y).Output(*out).Run(dev_ctx);
  } else {
    phi::DenseTensor transformed_x, transformed_y;
    transformed_x.Resize(out->dims());
    transformed_y.Resize(out->dims());
    dev_ctx.template Alloc<T>(&transformed_x);
    dev_ctx.template Alloc<T>(&transformed_y);
    experimental::OpCommandHelper::BroadcastTo(
        dev_ctx, x, axis, &transformed_x);
    experimental::OpCommandHelper::BroadcastTo(
        dev_ctx, y, axis, &transformed_y);
    experimental::OpCommand("Mul")
        .Input(transformed_x)
        .Input(transformed_y)
        .Output(*out)
        .Run(dev_ctx);
  }
}

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& y,
                    phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::MultiplyRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MultiplyGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const phi::DenseTensor& y,
                        const phi::DenseTensor& dout,
                        int axis,
                        phi::DenseTensor* dx,
                        phi::DenseTensor* dy) {
  auto stream = dev_ctx.stream();

  axis = (axis == -1 ? std::abs(x.dims().size() - y.dims().size()) : axis);

  phi::DenseTensor trans_x, trans_y;
  // NpuElementWiseOpBroadcast<T>(dev_ctx, &x, &y, axis, &trans_x, &trans_y);
  trans_x.Resize(dout.dims());
  trans_y.Resize(dout.dims());
  dev_ctx.template Alloc<T>(&trans_x);
  dev_ctx.template Alloc<T>(&trans_y);
  experimental::OpCommandHelper::BroadcastTo(dev_ctx, x, axis, &trans_x);
  experimental::OpCommandHelper::BroadcastTo(dev_ctx, y, axis, &trans_y);

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    if (dx->dims() == dout.dims()) {
      // const auto& runner_dx = NpuOpRunner("Mul", {dout, trans_y}, {*dx}, {});
      // runner_dx.Run(stream);
      experimental::OpCommand("Mul").Input(dout).Input(trans_y).Output(*dx).Run(
          dev_ctx);
    } else {
      phi::DenseTensor dx_temp;
      phi::DenseTensorMeta dx_temp_meta = {x.dtype(), trans_x.dims()};
      dx_temp.set_meta(dx_temp_meta);
      dev_ctx.template Alloc<T>(&dx_temp);

      // const auto& runner_dx =
      //     NpuOpRunner("Mul", {dout, trans_y}, {dx_temp}, {});
      // runner_dx.Run(stream);
      // ReduceDims<T>(
      //     dev_ctx, stream, axis, dx->dims(), trans_x.dims(), dx_temp, dx);
      experimental::OpCommand("Mul")
          .Input(dout)
          .Input(trans_y)
          .Output(dx_temp)
          .Run(dev_ctx);
      experimental::OpCommandHelper::ElementwiseGradReduce(
          dev_ctx, dx_temp, axis, dx);
    }
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    if (dy->dims() == dout.dims()) {
      // const auto& runner_dy = NpuOpRunner("Mul", {trans_x, dout}, {*dy}, {});
      // runner_dy.Run(stream);
      experimental::OpCommand("Mul").Input(trans_x).Input(dout).Output(*dy).Run(
          dev_ctx);
    } else {
      phi::DenseTensor dy_temp;
      phi::DenseTensorMeta dy_temp_meta = {y.dtype(), trans_y.dims()};
      dy_temp.set_meta(dy_temp_meta);
      dev_ctx.template Alloc<T>(&dy_temp);

      // const auto& runner_dy =
      //     NpuOpRunner("Mul", {trans_x, dout}, {dy_temp}, {});
      // runner_dy.Run(stream);
      // ReduceDims<T>(
      //     dev_ctx, stream, axis, dy->dims(), trans_y.dims(), dy_temp, dy);
      experimental::OpCommand("Mul")
          .Input(trans_x)
          .Input(dout)
          .Output(dy_temp)
          .Run(dev_ctx);
      experimental::OpCommandHelper::ElementwiseGradReduce(
          dev_ctx, dy_temp, axis, dy);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(multiply_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyRawKernel,
                          int8_t,
                          int32_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(multiply,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyKernel,
                          int8_t,
                          int32_t,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(multiply_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

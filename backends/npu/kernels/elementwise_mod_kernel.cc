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
void ModuloRawKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     int axis,
                     phi::DenseTensor* out) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();

  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);

  bool direct_compute = false;
  if (x_dims.size() >= y_dims.size()) {
    direct_compute = y_dims == phi::slice_ddim(x_dims, axis, x_dims.size());
  } else {
    direct_compute = x_dims == phi::slice_ddim(y_dims, axis, y_dims.size());
  }
  phi::DenseTensor transformed_x(x), transformed_y(y);
  if (y_dims.size() == 0 && x_dims.size() == 0) {
    transformed_x.Resize({1});
    transformed_y.Resize({1});
  } else if (direct_compute) {
    transformed_x = x;
    transformed_y = y;
  } else {
    custom_kernel::NpuElementWiseOpBroadcast<T>(
        dev_ctx, &x, &y, axis, &transformed_x, &transformed_y);
  }
  dev_ctx.template Alloc<T>(out);
  auto stream = dev_ctx.stream();
  if (transformed_x.dtype() != phi::DataType::FLOAT16) {
    const auto& runner =
        NpuOpRunner("FloorMod", {transformed_x, transformed_y}, {*out}, {});
    runner.Run(stream);
  } else {
    // TODO(songkai05): In CANN512, npu op FloorMod returns false results in
    // dtype FLOAT16, so cast inputs to FLOAT32 temporarily until npu fix
    // this problem.
    phi::DenseTensor x_float32, y_float32, out_float32;
    phi::DenseTensorMeta meta = {phi::DataType::FLOAT32, transformed_x.dims()};
    x_float32.set_meta(meta);
    y_float32.set_meta(meta);
    out_float32.set_meta(meta);
    dev_ctx.template Alloc<float>(&x_float32);
    dev_ctx.template Alloc<float>(&y_float32);
    dev_ctx.template Alloc<float>(&out_float32);

    const auto& cast_runner1 = NpuOpRunner(
        "Cast", {transformed_x}, {x_float32}, {{"dst_type", ACL_FLOAT}});
    cast_runner1.Run(stream);

    const auto& cast_runner2 = NpuOpRunner(
        "Cast", {transformed_y}, {y_float32}, {{"dst_type", ACL_FLOAT}});
    cast_runner2.Run(stream);

    const auto& runner =
        NpuOpRunner("FloorMod", {x_float32, y_float32}, {out_float32}, {});
    runner.Run(stream);

    const auto& cast_runner3 =
        NpuOpRunner("Cast", {out_float32}, {*out}, {{"dst_type", ACL_FLOAT16}});
    cast_runner3.Run(stream);
  }
}

template <typename T, typename Context>
void ModuloKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::ModuloRawKernel<T>(dev_ctx, x, y, axis, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(remainder,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ModuloKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(remainder_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::ModuloRawKernel,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::dtype::float16) {}

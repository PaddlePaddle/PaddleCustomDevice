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
void MaximumRawKernel(const Context& dev_ctx,
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
    direct_compute = y_dims == phi::slice_ddim(x_dims, axis, x_dims.size());
  } else {
    direct_compute = x_dims == phi::slice_ddim(y_dims, axis, y_dims.size());
  }

  if (direct_compute) {
    const auto& runner = NpuOpRunner("Maximum", {x, y}, {*out}, {});
    runner.Run(stream);
  } else {
    phi::DenseTensor transformed_x, transformed_y;
    NpuElementWiseOpBroadcast<T>(
        dev_ctx, &x, &y, axis, &transformed_x, &transformed_y);
    const auto& runner =
        NpuOpRunner("Maximum", {transformed_x, transformed_y}, {*out}, {});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::MaximumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MaximumGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       const phi::DenseTensor& dout,
                       int axis,
                       phi::DenseTensor* dx,
                       phi::DenseTensor* dy) {
  auto stream = dev_ctx.stream();

  // The npu elementwise_max_grad op only supports broadcast
  // when axis is -1, and requires all the inputs must have the
  // same shape when axis is not -1. For convenience, we should
  // broadcast the original input x and y to transformed_x and
  // transformed_x firstly, then use tmp tensor to get the op
  // output, last reduce the tmp tensor shape to match the
  // paddle output.

  auto x_dims = x.dims();
  auto y_dims = y.dims();
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
  phi::DenseTensor transformed_x, transformed_y;
  NpuElementWiseOpBroadcast<T>(
      dev_ctx, &x, &y, axis, &transformed_x, &transformed_y);

  auto dout_dims = dout.dims();
  NPUAttributeMap attr_input = {{"grad_x", true}, {"grad_y", true}};
  // Reshape info vector.
  std::vector<int> reduce_axes;

  if (dx && dy) {
    dev_ctx.template Alloc<T>(dx);
    dev_ctx.template Alloc<T>(dy);

    phi::DenseTensor tmp_dx;
    tmp_dx.Resize(dout_dims);
    dev_ctx.template Alloc<T>(&tmp_dx);

    phi::DenseTensor tmp_dy;
    tmp_dy.Resize(dout_dims);
    dev_ctx.template Alloc<T>(&tmp_dy);

    const auto& runner = NpuOpRunner("MaximumGrad",
                                     {dout, transformed_x, transformed_y},
                                     {tmp_dx, tmp_dy},
                                     attr_input);
    runner.Run(stream);

    if (x_dims != dout_dims) {
      reduce_axes.clear();
      int src_axis = (x_dims.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis || ax >= src_axis + x_dims.size()) ||
            (dout_dims[ax] > 1 && x_dims[ax - src_axis] == 1)) {
          reduce_axes.push_back(ax);
        }
      }
      if (!reduce_axes.empty()) {
        const auto& runner =
            NpuOpRunner("ReduceSumD",
                        {tmp_dx},
                        {*dx},
                        {{"axes", reduce_axes}, {"keep_dims", false}});
        runner.Run(stream);
      }
    } else {
      TensorCopy(dev_ctx, tmp_dx, false, dx);
    }

    if (y_dims != dout_dims) {
      reduce_axes.clear();
      int src_axis = (y_dims.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis || ax >= src_axis + y_dims.size()) ||
            (dout_dims[ax] > 1 && y_dims[ax - src_axis] == 1)) {
          reduce_axes.push_back(ax);
        }
      }
      if (!reduce_axes.empty()) {
        const auto& runner =
            NpuOpRunner("ReduceSumD",
                        {tmp_dy},
                        {*dy},
                        {{"axes", reduce_axes}, {"keep_dims", false}});
        runner.Run(stream);
      }
    } else {
      TensorCopy(dev_ctx, tmp_dy, false, dy);
    }

  } else if (dx) {
    phi::DenseTensor zero_tensor;
    phi::DenseTensorMeta zero_tensor_meta = {dout.dtype(), dout_dims};
    zero_tensor.set_meta(zero_tensor_meta);
    dev_ctx.template Alloc<T>(&zero_tensor);
    FillNpuTensorWithConstant<T>(&zero_tensor, dev_ctx, static_cast<T>(0));

    dev_ctx.template Alloc<T>(dx);

    phi::DenseTensor tmp_dx;
    phi::DenseTensorMeta tmp_dx_meta = {dout.dtype(), dout_dims};
    tmp_dx.set_meta(tmp_dx_meta);
    dev_ctx.template Alloc<T>(&tmp_dx);

    const auto& runner = NpuOpRunner("MaximumGrad",
                                     {dout, transformed_x, transformed_y},
                                     {tmp_dx, zero_tensor},
                                     attr_input);
    runner.Run(stream);

    if (x_dims != dout_dims) {
      reduce_axes.clear();

      int src_axis = (x_dims.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis || ax >= src_axis + x_dims.size()) ||
            (dout_dims[ax] > 1 && x_dims[ax - src_axis] == 1)) {
          reduce_axes.push_back(ax);
        }
      }
      if (!reduce_axes.empty()) {
        const auto& runner =
            NpuOpRunner("ReduceSumD",
                        {tmp_dx},
                        {*dx},
                        {{"axes", reduce_axes}, {"keep_dims", false}});
        runner.Run(stream);
      }
    } else {
      TensorCopy(dev_ctx, tmp_dx, false, dx);
    }

  } else if (dy) {
    phi::DenseTensor zero_tensor;
    phi::DenseTensorMeta zero_tensor_meta = {dout.dtype(), dout_dims};
    zero_tensor.set_meta(zero_tensor_meta);
    dev_ctx.template Alloc<T>(&zero_tensor);
    FillNpuTensorWithConstant<T>(&zero_tensor, dev_ctx, static_cast<T>(0));

    dev_ctx.template Alloc<T>(dy);

    phi::DenseTensor tmp_dy;
    phi::DenseTensorMeta tmp_dy_meta = {dout.dtype(), dout_dims};
    tmp_dy.set_meta(tmp_dy_meta);
    dev_ctx.template Alloc<T>(&tmp_dy);

    const auto& runner = NpuOpRunner("MaximumGrad",
                                     {dout, transformed_x, transformed_y},
                                     {zero_tensor, tmp_dy},
                                     attr_input);
    runner.Run(stream);

    if (y_dims != dout_dims) {
      reduce_axes.clear();

      int src_axis = (y_dims.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis || ax >= src_axis + y_dims.size()) ||
            (dout_dims[ax] > 1 && y_dims[ax - src_axis] == 1)) {
          reduce_axes.push_back(ax);
        }
      }
      if (!reduce_axes.empty()) {
        const auto& runner =
            NpuOpRunner("ReduceSumD",
                        {tmp_dy},
                        {*dy},
                        {{"axes", reduce_axes}, {"keep_dims", false}});
        runner.Run(stream);
      }
    } else {
      TensorCopy(dev_ctx, tmp_dy, false, dy);
    }
  } else {
    PADDLE_THROW(
        phi::errors::Unavailable("Do not support all outputs to be empty."));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(maximum_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MaximumRawKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(maximum,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MaximumKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(maximum_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MaximumGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

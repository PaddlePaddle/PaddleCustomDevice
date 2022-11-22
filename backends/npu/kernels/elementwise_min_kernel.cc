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
void MinimumRawKernel(const Context& dev_ctx,
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
  phi::DenseTensor transformed_x, transformed_y;
  if (direct_compute) {
    transformed_x = x;
    transformed_y = y;
  } else {
    NpuElementWiseOpBroadcast<T>(
        dev_ctx, &x, &y, axis, &transformed_x, &transformed_y);
  }
  const auto& runner =
      NpuOpRunner("Minimum", {transformed_x, transformed_y}, {*out}, {});
  runner.Run(stream);
}

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::MinimumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MinimumGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       const phi::DenseTensor& dout,
                       int axis,
                       phi::DenseTensor* dx,
                       phi::DenseTensor* dy) {
  auto stream = dev_ctx.stream();

  axis = (axis == -1 ? std::abs(x.dims().size() - y.dims().size()) : axis);
  if (dx && dy) {
    // dx
    dev_ctx.template Alloc<T>(dx);

    phi::DenseTensor tmp_x(*dx);

    if (dx->dims() != dout.dims()) {
      std::vector<int> dst_dims_vec_x;
      std::vector<int> reduce_axes_x;
      auto src_dims_x = dx->dims();
      auto dout_dims = dout.dims();

      int src_axis_x = (src_dims_x.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis_x || ax >= src_axis_x + src_dims_x.size()) ||
            (dout_dims[ax] > 1 && src_dims_x[ax - src_axis_x] == 1)) {
          reduce_axes_x.push_back(ax);
        } else {
          dst_dims_vec_x.push_back(dout_dims[ax]);
        }
      }
      if (!reduce_axes_x.empty()) {
        tmp_x.Resize(phi::make_ddim(dst_dims_vec_x));
      }
    }
    // dy
    dev_ctx.template Alloc<T>(dy);
    phi::DenseTensor tmp_y(*dy);
    if (dy->dims() != dout.dims()) {
      std::vector<int> dst_dims_vec_y;
      std::vector<int> reduce_axes_y;
      auto src_dims_y = dy->dims();
      auto dout_dims = dout.dims();

      int src_axis_y = (src_dims_y.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis_y || ax >= src_axis_y + src_dims_y.size()) ||
            (dout_dims[ax] > 1 && src_dims_y[ax - src_axis_y] == 1)) {
          reduce_axes_y.push_back(ax);
        } else {
          dst_dims_vec_y.push_back(dout_dims[ax]);
        }
      }
      if (!reduce_axes_y.empty()) {
        tmp_y.Resize(phi::make_ddim(dst_dims_vec_y));
      }
    }

    const auto& runner = NpuOpRunner("MinimumGrad",
                                     {dout, x, y},
                                     {tmp_x, tmp_y},
                                     {{"grad_x", true}, {"grad_y", true}});
    runner.Run(stream);

  } else if (dx) {
    phi::DenseTensor zero_tensor;
    phi::DenseTensorMeta zero_tensor_meta = {dout.dtype(), y.dims()};
    zero_tensor.set_meta(zero_tensor_meta);
    dev_ctx.template Alloc<T>(&zero_tensor);
    FillNpuTensorWithConstant<T>(&zero_tensor, dev_ctx, static_cast<T>(0));

    // dx
    dev_ctx.template Alloc<T>(dx);
    phi::DenseTensor tmp_x(*dx);

    if (dx->dims() != dout.dims()) {
      std::vector<int> dst_dims_vec_x;
      std::vector<int> reduce_axes_x;
      auto src_dims_x = dx->dims();
      auto dout_dims = dout.dims();

      int src_axis_x = (src_dims_x.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis_x || ax >= src_axis_x + src_dims_x.size()) ||
            (dout_dims[ax] > 1 && src_dims_x[ax - src_axis_x] == 1)) {
          reduce_axes_x.push_back(ax);
        } else {
          dst_dims_vec_x.push_back(dout_dims[ax]);
        }
      }
      if (!reduce_axes_x.empty()) {
        tmp_x.Resize(phi::make_ddim(dst_dims_vec_x));
      }
    }

    const auto& runner = NpuOpRunner("MinimumGrad",
                                     {dout, x, y},
                                     {tmp_x, zero_tensor},
                                     {{"grad_x", true}, {"grad_y", true}});
    runner.Run(stream);

  } else if (dy) {
    phi::DenseTensor zero_tensor;
    phi::DenseTensorMeta zero_tensor_meta = {dout.dtype(), x.dims()};
    zero_tensor.set_meta(zero_tensor_meta);
    dev_ctx.template Alloc<T>(&zero_tensor);
    FillNpuTensorWithConstant<T>(&zero_tensor, dev_ctx, static_cast<T>(0));

    // dy
    dev_ctx.template Alloc<T>(dy);
    phi::DenseTensor tmp_y(*dy);
    if (dy->dims() != dout.dims()) {
      std::vector<int> dst_dims_vec_y;
      std::vector<int> reduce_axes_y;
      auto src_dims_y = dy->dims();
      auto dout_dims = dout.dims();

      int src_axis_y = (src_dims_y.size() < dout_dims.size() ? axis : 0);
      for (int ax = 0; ax < dout_dims.size(); ++ax) {
        if ((ax < src_axis_y || ax >= src_axis_y + src_dims_y.size()) ||
            (dout_dims[ax] > 1 && src_dims_y[ax - src_axis_y] == 1)) {
          reduce_axes_y.push_back(ax);
        } else {
          dst_dims_vec_y.push_back(dout_dims[ax]);
        }
      }
      if (!reduce_axes_y.empty()) {
        tmp_y.Resize(phi::make_ddim(dst_dims_vec_y));
      }
    }

    const auto& runner = NpuOpRunner("MinimumGrad",
                                     {dout, x, y},
                                     {zero_tensor, tmp_y},
                                     {{"grad_x", true}, {"grad_y", true}});
    runner.Run(stream);

  } else {
    std::cout << "error" << std::endl;
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(minimum_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MinimumRawKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(minimum,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MinimumKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(minimum_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MinimumGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

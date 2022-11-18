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
void AddRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  LOG(INFO) << "add x.dims: " << x.dims() << " add y.dims: " << y.dims() << " add axis: " << axis;

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
    const auto& runner = NpuOpRunner("Add", {x, y}, {*out}, {});
    runner.Run(stream);
  } else {
    phi::DenseTensor transformed_x, transformed_y;
    NpuElementWiseOpBroadcast<T>(
        dev_ctx, &x, &y, axis, &transformed_x, &transformed_y);
    const auto& runner =
        NpuOpRunner("Add", {transformed_x, transformed_y}, {*out}, {});
    runner.Run(stream);
  }
}

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::AddRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void AddGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   const phi::DenseTensor& dout,
                   int axis,
                   phi::DenseTensor* dx,
                   phi::DenseTensor* dy) {

  LOG(INFO) << "add_grad x.dims: " << x.dims()  << " add_grad y.dims: " << y.dims() <<  " add_grad dout.dims: " << dout.dims() << " add_grad axis: " << axis;

  auto stream = dev_ctx.stream();

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    // For dx
    // stage 1
    auto reduce_ndim = dout.dims().size() - dx->dims().size();
    std::vector<int> axes;
    for (auto i = 0; i < reduce_ndim; ++i) {
      axes.push_back(i);
    }
    phi::DenseTensor* tmp_dout = const_cast<phi::DenseTensor*>(&dout);
    phi::DenseTensor reduced_dout;
    if (axes.size() != 0) {
      std::vector<int64_t> reduced_dout_dims;
      for (auto i = reduce_ndim; i < dout.dims().size(); ++i) {
        reduced_dout_dims.push_back(dout.dims()[i]);
      }

      phi::DenseTensorMeta reduced_dout_meta = {
          dx->dtype(), phi::make_ddim(reduced_dout_dims)};
      reduced_dout.set_meta(reduced_dout_meta);
      dev_ctx.template Alloc<T>(&reduced_dout);

      const auto& runner = NpuOpRunner("ReduceSumD",
                                       {dout},
                                       {reduced_dout},
                                       {{"axes", axes}, {"keep_dims", false}});
      runner.Run(stream);
      tmp_dout = &reduced_dout;
    }

    // stage 2
    axes.clear();
    for (auto i = 0; i < dx->dims().size(); ++i) {
      if (dx->dims()[i] == 1) {
        axes.push_back(i);
      }
    }
    if (axes.size() != 0) {
      const auto& runner = NpuOpRunner("ReduceSumD",
                                       {*tmp_dout},
                                       {*dx},
                                       {{"axes", axes}, {"keep_dims", true}});
      runner.Run(stream);
    } else {
      TensorCopy(dev_ctx, *tmp_dout, false, dx);
    }
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    // For dy
    // stage 1
    auto reduce_ndim = dout.dims().size() - dy->dims().size();
    std::vector<int> axes;
    for (auto i = 0; i < reduce_ndim; ++i) {
      axes.push_back(i);
    }
    phi::DenseTensor* tmp_dout = const_cast<phi::DenseTensor*>(&dout);
    phi::DenseTensor reduced_dout;

    if (axes.size() != 0) {
      std::vector<int64_t> reduced_dout_dims;
      for (auto i = reduce_ndim; i < dout.dims().size(); ++i) {
        reduced_dout_dims.push_back(dout.dims()[i]);
      }

      phi::DenseTensorMeta reduced_dout_meta = {
          dy->dtype(), phi::make_ddim(reduced_dout_dims)};
      reduced_dout.set_meta(reduced_dout_meta);
      dev_ctx.template Alloc<T>(&reduced_dout);

      const auto& runner = NpuOpRunner("ReduceSumD",
                                       {dout},
                                       {reduced_dout},
                                       {{"axes", axes}, {"keep_dims", false}});
      runner.Run(stream);
      tmp_dout = &reduced_dout;
    }

    // stage 2
    axes.clear();
    phi::DenseTensor* tmp_dy = tmp_dout;
    for (auto i = 0; i < dy->dims().size(); ++i) {
      if (dy->dims()[i] == 1) {
        axes.push_back(i);
      }
    }
    if (axes.size() != 0) {
      const auto& runner = NpuOpRunner("ReduceSumD",
                                       {*tmp_dout},
                                       {*dy},
                                       {{"axes", axes}, {"keep_dims", true}});
      runner.Run(stream);
    } else {
      TensorCopy(dev_ctx, *tmp_dout, false, dy);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AddRawKernel,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(add,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AddKernel,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(add_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AddGradKernel,
                          float,
                          phi::dtype::float16) {}

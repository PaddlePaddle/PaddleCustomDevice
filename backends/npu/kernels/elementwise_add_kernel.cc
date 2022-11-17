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

inline phi::DDim GetDimsWithAxis(const phi::DDim& x_dims,
                                 const phi::DDim& y_dims,
                                 const int axis) {
  std::vector<int64_t> y_shape(x_dims.size(), 1);
  for (int i = 0; i < axis; ++i) {
    y_shape[i] = 1;
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    bool is_common_boardcast = x_dims[i + axis] == y_dims[i]
                                   ? true
                                   : (y_dims[i] == 1 || x_dims[i + axis] == 1);
    PADDLE_ENFORCE_EQ(is_common_boardcast,
                      true,
                      phi::errors::InvalidArgument(
                          "Broadcast dimension mismatch. Operands "
                          "could not be broadcast together with the shape of "
                          "X = [%s] and the shape of Y = [%s]. Received [%d] "
                          "in X is not equal to [%d] in Y.",
                          x_dims,
                          y_dims,
                          x_dims[i + axis],
                          y_dims[i]));
    y_shape[i + axis] = y_dims[i];
  }
  for (int i = axis + y_dims.size(); i < x_dims.size(); ++i) {
    y_shape[i] = 1;
  }
  return phi::make_ddim(y_shape);
}

template <typename T, typename Context>
void AddRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  if (x.storage_properties_initialized()) {
    auto npu_properties = x.storage_properties<phi::NPUStorageProperties>();
    int64_t storage_format = npu_properties.storage_format;
    AllocNPUTensor<T>(dev_ctx, aclFormat(storage_format), out);
  } else {
    dev_ctx.template Alloc<T>(out);
  }

  auto stream = dev_ctx.stream();

  bool direct_compute = false;
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);

  if (direct_compute) {
    experimental::OpCommand("Add").Input(x).Input(y).Output(*out).Run(dev_ctx);
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
    experimental::OpCommand("Add")
        .Input(transformed_x)
        .Input(transformed_y)
        .Output(*out)
        .Run(dev_ctx);
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
  auto stream = dev_ctx.stream();

  axis = (axis == -1 ? std::abs(x.dims().size() - y.dims().size()) : axis);

  if (dx) {
    if (x.storage_properties_initialized()) {
      auto npu_properties = x.storage_properties<phi::NPUStorageProperties>();
      int64_t storage_format = npu_properties.storage_format;
      AllocNPUTensor<T>(dev_ctx, aclFormat(storage_format), dx);
    } else {
      dev_ctx.template Alloc<T>(dx);
    }

    if (dx->dims() != dout.dims()) {
      experimental::OpCommandHelper::ElementwiseGradReduce(
          dev_ctx, dout, axis, dx);
    } else {
      experimental::OpCommandHelper::Assign(dev_ctx, dout, dx);
    }
  }
  if (dy) {
    if (y.storage_properties_initialized()) {
      auto npu_properties = y.storage_properties<phi::NPUStorageProperties>();
      int64_t storage_format = npu_properties.storage_format;
      AllocNPUTensor<T>(dev_ctx, aclFormat(storage_format), dy);
    } else {
      dev_ctx.template Alloc<T>(dy);
    }

    if (dy->dims() != dout.dims()) {
      experimental::OpCommandHelper::ElementwiseGradReduce(
          dev_ctx, dout, axis, dy);
    } else {
      experimental::OpCommandHelper::Assign(dev_ctx, dout, dy);
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add_raw,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AddRawKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(add,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AddKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(add_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AddGradKernel,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          double) {}

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

#include "kernels/funcs/elementwise_utils.h"

namespace custom_kernel {

template <typename T, typename Context>
void ElementwisePowRawKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& y,
                             int axis,
                             phi::DenseTensor* out) {
  MLUBinaryOp<POW, T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          phi::DenseTensor* out) {
  int axis = -1;
  custom_kernel::ElementwisePowRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void ElementwisePowGradKernel(const Context& dev_ctx,
                              const phi::DenseTensor& x,
                              const phi::DenseTensor& y,
                              const phi::DenseTensor& dout,
                              int axis,
                              phi::DenseTensor* dx,
                              phi::DenseTensor* dy) {
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  axis = (axis < 0 ? std::abs(x_dims.size() - y_dims.size()) + axis + 1 : axis);

  int max_dim = std::max(x_dims.size(), y_dims.size());
  std::vector<int> x_dims_array(max_dim);
  std::vector<int> y_dims_array(max_dim);
  std::vector<int> out_dims_array(max_dim);
  GetBroadcastDimsArrays(x_dims,
                         y_dims,
                         x_dims_array.data(),
                         y_dims_array.data(),
                         out_dims_array.data(),
                         max_dim,
                         axis);
  cnnlDataType_t data_type = ToCnnlDataType<T>();
  MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(), data_type);
  MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(), data_type);
  MLUCnnlTensorDesc out_desc(max_dim, out_dims_array.data(), data_type);

  auto dout_dims = dout.dims();
  if (dx) {
    // dx = dout * y * pow(x, y - 1);
    Tensor one_dx;
    one_dx.Resize(phi::make_ddim(y_dims_array));
    dev_ctx.template Alloc<T>(&one_dx);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(1), &one_dx);

    Tensor sub_dx;
    sub_dx.Resize(phi::make_ddim(y_dims_array));
    dev_ctx.template Alloc<T>(&sub_dx);
    MLUCnnlOpTensorDesc op_tensor_desc(
        CNNL_OP_TENSOR_SUB, data_type, CNNL_NOT_PROPAGATE_NAN);
    MLUCnnl::OpTensor(dev_ctx,
                      op_tensor_desc.get(),
                      y_desc.get(),
                      GetBasePtr(&y),
                      y_desc.get(),
                      GetBasePtr(&one_dx),
                      y_desc.get(),
                      GetBasePtr(&sub_dx),
                      data_type);

    Tensor tmp_dx;
    tmp_dx.Resize(phi::make_ddim(out_dims_array));
    dev_ctx.template Alloc<T>(&tmp_dx);
    MLUCnnl::Pow(dev_ctx,
                 CNNL_COMPUTATION_HIGH_PRECISION,
                 x_desc.get(),
                 GetBasePtr(&x),
                 y_desc.get(),
                 GetBasePtr(&sub_dx),
                 out_desc.get(),
                 GetBasePtr(&tmp_dx));

    MLUCnnl::MulAx(dev_ctx,
                   y_desc.get(),
                   GetBasePtr(&y),
                   out_desc.get(),
                   GetBasePtr(&tmp_dx));
    MLUCnnl::MulAx(dev_ctx,
                   out_desc.get(),
                   GetBasePtr(&dout),
                   out_desc.get(),
                   GetBasePtr(&tmp_dx));

    if (x_dims != dout_dims) {
      dev_ctx.template Alloc<T>(dx);
      std::vector<int> reduce_axes;
      GetReduceAxes(axis, dout_dims, x_dims, &reduce_axes);
      if (!reduce_axes.empty()) {
        MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                         CNNL_REDUCE_ADD,
                                         data_type,
                                         CNNL_NOT_PROPAGATE_NAN,
                                         CNNL_REDUCE_NO_INDICES,
                                         CNNL_32BIT_INDICES);
        MLUCnnlTensorDesc dx_desc(*dx);
        MLUCnnl::Reduce(dev_ctx,
                        true /*need_workspace*/,
                        reduction_desc.get(),
                        nullptr,
                        out_desc.get(),
                        GetBasePtr(&tmp_dx),
                        0,
                        nullptr,
                        nullptr,
                        dx_desc.get(),
                        GetBasePtr(dx));
      }
    } else {
      *dx = tmp_dx;
    }
  }
  if (dy) {
    // dy = dout * log(x) * pow(x, y)
    Tensor tmp_dy;
    tmp_dy.Resize(phi::make_ddim(out_dims_array));
    dev_ctx.template Alloc<T>(&tmp_dy);
    MLUCnnl::Pow(dev_ctx,
                 CNNL_COMPUTATION_HIGH_PRECISION,
                 x_desc.get(),
                 GetBasePtr(&x),
                 y_desc.get(),
                 GetBasePtr(&y),
                 out_desc.get(),
                 GetBasePtr(&tmp_dy));

    Tensor log_x;
    log_x.Resize(x.dims());
    dev_ctx.template Alloc<T>(&log_x);
    MLUCnnl::Log(dev_ctx,
                 CNNL_COMPUTATION_HIGH_PRECISION,
                 CNNL_LOG_E,
                 x_desc.get(),
                 GetBasePtr(&x),
                 x_desc.get(),
                 GetBasePtr(&log_x));
    MLUCnnl::MulAx(dev_ctx,
                   x_desc.get(),
                   GetBasePtr(&log_x),
                   out_desc.get(),
                   GetBasePtr(&tmp_dy));
    MLUCnnl::MulAx(dev_ctx,
                   out_desc.get(),
                   GetBasePtr(&dout),
                   out_desc.get(),
                   GetBasePtr(&tmp_dy));

    if (y_dims != dout_dims) {
      dev_ctx.template Alloc<T>(dy);
      std::vector<int> reduce_axes;
      GetReduceAxes(axis, dout_dims, y_dims, &reduce_axes);
      if (!reduce_axes.empty()) {
        MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                         CNNL_REDUCE_ADD,
                                         data_type,
                                         CNNL_NOT_PROPAGATE_NAN,
                                         CNNL_REDUCE_NO_INDICES,
                                         CNNL_32BIT_INDICES);
        MLUCnnlTensorDesc dy_desc(*dy);
        MLUCnnl::Reduce(dev_ctx,
                        true /*need_workspace*/,
                        reduction_desc.get(),
                        nullptr,
                        out_desc.get(),
                        GetBasePtr(&tmp_dy),
                        0,
                        nullptr,
                        nullptr,
                        dy_desc.get(),
                        GetBasePtr(dy));
      }
    } else {
      *dy = tmp_dy;
    }
  }
  if (!dx && !dy) {
    PADDLE_THROW(
        phi::errors::Unavailable("Not support all outputs to be empty."));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(elementwise_pow,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ElementwisePowKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(elementwise_pow_raw,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ElementwisePowRawKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(elementwise_pow_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ElementwisePowGradKernel,
                          float,
                          phi::dtype::float16) {}

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
void MultiplyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& y,
                       int axis,
                       phi::DenseTensor* out) {
  MLUOpTensorKernel<T>(dev_ctx, x, y, axis, CNNL_OP_TENSOR_MUL, out);
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
  const auto& x_dims = x.dims();
  const auto& y_dims = y.dims();
  axis =
      (axis < 0 ? (std::abs(x_dims.size() - y_dims.size()) + axis + 1) : axis);
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

  MLUCnnlTensorDesc x_desc(max_dim, x_dims_array.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc y_desc(max_dim, y_dims_array.data(), ToCnnlDataType<T>());
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnlOpTensorDesc mul_op_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    if (dx->dims() == dout.dims()) {
      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        dout_desc.get(),
                        GetBasePtr(&dout),
                        y_desc.get(),
                        GetBasePtr(&y),
                        x_desc.get(),
                        GetBasePtr(dx),
                        ToCnnlDataType<T>());
    } else {
      Tensor dx_temp;
      dx_temp.Resize(dout.dims());
      dev_ctx.template Alloc<T>(&dx_temp);
      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        dout_desc.get(),
                        GetBasePtr(&dout),
                        y_desc.get(),
                        GetBasePtr(&y),
                        dout_desc.get(),
                        GetBasePtr(&dx_temp),
                        ToCnnlDataType<T>());

      std::vector<int> reduce_axes;
      GetReduceAxes(axis, dx_temp.dims(), dx->dims(), &reduce_axes);
      MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                       CNNL_REDUCE_ADD,
                                       ToCnnlDataType<T>(),
                                       CNNL_NOT_PROPAGATE_NAN,
                                       CNNL_REDUCE_NO_INDICES,
                                       CNNL_32BIT_INDICES);
      MLUCnnlTensorDesc dx_desc(*dx);
      MLUCnnl::Reduce(dev_ctx,
                      true /*need_workspace*/,
                      reduction_desc.get(),
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dx_temp),
                      0,
                      nullptr,
                      nullptr,
                      dx_desc.get(),
                      GetBasePtr(dx));
    }
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    if (dy->dims() == dout.dims()) {
      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        dout_desc.get(),
                        GetBasePtr(&dout),
                        x_desc.get(),
                        GetBasePtr(&x),
                        y_desc.get(),
                        GetBasePtr(dy),
                        ToCnnlDataType<T>());
    } else {
      Tensor dy_temp;
      dy_temp.Resize(dout.dims());
      dev_ctx.template Alloc<T>(&dy_temp);
      MLUCnnl::OpTensor(dev_ctx,
                        mul_op_desc.get(),
                        dout_desc.get(),
                        GetBasePtr(&dout),
                        x_desc.get(),
                        GetBasePtr(&x),
                        dout_desc.get(),
                        GetBasePtr(&dy_temp),
                        ToCnnlDataType<T>());

      std::vector<int> reduce_axes;
      GetReduceAxes(axis, dy_temp.dims(), dy->dims(), &reduce_axes);
      MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                       CNNL_REDUCE_ADD,
                                       ToCnnlDataType<T>(),
                                       CNNL_NOT_PROPAGATE_NAN,
                                       CNNL_REDUCE_NO_INDICES,
                                       CNNL_32BIT_INDICES);
      MLUCnnlTensorDesc dy_desc(*dy);
      MLUCnnl::Reduce(dev_ctx,
                      true /*need_workspace*/,
                      reduction_desc.get(),
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dy_temp),
                      0,
                      nullptr,
                      nullptr,
                      dy_desc.get(),
                      GetBasePtr(dy));
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(multiply_raw,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyRawKernel,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(multiply,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyKernel,
                          int,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(multiply_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::MultiplyGradKernel,
                          int,
                          float,
                          phi::dtype::float16) {}

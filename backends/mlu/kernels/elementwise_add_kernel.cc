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
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void AddRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& y,
                  int axis,
                  phi::DenseTensor* out) {
  MLUOpTensorKernel<T>(dev_ctx, x, y, axis, CNNL_OP_TENSOR_ADD, out);
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
  axis = (axis == -1 ? std::abs(x.dims().size() - y.dims().size()) : axis);
  MLUCnnlTensorDesc dout_desc(dout);
  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    if (dx->dims() != dout.dims()) {
      std::vector<int> dst_dims_vec;
      std::vector<int> reduce_axes;
      GetReduceAxesAndDstDims(
          axis, dout.dims(), dx->dims(), &reduce_axes, &dst_dims_vec);

      MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                       CNNL_REDUCE_ADD,
                                       ToCnnlDataType<T>(),
                                       CNNL_NOT_PROPAGATE_NAN,
                                       CNNL_REDUCE_NO_INDICES,
                                       CNNL_32BIT_INDICES);
      MLUCnnlTensorDesc dx_desc(
          dst_dims_vec.size(), dst_dims_vec.data(), ToCnnlDataType<T>());
      MLUCnnl::Reduce(dev_ctx,
                      true /*need_workspace*/,
                      reduction_desc.get(),
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      0,
                      nullptr,
                      nullptr,
                      dx_desc.get(),
                      GetBasePtr(dx));
    } else {
      *dx = dout;
    }
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    if (dy->dims() != dout.dims()) {
      std::vector<int> dst_dims_vec;
      std::vector<int> reduce_axes;
      GetReduceAxesAndDstDims(
          axis, dout.dims(), dy->dims(), &reduce_axes, &dst_dims_vec);

      MLUCnnlReduceDesc reduction_desc(reduce_axes,
                                       CNNL_REDUCE_ADD,
                                       ToCnnlDataType<T>(),
                                       CNNL_NOT_PROPAGATE_NAN,
                                       CNNL_REDUCE_NO_INDICES,
                                       CNNL_32BIT_INDICES);
      MLUCnnlTensorDesc dy_desc(
          dst_dims_vec.size(), dst_dims_vec.data(), ToCnnlDataType<T>());
      MLUCnnl::Reduce(dev_ctx,
                      true /*need_workspace*/,
                      reduction_desc.get(),
                      nullptr,
                      dout_desc.get(),
                      GetBasePtr(&dout),
                      0,
                      nullptr,
                      nullptr,
                      dy_desc.get(),
                      GetBasePtr(dy));
    } else {
      *dy = dout;
    }
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(add_raw,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AddRawKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(add,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AddKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(add_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AddGradKernel,
                          float,
                          phi::dtype::float16) {}

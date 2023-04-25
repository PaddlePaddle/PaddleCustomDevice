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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void GatherKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::DenseTensor& index,
                  const phi::Scalar& axis,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  const auto index_dims = index.dims();
  if (index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(
        index_dims[1],
        1,
        phi::errors::InvalidArgument(
            "The last dim of index should be 1 when it is 2D, but we get %d",
            index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        index_dims.size() == 1 || index_dims.size() == 0,
        true,
        phi::errors::InvalidArgument(
            "The index should be 0D or 1D, when it is not 2D, but we get %d",
            index_dims.size()));
  }

  auto axis_v = axis.to<int>();

  MLUCnnlTensorDesc x_desc(x);
  int index_shape_1d[1] = {static_cast<int>(index_dims[0])};
  MLUCnnlTensorDesc index_desc(
      1, index_shape_1d, ToCnnlDataType(index.dtype()));
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::GatherFunctor(dev_ctx,
                         axis_v,
                         0 /*batch_dims*/,
                         x_desc.get(),
                         GetBasePtr(&x),
                         index_desc.get(),
                         GetBasePtr(&index),
                         out_desc.get(),
                         GetBasePtr(out));
}

template <typename T, typename Context>
void GatherGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& index,
                      const phi::DenseTensor& out_grad,
                      const phi::Scalar& axis,
                      phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  const auto index_dims = index.dims();
  if (index_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(
        index_dims[1],
        1,
        phi::errors::InvalidArgument(
            "The last dim of index should be 1 when it is 2D, but we get %d",
            index_dims[1]));
  } else {
    PADDLE_ENFORCE_EQ(
        index_dims.size() == 1 || index_dims.size() == 0,
        true,
        phi::errors::InvalidArgument(
            "The index should be 0D or 1D, when it is not 2D, but we get %d",
            index_dims.size()));
  }

  MLUCnnlTensorDesc dx_desc(*x_grad);
  auto value = static_cast<T>(0);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &value,
                dx_desc.get(),
                GetBasePtr(x_grad));

  int index_shape_1d[1] = {static_cast<int>(index_dims[0])};
  MLUCnnlTensorDesc index_desc(
      1, index_shape_1d, ToCnnlDataType(index.dtype()));
  MLUCnnlTensorDesc dout_desc(out_grad);
  const cnnlScatterRefMode_t mode = CNNL_SCATTERREF_UPDATE;
  MLUCnnl::ScatterRefFunctor(dev_ctx,
                             dx_desc.get(),
                             GetBasePtr(x_grad),
                             dout_desc.get(),
                             GetBasePtr(&out_grad),
                             index_desc.get(),
                             GetBasePtr(&index),
                             mode);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(gather,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GatherKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(gather_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::GatherGradKernel,
                          float,
                          phi::dtype::float16) {}

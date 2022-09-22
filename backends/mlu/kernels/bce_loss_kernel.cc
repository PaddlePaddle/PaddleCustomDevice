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

namespace custom_kernel {

template <typename T, typename Context>
void BCELossKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& labels,
                   phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc label_desc(labels);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::BceLoss(dev_ctx,
                   CNNL_BCE_LOSS_NONE,
                   x_desc.get(),
                   GetBasePtr(&x),
                   label_desc.get(),
                   GetBasePtr(&labels),
                   nullptr,
                   nullptr,
                   out_desc.get(),
                   GetBasePtr(out));
}

template <typename T, typename Context>
void BCELossGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& labels,
                       const phi::DenseTensor& dout,
                       phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc label_desc(labels);
  MLUCnnlTensorDesc dout_desc(dout);
  MLUCnnl::BceLossBackward(dev_ctx,
                           CNNL_BCE_LOSS_NONE,
                           dout_desc.get(),
                           GetBasePtr(&dout),
                           x_desc.get(),
                           GetBasePtr(&x),
                           label_desc.get(),
                           GetBasePtr(&labels),
                           nullptr,
                           nullptr,
                           x_desc.get(),
                           GetBasePtr(dx));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(bce_loss,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::BCELossKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(bce_loss_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::BCELossGradKernel,
                          float,
                          phi::dtype::float16) {}

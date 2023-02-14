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
void TrilTriuKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    int diagonal,
                    bool lower,
                    phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  bool upper;
  if (lower) {
    upper = 0;
  } else {
    upper = 1;
  }

  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::TrilTriu(dev_ctx,
                    diagonal,
                    upper,
                    x_desc.get(),
                    GetBasePtr(&x),
                    out_desc.get(),
                    GetBasePtr(out));
}

template <typename T, typename Context>
void TrilKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                int diagonal,
                phi::DenseTensor* out) {
  custom_kernel::TrilTriuKernel<T, Context>(dev_ctx, x, diagonal, true, out);
}

template <typename T, typename Context>
void TriuKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                int diagonal,
                phi::DenseTensor* out) {
  custom_kernel::TrilTriuKernel<T, Context>(dev_ctx, x, diagonal, false, out);
}

template <typename T, typename Context>
void TrilTriuGradKernel(const Context& dev_ctx,
                        const phi::DenseTensor& out_grad,
                        int diagonal,
                        bool lower,
                        phi::DenseTensor* x_grad) {
  custom_kernel::TrilTriuKernel<T, Context>(
      dev_ctx, out_grad, diagonal, lower, x_grad);
}

template <typename T, typename Context>
void TrilGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out_grad,
                    int diagonal,
                    phi::DenseTensor* x_grad) {
  custom_kernel::TrilTriuGradKernel<T, Context>(
      dev_ctx, out_grad, diagonal, true, x_grad);
}

template <typename T, typename Context>
void TriuGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& out_grad,
                    int diagonal,
                    phi::DenseTensor* x_grad) {
  custom_kernel::TrilTriuGradKernel<T, Context>(
      dev_ctx, out_grad, diagonal, false, x_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tril_triu,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::TrilTriuKernel,
                          float,
                          int,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tril,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::TrilKernel,
                          float,
                          int,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(triu,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::TriuKernel,
                          float,
                          int,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tril_triu_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::TrilTriuGradKernel,
                          float,
                          int,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tril_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::TrilGradKernel,
                          float,
                          int,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(triu_grad,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::TriuGradKernel,
                          float,
                          int,
                          phi::dtype::float16) {}

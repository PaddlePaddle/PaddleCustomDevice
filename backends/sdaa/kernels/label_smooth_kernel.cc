// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void LabelSmoothKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const paddle::optional<phi::DenseTensor>& dist,
                       float epsilon,
                       phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA LabelSmoothKernel";
  auto label_dim = x.dims()[x.dims().size() - 1];
  dev_ctx.template Alloc<T>(out);
  // (1 − epsilon) ∗ x
  phi::DenseTensor x_temp;
  x_temp.Resize(x.dims());
  dev_ctx.template Alloc<T>(&x_temp);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, x, 1 - epsilon, UnaryOpMode::MUL_A, &x_temp);

  if (dist) {
    // epsilon * dist
    auto& dist_tensor = dist.get();
    phi::DenseTensor dist_temp;
    dist_temp.Resize({1, label_dim});
    dev_ctx.template Alloc<T>(&dist_temp);
    sdaa_ops::doUnaryOpTensor(
        dev_ctx, dist_tensor, epsilon, UnaryOpMode::MUL_A, &dist_temp);
    // (1 − epsilon) ∗ x + epsilon * dist
    sdaa_ops::doElementAdd(dev_ctx, x_temp, dist_temp, -1, out);
  } else {
    sdaa_ops::doUnaryOpTensor(
        dev_ctx, x_temp, (epsilon / label_dim), UnaryOpMode::ADD_A, out);
  }
}

template <typename T, typename Context>
void LabelSmoothGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& dout,
                           float epsilon,
                           phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA LabelSmoothGradKernel";
  dev_ctx.template Alloc<T>(dx);
  sdaa_ops::doUnaryOpTensor(dev_ctx, dout, 1 - epsilon, UnaryOpMode::MUL_A, dx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(label_smooth,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LabelSmoothKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(label_smooth_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LabelSmoothGradKernel,
                          float,
                          phi::dtype::float16) {}

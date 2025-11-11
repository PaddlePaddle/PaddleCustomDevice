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

#include "kernels/funcs/tblas_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void BmmKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::DenseTensor& y,
               phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA BmmKernel.";

  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 0 || y.numel() == 0) {
    return;
  }

  tblas_ops::BatchMatmul<T>(dev_ctx, x, y, false, false, out);
}

template <typename T, typename Context>
void BmmGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& y,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx,
                   phi::DenseTensor* dy) {
  VLOG(4) << "CALL SDAA BmmGradKernel.";

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    tblas_ops::BatchMatmul<T>(dev_ctx, dout, y, false, true, dx);
  }

  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    tblas_ops::BatchMatmul<T>(dev_ctx, x, dout, true, false, dy);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(bmm,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::BmmKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(bmm_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::BmmGradKernel,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

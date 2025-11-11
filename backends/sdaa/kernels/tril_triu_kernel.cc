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

#include "kernels/funcs/tblas_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void TrilTriuKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    int diagonal,
                    bool lower,
                    phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA TrilTriuKernel";
  dev_ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  int H = x_dims[x_dims.size() - 2];
  int W = x_dims[x_dims.size() - 1];
  int B = phi::product(phi::slice_ddim(x_dims, 0, x_dims.size() - 2));
  if (lower) {
    tblas_ops::TecoBlas<T>::Tril(
        dev_ctx, H, W, B, diagonal, x.data(), out->data());
  } else {
    tblas_ops::TecoBlas<T>::Triu(
        dev_ctx, H, W, B, diagonal, x.data(), out->data());
  }
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

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(tril_triu,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::TrilTriuKernel,
                          bool,
                          double,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(tril,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::TrilKernel,
                          bool,
                          double,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(triu,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::TriuKernel,
                          bool,
                          double,
                          int64_t,
                          float,
                          phi::dtype::float16) {}

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

#include "kernels/amp/amp_funcs.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void CheckFiniteAndUnscale(const Context& dev_ctx,
                           const std::vector<const phi::DenseTensor*>& xs,
                           const phi::DenseTensor& t_scale,
                           std::vector<phi::DenseTensor*> outs,
                           phi::DenseTensor* found_inf) {
  VLOG(4) << "Call SDAA CheckFiniteAndUnscale";

  // step 1: check whether the tensor has nan or inf
  // step 2: scale in tensor
  // out = in, if found_inf = true
  // out = in/scale, if found_inf = false
  dev_ctx.template Alloc<bool>(found_inf);

  for (size_t i = 0; i < xs.size(); ++i) {
    auto* out = outs[i];
    dev_ctx.template Alloc<T>(out);
  }

  amp_funcs::AbnormCheckAndScale<T, Context>(
      dev_ctx, xs, t_scale, outs, found_inf);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(check_finite_and_unscale,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::CheckFiniteAndUnscale,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::BOOL);
}

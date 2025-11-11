// BSD 3- Clause License Copyright (c) 2024, Tecorigin Co., Ltd. All rights
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
void UnbindKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  int axis,
                  std::vector<phi::DenseTensor*> outs) {
  VLOG(4) << "CALL SDAA UnbindKernel.";

  auto x_dims = x.dims();
  axis = axis < 0 ? x_dims.size() + axis : axis;
  int num = x_dims[axis];

  auto split_dims = x.dims();
  split_dims[axis] = 1;

  phi::DDim out_origin_dims;

  for (size_t i = 0; i < outs.size(); i++) {
    if (i == 0) {
      out_origin_dims = outs[0]->dims();
    }
    dev_ctx.template Alloc<T>(outs[i]);
    outs[i]->Resize(split_dims);
  }

  sdaa_ops::doSplitTensor(dev_ctx, x, axis, outs);

  for (size_t i = 0; i < outs.size(); i++) {
    outs[i]->Resize(out_origin_dims);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(unbind,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::UnbindKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          int64_t,
                          int,
                          bool) {}

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

#include "kernels/funcs/sdaa_baseop.h"

namespace custom_kernel {

template <typename T, typename Context>
void ScatterNdAddKernel(const Context &ctx,
                        const phi::DenseTensor &x,
                        const phi::DenseTensor &index,
                        const phi::DenseTensor &updates,
                        phi::DenseTensor *out) {
  VLOG(4) << "Call SDAA ScatterNdAddKernel";

  // In place output: Out = X
  phi::Copy(ctx, x, ctx.GetPlace(), true, out);

  auto r_index_dims = phi::vectorize(index.dims());
  PADDLE_ENFORCE_EQ(r_index_dims[r_index_dims.size() - 1],
                    x.dims().size(),
                    phi::errors::InvalidArgument(
                        "The last dimension of index's shape not equal "
                        "dimension size of x is not support on %s.",
                        ctx.GetPlace()));

  sdaa_ops::doScatterNdAdd<T>(ctx, index, updates, out);
}

}  // namespace custom_kernel

// TODO(zhangrb): sdcops scatter_nd_add not support yolov5/v7
// PD_REGISTER_PLUGIN_KERNEL(scatter_nd_add,
//                           sdaa,
//                           ALL_LAYOUT,
//                           custom_kernel::ScatterNdAddKernel,
//                           float) {}

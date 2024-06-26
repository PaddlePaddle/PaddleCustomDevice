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

#define MAX_RANK_SUPPORTED 6

template <typename T, typename Context>
void ExpandAsKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const paddle::optional<phi::DenseTensor>& y,
                    const std::vector<int>& target_shape,
                    phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA ExpandAsKernel";
  auto rank = x.dims().size();
  auto target_rank = target_shape.size();

  PADDLE_ENFORCE_GE(target_rank,
                    rank,
                    phi::errors::InvalidArgument(
                        "The rank (%d) of the input 'target_tensor' for "
                        "expand_as_v2 op must be greater than or equal to "
                        "the rank (%d) of the input 'x'.",
                        target_rank,
                        rank));
  PADDLE_ENFORCE_GE(
      rank,
      0,
      phi::errors::InvalidArgument("The rank (%d) of the input 'x' for "
                                   "expand_as_v2 op must be positive.",
                                   rank));
  PADDLE_ENFORCE_LE(target_rank,
                    MAX_RANK_SUPPORTED,
                    phi::errors::InvalidArgument(
                        "The rank (%d) of the input 'target_tensor' for "
                        "expand_as_v2 op must be less than or equal to %d.",
                        target_rank,
                        MAX_RANK_SUPPORTED));

  // support 0D tensor
  if (target_rank == 0) {
    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  auto in_dims = phi::vectorize<int>(x.dims());
  auto diff = target_rank - rank;
  in_dims.insert(in_dims.begin(), diff, 1);

  for (size_t i = 0; i < in_dims.size(); i++) {
    PADDLE_ENFORCE_NE(target_shape[i],
                      0,
                      phi::errors::InvalidArgument(
                          "The value of target shape cannot be zero."));
    if (in_dims[i] != 1) {
      PADDLE_ENFORCE_EQ(
          in_dims[i],
          target_shape[i],
          phi::errors::InvalidArgument(
              "The value (%d) of the non-singleton dimension does not match"
              " the corresponding value (%d) in "
              "target tensor for expand_as_v2 op.",
              in_dims[i],
              target_shape[i]));
    }
  }

  out->Resize(phi::make_ddim(target_shape));
  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doExpandTensor(dev_ctx, x, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(expand_as,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ExpandAsKernel,
                          float,
                          phi::dtype::float16,
                          double,
                          int32_t,
                          int64_t,
                          bool) {}

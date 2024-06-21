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

static inline int ComputeAxis(int axis, int rank) {
  PADDLE_ENFORCE_EQ(
      axis >= -rank & axis < rank,
      true,
      phi::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d.",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

template <typename T, typename Context>
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& ins,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA ConcatKernel.";
  dev_ctx.template Alloc<T>(out);

  int axis = axis_scalar.to<int>();
  axis = ComputeAxis(axis, static_cast<int>(ins[0]->dims().size()));
  sdaa_ops::doConcatTensor(dev_ctx, ins, axis, out);
}

template <typename T, typename Context>
void ConcatGradKernel(const Context& dev_ctx,
                      const std::vector<const phi::DenseTensor*>& ins,
                      const phi::DenseTensor& dout,
                      const phi::Scalar& axis_scalar,
                      std::vector<phi::DenseTensor*> outs) {
  VLOG(4) << "CALL SDAA ConcatGradKernel";
  std::vector<phi::DenseTensor*> outputs_vec;
  std::vector<phi::DenseTensor> tmp_outputs_vec;
  int axis = axis_scalar.to<int>();
  axis = ComputeAxis(axis, static_cast<int>(ins[0]->dims().size()));
  for (int i = 0; i < outs.size(); ++i) {
    if (outs[i]) {
      dev_ctx.template Alloc<T>(outs[i]);
      outputs_vec.push_back(outs[i]);
    } else {
      phi::DenseTensor tmp_tensor;
      tmp_tensor.Resize(ins[i]->dims());
      dev_ctx.template Alloc<T>(&tmp_tensor);
      tmp_outputs_vec.push_back((std::move(tmp_tensor)));
      outputs_vec.push_back(&(tmp_outputs_vec.back()));
    }
  }

  sdaa_ops::doSplitTensor(dev_ctx, dout, axis, outputs_vec);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(concat,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ConcatKernel,
                          float,
                          phi::dtype::float16,
                          bool,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(concat_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ConcatGradKernel,
                          float,
                          phi::dtype::float16,
                          bool,
                          int,
                          int64_t) {}

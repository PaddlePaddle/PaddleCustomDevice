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
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void UnStackKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   int axis,
                   int num UNUSED,
                   std::vector<phi::DenseTensor*> outs) {
  VLOG(4) << "Call SDAA UnStackKernel";

  // get outs dims
  std::vector<int> output_dims = phi::vectorize<int>(outs[0]->dims());

  if (axis < 0) axis += x.dims().size();

  std::vector<int> output_dims_origin(output_dims);
  output_dims.insert(output_dims.begin() + axis, 1);

  std::vector<phi::DenseTensor> tmp_outputs_vec;
  tmp_outputs_vec.resize(outs.size());
  std::vector<phi::DenseTensor*> outs_;

  const phi::DenseTensorMeta meta_data(x.dtype(), phi::make_ddim(output_dims));
  for (int i = 0; i < outs.size(); ++i) {
    dev_ctx.template Alloc<T>(outs[i]);
    outs_.push_back(outs[i]);
    outs_[i]->Resize(phi::make_ddim(output_dims));
  }

  sdaa_ops::doSplitTensor(dev_ctx, x, axis, outs_);

  for (int i = 0; i < num; i++) {
    outs_[i]->Resize(phi::make_ddim(output_dims_origin));
  }
}

template <typename T, typename Context>
void UnStackGradKernel(const Context& dev_ctx,
                       const std::vector<const phi::DenseTensor*>& x,
                       int axis,
                       phi::DenseTensor* x_grad) {
  VLOG(4) << "CALL SDAA UnStackGradKernel.";

  dev_ctx.template Alloc<T>(x_grad);

  if (axis < 0) axis += (x[0]->dims().size() + 1);

  int num = static_cast<int>(x.size());

  std::vector<const phi::DenseTensor*> x_;
  std::vector<int> input_dims = phi::vectorize<int>(x[0]->dims());
  input_dims.insert(input_dims.begin() + axis, 1);

  phi::DenseTensor* temp;

  for (int i = 0; i < num; i++) {
    temp = const_cast<phi::DenseTensor*>(x[i]);
    temp->Resize(phi::make_ddim(input_dims));
    x_.push_back(temp);
  }

  sdaa_ops::doConcatTensor(dev_ctx, x_, axis, x_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(unstack,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::UnStackKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          bool,
                          int,
                          uint8_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(unstack_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::UnStackGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          bool,
                          int,
                          uint8_t,
                          int64_t) {}

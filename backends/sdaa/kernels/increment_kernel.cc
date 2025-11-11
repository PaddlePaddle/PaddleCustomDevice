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

#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void IncrementKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     float value,
                     phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA IncrementKernel.";

  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor value_tensor;
  phi::DenseTensorMeta meta = {x.dtype(), {1}};
  value_tensor.set_meta(meta);
  dev_ctx.template Alloc<T>(&value_tensor);

  sdaa_ops::doFillTensor<T>(
      dev_ctx, static_cast<T>(value), x.dtype(), &value_tensor);

  sdaa_ops::doElementAdd(dev_ctx, x, value_tensor, static_cast<int>(-1), out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(increment,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::IncrementKernel,
                          int64_t,
                          int,
                          double,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

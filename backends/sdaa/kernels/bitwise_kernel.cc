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
void BitwiseOrKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA BitwiseOrKernel";
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor x_temp(x), y_temp(y), out_temp(*out);
  if (x.dims().size() == 0 && y.dims().size() == 0) {
    x_temp.Resize(phi::make_ddim({1}));
    y_temp.Resize(phi::make_ddim({1}));
    out_temp.Resize(phi::make_ddim({1}));
  }

  sdaa_ops::doBitwiseBinaryOpTensor(
      dev_ctx, x_temp, y_temp, BitwiseOpType::Or, &out_temp);
}

template <typename T, typename Context>
void BitwiseNotKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA BitwiseNotKernel";
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor x_temp(x), out_temp(*out);
  if (x.dims().size() == 0) {
    x_temp.Resize(phi::make_ddim({1}));
    out_temp.Resize(phi::make_ddim({1}));
  }

  sdaa_ops::doBitwiseUnaryOpTensor(
      dev_ctx, x_temp, BitwiseOpType::Not, &out_temp);
}

template <typename T, typename Context>
void BitwiseAndKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA BitwiseAndKernel";
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor x_temp(x), y_temp(y), out_temp(*out);
  if (x.dims().size() == 0 && y.dims().size() == 0) {
    x_temp.Resize(phi::make_ddim({1}));
    y_temp.Resize(phi::make_ddim({1}));
    out_temp.Resize(phi::make_ddim({1}));
  }

  sdaa_ops::doBitwiseBinaryOpTensor(
      dev_ctx, x_temp, y_temp, BitwiseOpType::And, &out_temp);
}

template <typename T, typename Context>
void BitwiseXorKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& y,
                      phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA BitwiseXorKernel";
  dev_ctx.template Alloc<T>(out);

  phi::DenseTensor x_temp(x), y_temp(y), out_temp(*out);
  if (x.dims().size() == 0 && y.dims().size() == 0) {
    x_temp.Resize(phi::make_ddim({1}));
    y_temp.Resize(phi::make_ddim({1}));
    out_temp.Resize(phi::make_ddim({1}));
  }

  sdaa_ops::doBitwiseBinaryOpTensor(
      dev_ctx, x_temp, y_temp, BitwiseOpType::Xor, &out_temp);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(bitwise_or,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseOrKernel,
                          bool,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(bitwise_not,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseNotKernel,
                          bool,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(bitwise_and,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseAndKernel,
                          bool,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(bitwise_xor,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::BitwiseXorKernel,
                          bool,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t) {}

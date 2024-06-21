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

// out = -label*log(input+epsilon)-(1-label)*log(1-input+epsilon)
template <typename T, typename Context>
void LogLossKernel(const Context& dev_ctx,
                   const phi::DenseTensor& input,
                   const phi::DenseTensor& label,
                   float epsilon,
                   phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA LogLossKernel";
  dev_ctx.template Alloc<T>(out);
  // compute out=-label*log(input+epsilon)
  phi::DenseTensor input_temp;
  input_temp.Resize(input.dims());
  dev_ctx.template Alloc<T>(&input_temp);
  phi::DenseTensor label_temp;
  label_temp.Resize(label.dims());
  dev_ctx.template Alloc<T>(&label_temp);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, label, -1.0, UnaryOpMode::MUL_A, &label_temp);

  sdaa_ops::doUnaryOpTensor(
      dev_ctx, input, epsilon, UnaryOpMode::ADD_A, &input_temp);
  sdaa_ops::doUnaryOpTensor(dev_ctx, input_temp, 1.0, UnaryOpMode::LOG, out);
  sdaa_ops::doElementMul(dev_ctx, *out, label_temp, -1, out);
  // compute (1-label)*log(1-input+epsilon)
  phi::DenseTensor out_2;
  out_2.Resize(out->dims());
  dev_ctx.template Alloc<T>(&out_2);

  sdaa_ops::doUnaryOpTensor(
      dev_ctx, label_temp, 1.0, UnaryOpMode::ADD_A, &label_temp);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, input, -1.0, UnaryOpMode::MUL_A, &input_temp);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, input_temp, 1.0 + epsilon, UnaryOpMode::ADD_A, &input_temp);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, input_temp, 1.0, UnaryOpMode::LOG, &input_temp);
  sdaa_ops::doElementMul(dev_ctx, label_temp, input_temp, -1, &out_2);
  sdaa_ops::doElementSub(dev_ctx, *out, out_2, -1, out);
}
// dout/dx = -label*1/(input+epsilon)+(1-label)*1/(1-input+epsilon)
template <typename T, typename Context>
void LogLossGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& input,
                       const phi::DenseTensor& label,
                       const phi::DenseTensor& out_grad,
                       float epsilon,
                       phi::DenseTensor* in_grad) {
  VLOG(4) << "Call SDAA LogLossGradKernel";
  dev_ctx.template Alloc<T>(in_grad);
  // compute out=-label*1/(input+epsilon)
  phi::DenseTensor input_temp;
  input_temp.Resize(input.dims());
  dev_ctx.template Alloc<T>(&input_temp);
  phi::DenseTensor label_temp;
  label_temp.Resize(label.dims());
  dev_ctx.template Alloc<T>(&label_temp);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, label, -1.0, UnaryOpMode::MUL_A, &label_temp);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, input, epsilon, UnaryOpMode::ADD_A, &input_temp);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, input_temp, 1.0, UnaryOpMode::RDIV, in_grad);
  sdaa_ops::doElementMul(dev_ctx, *in_grad, label_temp, -1, in_grad);
  // compute (1-label)*1/(1-input+epsilon)
  phi::DenseTensor out_2;
  out_2.Resize(in_grad->dims());
  dev_ctx.template Alloc<T>(&out_2);

  sdaa_ops::doUnaryOpTensor(
      dev_ctx, label_temp, 1.0, UnaryOpMode::ADD_A, &label_temp);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, input, -1.0, UnaryOpMode::MUL_A, &input_temp);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, input_temp, 1.0 + epsilon, UnaryOpMode::ADD_A, &input_temp);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, input_temp, 1.0, UnaryOpMode::RDIV, &input_temp);
  sdaa_ops::doElementMul(dev_ctx, label_temp, input_temp, -1, &out_2);
  sdaa_ops::doElementAdd(dev_ctx, *in_grad, out_2, -1, in_grad);
  sdaa_ops::doElementMul(dev_ctx, *in_grad, out_grad, -1, in_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(log_loss,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LogLossKernel,
                          float,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(log_loss_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LogLossGradKernel,
                          float,
                          phi::dtype::float16) {}

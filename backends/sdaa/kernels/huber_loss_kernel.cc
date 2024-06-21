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
void HuberLossKernel(const Context& dev_ctx,
                     const phi::DenseTensor& input,
                     const phi::DenseTensor& label,
                     float delta,
                     phi::DenseTensor* out,
                     phi::DenseTensor* residual) {
  VLOG(4) << "CALL SDAA HuberLossKernel.";

  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(residual);

  // input.dims() must be same to label.dims(), this check is already in
  // HuberLossInferMeta.
  // Resize input dims to {1, numel}, because tecodnnHuberLossForward only
  // support 2D.
  std::vector<int> input_dims = {1, static_cast<int>(input.numel())};

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnLossReductionMode_t HuberLossReductionMode =
      TECODNN_LOSS_REDUCTION_NONE;

  tecodnnTensorDescriptor_t Desc = sdaa_ops::GetTecodnnTensorDesc(
      input_dims, input.dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnHuberLossForward(tecodnnHandle,
                                        HuberLossReductionMode,
                                        delta,
                                        Desc,
                                        input.data(),
                                        Desc,
                                        label.data(),
                                        Desc,
                                        out->data(),
                                        Desc,
                                        residual->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

template <typename T, typename Context>
void HuberLossGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& residual,
                         const phi::DenseTensor& out_grad,
                         float delta,
                         phi::DenseTensor* input_grad,
                         phi::DenseTensor* label_grad) {
  VLOG(4) << "CALL SDAA HuberLossGradKernel";

  void* input_grad_ptr = nullptr;
  void* label_grad_ptr = nullptr;

  phi::DenseTensor input_grad_temp, label_grad_temp;

  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    input_grad_ptr = input_grad->data();
  } else {
    input_grad_temp.Resize(out_grad.dims());
    dev_ctx.template Alloc<T>(&input_grad_temp);
    input_grad_ptr = input_grad_temp.data();
  }

  if (label_grad) {
    dev_ctx.template Alloc<T>(label_grad);
    label_grad_ptr = label_grad->data();
  } else {
    label_grad_temp.Resize(out_grad.dims());
    dev_ctx.template Alloc<T>(&label_grad_temp);
    label_grad_ptr = label_grad_temp.data();
  }

  // Resize dims to {1, numel}, because tecodnnHuberLossBackward only
  // support 2D.
  std::vector<int> dout_dims = {1, static_cast<int>(out_grad.numel())};

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnLossReductionMode_t HuberLossReductionMode =
      TECODNN_LOSS_REDUCTION_NONE;

  tecodnnTensorDescriptor_t Desc = sdaa_ops::GetTecodnnTensorDesc(
      dout_dims, out_grad.dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnHuberLossBackward(tecodnnHandle,
                                         HuberLossReductionMode,
                                         delta,
                                         Desc,
                                         residual.data(),
                                         Desc,
                                         out_grad.data(),
                                         Desc,
                                         input_grad_ptr,
                                         Desc,
                                         label_grad_ptr));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(huber_loss,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::HuberLossKernel,
                          double,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(huber_loss_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::HuberLossGradKernel,
                          float) {}

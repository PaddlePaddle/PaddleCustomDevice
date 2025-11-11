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
void ElementwisePowRawKernel(const Context& dev_ctx,
                             const phi::DenseTensor& x,
                             const phi::DenseTensor& y,
                             int axis,
                             phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA ElementwisePowRawKernel.";

  dev_ctx.template Alloc<T>(out);

  PADDLE_ENFORCE_EQ(
      x.dims(),
      y.dims(),
      phi::errors::InvalidArgument("The input tensor x and y dims must be same "
                                   "in ElementwisePowRawKernel"));

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(x.dims()), x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t y_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(y.dims()), y.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(out->dims()), out->dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnElementwisePow(tecodnnHandle,
                                      x_Desc,
                                      x.data(),
                                      y_Desc,
                                      y.data(),
                                      out_Desc,
                                      out->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& y,
                          phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA ElementwisePowKernel.";
  int axis = -1;
  custom_kernel::ElementwisePowRawKernel<T, Context>(dev_ctx, x, y, axis, out);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(elementwise_pow,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ElementwisePowKernel,
                          float,
                          phi::dtype::float16) {}

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
void PReluKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& alpha,
                 const std::string& data_format,
                 const std::string& mode,
                 phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA PReluKernel";

  if (1 == alpha.numel()) {
    PADDLE_ENFORCE_LE(x.dims().size(),
                      4,
                      phi::errors::InvalidArgument(
                          "The size of x shape should not be greater than 4,"
                          "but received %d.",
                          x.dims().size()));

    PADDLE_ENFORCE_LE(out->dims().size(),
                      4,
                      phi::errors::InvalidArgument(
                          "The size of out shape should not be greater than 4,"
                          "but received %d.",
                          out->dims().size()));

    PADDLE_ENFORCE_EQ(
        x.dims(),
        out->dims(),
        phi::errors::InvalidArgument("The shape of x and out should be equal,"
                                     "but received x shape %d, out shape %d.",
                                     x.dims(),
                                     out->dims()));

    tecodnnHandle_t tecodnn_handle = custom_kernel::GetHandleFromCTX(dev_ctx);

    dev_ctx.template Alloc<T>(out);

    std::vector<int> x_dims = phi::vectorize<int>(x.dims());
    std::vector<int> out_dims = phi::vectorize<int>(out->dims());
    std::vector<int> alpha_dims = phi::vectorize<int>(alpha.dims());

    tecodnnTensorDescriptor_t x_desc =
        custom_kernel::sdaa_ops::GetTecodnnTensorDesc(
            x_dims, x.dtype(), custom_kernel::TensorFormat::NCHW);
    tecodnnTensorDescriptor_t out_desc =
        custom_kernel::sdaa_ops::GetTecodnnTensorDesc(
            out_dims, out->dtype(), custom_kernel::TensorFormat::NCHW);
    tecodnnTensorDescriptor_t alpha_desc =
        custom_kernel::sdaa_ops::GetTecodnnTensorDesc(
            alpha_dims, alpha.dtype(), custom_kernel::TensorFormat::NCHW);

    TECODNN_CHECK(tecodnnCustomPreluForward(tecodnn_handle,
                                            x_desc,
                                            x.data(),
                                            alpha_desc,
                                            alpha.data(),
                                            out_desc,
                                            out->data()));

    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(alpha_desc));
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "It only supports one weight parameter. Now is %d.", alpha.numel()));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(prelu,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::PReluKernel,
                          float,
                          phi::dtype::float16) {}

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

#include "kernels/funcs/tecodnn_conv_impl.h"

namespace custom_kernel {

template <typename T, typename Context>
void Conv2dTransposeKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& padding,
                           const std::vector<int>& output_padding,
                           const phi::IntArray& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilation,
                           const std::string& data_format,
                           phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA Conv2dTransposeKernel";

  PADDLE_ENFORCE_EQ(
      x.dims().size(),
      4,
      phi::errors::InvalidArgument("tecodnn not support ND tensor"
                                   "But recieved: input dims size is %d",
                                   x.dims().size()));
  PADDLE_ENFORCE_EQ(
      groups,
      1,
      phi::errors::InvalidArgument("tecodnn not support group conv"
                                   "But recieved: group value is %d",
                                   groups));

  custom_kernel::ConvBackwardKernel<T, Context>(dev_ctx,
                                                2,
                                                x,
                                                filter,
                                                x,
                                                strides,
                                                padding,
                                                padding_algorithm,
                                                dilation,
                                                groups,
                                                false,
                                                data_format,
                                                out,
                                                NULL);
}

template <typename T, typename Context>
void Conv2dTransposeGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& x,
                               const phi::DenseTensor& filter,
                               const phi::DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& padding,
                               const std::vector<int>& output_padding,
                               const phi::IntArray& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilation,
                               const std::string& data_format,
                               phi::DenseTensor* dx,
                               phi::DenseTensor* dfilter) {
  VLOG(4) << "CALL SDAA Conv2dTransposeGardKernel";

  if (!dx && !dfilter) return;

  PADDLE_ENFORCE_EQ(
      x.dims().size(),
      4,
      phi::errors::InvalidArgument("tecodnn not support ND tensor"
                                   "But recieved: input dims size is %d",
                                   x.dims().size()));
  PADDLE_ENFORCE_EQ(
      groups,
      1,
      phi::errors::InvalidArgument("tecodnn not support group conv"
                                   "But recieved: group value is %d",
                                   groups));

  if (dfilter) {
    custom_kernel::ConvBackwardKernel<T, Context>(dev_ctx,
                                                  2,
                                                  dout,
                                                  filter,
                                                  x,
                                                  strides,
                                                  padding,
                                                  padding_algorithm,
                                                  dilation,
                                                  groups,
                                                  false,
                                                  data_format,
                                                  NULL,
                                                  dfilter);
  }

  if (dx) {
    custom_kernel::ConvKernel<T, Context>(dev_ctx,
                                          2,
                                          dout,
                                          filter,
                                          strides,
                                          padding,
                                          padding_algorithm,
                                          dilation,
                                          groups,
                                          false,
                                          data_format,
                                          dx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(conv2d_transpose,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTransposeKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_transpose_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTransposeGradKernel,
                          float,
                          phi::dtype::float16) {}

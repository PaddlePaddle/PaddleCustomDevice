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
void Conv2dTecodnnKernel(const Context& dev_ctx,
                         const phi::DenseTensor& input,
                         const phi::DenseTensor& filter,
                         const std::vector<int>& strides_t,
                         const std::vector<int>& paddings_t,
                         const std::string& padding_algorithm,
                         const std::vector<int>& dilations_t,
                         int groups,
                         const std::string& data_format,
                         phi::DenseTensor* output) {
  VLOG(4) << "CALL SDAA Conv2dTecodnnKernel";
  PADDLE_ENFORCE_EQ(
      groups,
      1,
      phi::errors::InvalidArgument("tecodnn not support group conv"
                                   "But recieved: group is %d",
                                   groups));
  ConvKernel<T, Context>(dev_ctx,
                         2,
                         input,
                         filter,
                         strides_t,
                         paddings_t,
                         padding_algorithm,
                         dilations_t,
                         groups,
                         false,
                         data_format,
                         output);
}

template <typename T, typename Context>
void DepthwiseConv2dTecodnnKernel(const Context& dev_ctx,
                                  const phi::DenseTensor& input,
                                  const phi::DenseTensor& filter,
                                  const std::vector<int>& strides_t,
                                  const std::vector<int>& paddings_t,
                                  const std::string& padding_algorithm,
                                  int groups,
                                  const std::vector<int>& dilations_t,
                                  const std::string& data_format,
                                  phi::DenseTensor* output) {
  VLOG(4) << "CALL SDAA DepthwiseConv2dTecodnnKernel";
  const bool is_NHWC = data_format == "NHWC";
  phi::DDim in_dims = input.dims();
  if (is_NHWC) {  // NHWC
    PADDLE_ENFORCE_EQ(groups,
                      in_dims[3],
                      phi::errors::InvalidArgument(
                          "depthwise conv require groups == channels of input, "
                          "But recieved: group is %d",
                          groups));
  } else {  // NCHW
    PADDLE_ENFORCE_EQ(groups,
                      in_dims[1],
                      phi::errors::InvalidArgument(
                          "depthwise conv require groups == channels, "
                          "But recieved: group is %d",
                          groups));
  }
  ConvKernel<T, Context>(dev_ctx,
                         2,
                         input,
                         filter,
                         strides_t,
                         paddings_t,
                         padding_algorithm,
                         dilations_t,
                         groups,
                         true,
                         data_format,
                         output);
}

template <typename T, typename Context>
void Conv2dGradTecodnnKernel(const Context& dev_ctx,
                             const phi::DenseTensor& input,
                             const phi::DenseTensor& filter,
                             const phi::DenseTensor& output_grad,
                             const std::vector<int>& strides_t,
                             const std::vector<int>& paddings_t,
                             const std::string& padding_algorithm,
                             const std::vector<int>& dilations_t,
                             int groups,
                             const std::string& data_format,
                             phi::DenseTensor* input_grad,
                             phi::DenseTensor* filter_grad) {
  VLOG(4) << "CALL SDAA Conv2dGradTecodnnKernel";
  PADDLE_ENFORCE_EQ(
      groups,
      1,
      phi::errors::InvalidArgument("tecodnn not support group conv"
                                   "But recieved: group is %d",
                                   groups));
  ConvBackwardKernel<T, Context>(dev_ctx,
                                 2,
                                 input,
                                 filter,
                                 output_grad,
                                 strides_t,
                                 paddings_t,
                                 padding_algorithm,
                                 dilations_t,
                                 groups,
                                 false,
                                 data_format,
                                 input_grad,
                                 filter_grad);
}

template <typename T, typename Context>
void DepthwiseConv2dGradTecodnnKernel(const Context& dev_ctx,
                                      const phi::DenseTensor& input,
                                      const phi::DenseTensor& filter,
                                      const phi::DenseTensor& output_grad,
                                      const std::vector<int>& strides_t,
                                      const std::vector<int>& paddings_t,
                                      const std::string& padding_algorithm,
                                      int groups,
                                      const std::vector<int>& dilations_t,
                                      const std::string& data_format,
                                      phi::DenseTensor* input_grad,
                                      phi::DenseTensor* filter_grad) {
  VLOG(4) << "CALL SDAA DepthwiseConv2dGradTecodnnKernel";
  const bool is_NHWC = data_format == "NHWC";
  phi::DDim in_dims = input.dims();
  ConvBackwardKernel<T, Context>(dev_ctx,
                                 2,
                                 input,
                                 filter,
                                 output_grad,
                                 strides_t,
                                 paddings_t,
                                 padding_algorithm,
                                 dilations_t,
                                 groups,
                                 true,
                                 data_format,
                                 input_grad,
                                 filter_grad);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(conv2d,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTecodnnKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dTecodnnKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dGradTecodnnKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dGradTecodnnKernel,
                          float,
                          phi::dtype::float16) {}

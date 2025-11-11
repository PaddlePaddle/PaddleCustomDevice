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
#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void PNormKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 float porder,
                 int axis,
                 float epsilon,
                 bool keepdim,
                 bool asvector,
                 phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA PNormKernel";

  PADDLE_ENFORCE_LT(x.dims().size(),
                    5,
                    phi::errors::InvalidArgument(
                        "tecodnnPNormForward only support input dims size "
                        "less than 5, "
                        "but received %d. ",
                        x.dims().size()));

  dev_ctx.template Alloc<T>(out);
  std::vector<int> axis_dims = {axis};
  std::vector<int> reduce_axis;
  sdaa_ops::GetReduceDimReduceAll(
      axis_dims, x.dims().size(), asvector, &reduce_axis);

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());

  for (int i = 0; i < x_dims.size(); i++) {
    PADDLE_ENFORCE_GT(x_dims[i],
                      0,
                      phi::errors::InvalidArgument(
                          "The dims of Input(X) should be greater than 0."));
  }
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::Undefined);
  TECODNN_CHECK(tecodnnPNormForward(tecodnnHandle,
                                    porder,
                                    reduce_axis.data(),
                                    reduce_axis.size(),
                                    x_Desc,
                                    x.data(),
                                    out_Desc,
                                    out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

template <typename T, typename Context>
void PNormGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     const phi::DenseTensor& dy,
                     float porder,
                     int axis,
                     float epsilon,
                     bool keepdim UNUSED,
                     bool asvector,
                     phi::DenseTensor* dx) {
  VLOG(4) << "CALL SDAA PNormGradKernel";
  dev_ctx.template Alloc<T>(dx);

  PADDLE_ENFORCE_LE(x.dims().size(),
                    4,
                    phi::errors::InvalidArgument(
                        "tecodnnPNormForward only support input dims size "
                        "less equal 4, "
                        "but received %d. ",
                        x.dims().size()));

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> y_dims = phi::vectorize<int>(y.dims());
  std::vector<int> dy_dims = phi::vectorize<int>(dy.dims());
  std::vector<int> dx_dims = phi::vectorize<int>(dx->dims());

  std::vector<int> axis_dims = {axis};
  std::vector<int> reduce_axis;
  sdaa_ops::GetReduceDimReduceAll(
      axis_dims, x_dims.size(), asvector, &reduce_axis);

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t y_Desc = sdaa_ops::GetTecodnnTensorDesc(
      y_dims, y.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t dy_Desc = sdaa_ops::GetTecodnnTensorDesc(
      dy_dims, dy.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t dx_Desc = sdaa_ops::GetTecodnnTensorDesc(
      dx_dims, dx->dtype(), TensorFormat::Undefined);

  TECODNN_CHECK(tecodnnPNormBackward(tecodnnHandle,
                                     porder,
                                     reduce_axis.size(),
                                     reduce_axis.data(),
                                     epsilon,
                                     x_Desc,
                                     x.data(),
                                     y_Desc,
                                     y.data(),
                                     dy_Desc,
                                     dy.data(),
                                     dx_Desc,
                                     dx->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dy_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dx_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    p_norm, sdaa, ALL_LAYOUT, custom_kernel::PNormKernel, double, float) {}

PD_REGISTER_PLUGIN_KERNEL(p_norm_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::PNormGradKernel,
                          double,
                          float) {}

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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void GridSampleKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& grid,
                      const std::string& mode,
                      const std::string& padding_mode,
                      bool align_corners,
                      phi::DenseTensor* out) {
  using PaddingMode = tecodnnGridSamplePaddingMode_t;
  using Mode = tecodnnGridSampleInterpolationMode_t;
  PaddingMode enum_padding_mode;
  Mode enum_mode;

  PADDLE_ENFORCE_EQ(
      align_corners,
      false,
      phi::errors::InvalidArgument("GridSampleKernel's padding mode only "
                                   "support align_corners == false."));

  PADDLE_ENFORCE_EQ(padding_mode,
                    "zeros",
                    phi::errors::InvalidArgument(
                        "GridSampleKernel's padding mode only support zero."));
  PADDLE_ENFORCE_EQ(mode,
                    "bilinear",
                    phi::errors::InvalidArgument(
                        "GridSampleKernel's mode only support bilinear."));

  PADDLE_ENFORCE_EQ(
      x.dims().size(),
      4,
      phi::errors::InvalidArgument("GridSampleKernel's mode only support 2d."));

  enum_padding_mode = PaddingMode::TECODNN_GRID_SAMPLE_ZEROS;
  enum_mode = Mode::TECODNN_GRID_SAMPLE_BILINEAR;

  const int n = grid.dims()[0];
  const int out_h = grid.dims()[1];
  const int out_w = grid.dims()[2];
  const int c = x.dims()[1];
  const int in_h = x.dims()[2];
  const int in_w = x.dims()[3];
  VLOG(3) << "n: " << n << "; c: " << c << "; out_h: " << out_h
          << "; out_w: " << out_w;

  auto* output_data = dev_ctx.template Alloc<T>(out);
  VLOG(3) << "out dims: " << out->dims()[0] << "; " << out->dims()[1] << "; "
          << out->dims()[2] << "; " << out->dims()[3];

  auto x_temp = phi::DenseTensor();
  auto x_temp_dims = std::vector<int>{n, in_h, in_w, c};
  auto tensor_meta =
      phi::DenseTensorMeta{x.dtype(), phi::make_ddim(x_temp_dims)};
  x_temp.set_meta(tensor_meta);
  dev_ctx.template Alloc<T>(&x_temp);
  sdaa_ops::doTransformTensor(dev_ctx, x, Convert_TF::NCHW2NHWC, &x_temp);

  auto out_temp = phi::DenseTensor{};
  auto out_meta = phi::DenseTensorMeta(x.dtype(), {n, out_h, out_w, c});
  out_temp.set_meta(out_meta);
  dev_ctx.template Alloc<T>(&out_temp);

  tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_desc = sdaa_ops::GetTecodnnTensorDesc(
      x_temp_dims, x.dtype(), TensorFormat::NHWC);

  tecodnnTensorDescriptor_t grid_desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(grid.dims()), grid.dtype(), TensorFormat::NHWC);

  tecodnnTensorDescriptor_t out_desc = sdaa_ops::GetTecodnnTensorDesc(
      {n, out_h, out_w, c}, out->dtype(), TensorFormat::NHWC);

  TECODNN_CHECK(tecodnnGridSampleForward(handle,
                                         align_corners,
                                         enum_mode,
                                         enum_padding_mode,
                                         x_desc,
                                         x_temp.data(),
                                         grid_desc,
                                         grid.data(),
                                         out_desc,
                                         out_temp.data()));

  dev_ctx.template Alloc<T>(out);
  sdaa_ops::doTransformTensor(dev_ctx, out_temp, Convert_TF::NHWC2NCHW, out);

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(grid_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_desc));
}

template <typename T, typename Context>
void GridSampleGradKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& grid,
                          const phi::DenseTensor& out_grad,
                          const std::string& mode,
                          const std::string& padding_mode,
                          bool align_corners,
                          phi::DenseTensor* x_grad,
                          phi::DenseTensor* grid_grad) {
  using PaddingMode = tecodnnGridSamplePaddingMode_t;
  using Mode = tecodnnGridSampleInterpolationMode_t;
  PaddingMode enum_padding_mode;
  Mode enum_mode;

  PADDLE_ENFORCE_EQ(
      align_corners,
      false,
      phi::errors::InvalidArgument("GridSampleKernel's padding mode only "
                                   "support align_corners == false."));

  PADDLE_ENFORCE_EQ(padding_mode,
                    "zeros",
                    phi::errors::InvalidArgument(
                        "GridSampleKernel's padding mode only support zero."));
  PADDLE_ENFORCE_EQ(mode,
                    "bilinear",
                    phi::errors::InvalidArgument(
                        "GridSampleKernel's mode only support bilinear."));
  PADDLE_ENFORCE_EQ(
      x.dims().size(),
      4,
      phi::errors::InvalidArgument("GridSampleKernel's mode only support 2d."));

  PADDLE_ENFORCE_EQ(grid_grad != nullptr,
                    true,
                    phi::errors::InvalidArgument(
                        "GridSampleKernel's not support grid_grad == null."));

  enum_padding_mode = PaddingMode::TECODNN_GRID_SAMPLE_ZEROS;
  enum_mode = Mode::TECODNN_GRID_SAMPLE_BILINEAR;

  const int n = grid.dims()[0];
  const int out_h = grid.dims()[1];
  const int out_w = grid.dims()[2];
  const int c = x.dims()[1];
  const int in_h = x.dims()[2];
  const int in_w = x.dims()[3];

  dev_ctx.template Alloc<T>(x_grad);
  sdaa_ops::doFillTensor<T>(
      dev_ctx, static_cast<T>(0), phi::CppTypeToDataType<T>::Type(), x_grad);

#define NCHW_TRANFORM_NHWC(tensor)                                            \
  auto tensor##_temp = phi::DenseTensor();                                    \
  auto tensor##_temp_dims =                                                   \
      sdaa_ops::doDimPermute(tensor, Convert_TF::NCHW2NHWC);                  \
  auto tensor##_meta =                                                        \
      phi::DenseTensorMeta{tensor.dtype(), tensor##_temp_dims};               \
  tensor##_temp.set_meta(tensor##_meta);                                      \
  dev_ctx.template Alloc<T>(&tensor##_temp);                                  \
  sdaa_ops::doTransformTensor(                                                \
      dev_ctx, tensor, Convert_TF::NCHW2NHWC, &tensor##_temp);                \
  tecodnnTensorDescriptor_t tensor##_desc =                                   \
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(tensor##_temp_dims), \
                                     tensor.dtype(),                          \
                                     TensorFormat::NHWC)

  NCHW_TRANFORM_NHWC(x);
  NCHW_TRANFORM_NHWC(out_grad);

  auto x_grad_temp = phi::DenseTensor{};
  auto x_grad_meta = phi::DenseTensorMeta{x.dtype(), {n, in_h, in_w, c}};
  x_grad_temp.set_meta(x_grad_meta);
  dev_ctx.template Alloc<T>(&x_grad_temp);

  T* grid_grad_data = dev_ctx.template Alloc<T>(grid_grad);

  tecodnnHandle_t handle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t grid_desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(grid.dims()), grid.dtype(), TensorFormat::NHWC);

  tecodnnTensorDescriptor_t x_grad_desc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(x_grad_temp.dims()),
                                     x_grad->dtype(),
                                     TensorFormat::NHWC);

  tecodnnTensorDescriptor_t grid_grad_desc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(grid_grad->dims()),
                                     grid_grad->dtype(),
                                     TensorFormat::NHWC);

  TECODNN_CHECK(tecodnnGridSampleBackward(handle,
                                          enum_mode,
                                          enum_padding_mode,
                                          align_corners,
                                          x_desc,
                                          x_temp.data(),
                                          grid_desc,
                                          grid.data(),
                                          out_grad_desc,
                                          out_grad_temp.data(),
                                          x_grad_desc,
                                          x_grad_temp.data(),
                                          grid_grad_desc,
                                          grid_grad_data));

  dev_ctx.template Alloc<T>(x_grad);
  sdaa_ops::doTransformTensor(
      dev_ctx, x_grad_temp, Convert_TF::NHWC2NCHW, x_grad);

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(grid_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_grad_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(grid_grad_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
#undef NCHW_TRANFORM_NHWC
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    grid_sample, sdaa, ALL_LAYOUT, custom_kernel::GridSampleKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(grid_sample_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::GridSampleGradKernel,
                          float) {}

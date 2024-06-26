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

#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/extension.h"  // NOLINT

#define CHECK_CUSTOM_INPUT(x) \
  PD_CHECK(x.is_custom_device(), #x " must be a custom Tensor.")

static void GetRowsCols(const std::vector<int64_t> &shape,
                        int *p_rows,
                        int *p_cols) {
  int rows = 1;
  for (int i = 0; i + 1 < shape.size(); ++i) {
    rows *= shape[i];
  }
  int cols = shape[shape.size() - 1];
  *p_rows = rows;
  *p_cols = cols;
}

std::vector<paddle::DataType> RMSLnFwdInferDtype(
    const paddle::DataType &x_dtype, const paddle::DataType &scale_dtype) {
  return {x_dtype, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> RMSLnFwdInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    float epsilon) {
  int rows, cols;
  GetRowsCols(x_shape, &rows, &cols);
  return {x_shape, {rows}};
}

std::vector<paddle::Tensor> RMSLnFwd(const paddle::Tensor &x,
                                     const paddle::Tensor &scale,
                                     float epsilon) {
  const auto &scale_shape = scale.shape();
  const auto &x_shape = x.shape();
  PD_CHECK(scale_shape.size() == 1);
  PD_CHECK(scale_shape[0] == x_shape[x_shape.size() - 1]);
  CHECK_CUSTOM_INPUT(x);
  CHECK_CUSTOM_INPUT(scale);

  // check dtype
  PADDLE_ENFORCE_EQ(
      x.dtype() == phi::DataType::FLOAT32,
      true,
      phi::errors::InvalidArgument("x dtype expect [%s], but got [%s]",
                                   phi::DataType::FLOAT32,
                                   x.dtype()));
  PADDLE_ENFORCE_EQ(
      scale.dtype() == phi::DataType::FLOAT32,
      true,
      phi::errors::InvalidArgument("scale dtype expect [%s], but got [%s]",
                                   phi::DataType::FLOAT32,
                                   scale.dtype()));
  // check shape
  PADDLE_ENFORCE_EQ(
      x.dims().size() <= 4,
      true,
      phi::errors::InvalidArgument(
          "x dims must less than or equal 4 on sdaa, but got [%d]",
          x.dims().size()));

  PADDLE_ENFORCE_EQ(
      scale.dims().size() <= 4,
      true,
      phi::errors::InvalidArgument(
          "scale dims must less than or equal 4 on sdaa, but got [%d]",
          scale.dims().size()));

  int rows, cols;
  GetRowsCols(x_shape, &rows, &cols);

  auto x_dims = phi::vectorize<int>(x.dims());
  auto scale_dims = phi::vectorize<int>(scale.dims());
  std::vector<int> invvar_dims(4, 0);

  // pad x/scale/invvar to 4D
  for (int i = x_dims.size(); i < 4; ++i) {
    x_dims.insert(x_dims.begin(), 1);
  }

  for (int i = scale_dims.size(); i < 4; ++i) {
    scale_dims.insert(scale_dims.begin(), 1);
  }

  for (int i = 0; i < x_dims.size() - 1; ++i) {
    invvar_dims[i] = x_dims[i];
  }
  invvar_dims[3] = 1;

  auto place = x.place();
  auto y = paddle::empty(x_shape, scale.type(), place);
  auto invvar = paddle::empty({rows}, paddle::DataType::FLOAT32, place);

  auto dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(x.place());
  auto custom_ctx = static_cast<const phi::CustomContext *>(dev_ctx);
  auto handle = custom_kernel::GetHandleFromCTX(*custom_ctx);

  auto x_desc =
      custom_kernel::sdaa_ops::GetTecodnnTensorDesc(x_dims, x.dtype());
  auto scale_desc =
      custom_kernel::sdaa_ops::GetTecodnnTensorDesc(scale_dims, scale.dtype());
  auto invvar_desc = custom_kernel::sdaa_ops::GetTecodnnTensorDesc(
      invvar_dims, invvar.dtype());

  TECODNN_CHECK(tecodnnRMSNormForward(handle,
                                      static_cast<double>(epsilon),
                                      x_desc,
                                      x.data(),
                                      scale_desc,
                                      scale.data(),
                                      x_desc,
                                      y.data(),
                                      invvar_desc,
                                      invvar.data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(scale_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(invvar_desc));

  return {y, invvar};
}

std::vector<paddle::Tensor> RMSLnBwd(const paddle::Tensor &x,
                                     const paddle::Tensor &scale,
                                     const paddle::Tensor &invvar,
                                     const paddle::Tensor &dy,
                                     float epsilon) {
  CHECK_CUSTOM_INPUT(dy);
  CHECK_CUSTOM_INPUT(x);
  CHECK_CUSTOM_INPUT(invvar);
  CHECK_CUSTOM_INPUT(scale);

  // check dtype
  PADDLE_ENFORCE_EQ(
      x.dtype() == phi::DataType::FLOAT32,
      true,
      phi::errors::InvalidArgument("x dtype expect [%s], but got [%s]",
                                   phi::DataType::FLOAT32,
                                   x.dtype()));
  PADDLE_ENFORCE_EQ(
      scale.dtype() == phi::DataType::FLOAT32,
      true,
      phi::errors::InvalidArgument("scale dtype expect [%s], but got [%s]",
                                   phi::DataType::FLOAT32,
                                   scale.dtype()));

  PADDLE_ENFORCE_EQ(
      dy.dtype() == phi::DataType::FLOAT32,
      true,
      phi::errors::InvalidArgument("dy dtype expect [%s], but got [%s]",
                                   phi::DataType::FLOAT32,
                                   dy.dtype()));
  // check shape
  PADDLE_ENFORCE_EQ(
      x.dims().size() <= 4,
      true,
      phi::errors::InvalidArgument(
          "x dims must less than or equal 4 on sdaa, but got [%d]",
          x.dims().size()));

  PADDLE_ENFORCE_EQ(
      scale.dims().size() <= 4,
      true,
      phi::errors::InvalidArgument(
          "scale dims must less than or equal 4 on sdaa, but got [%d]",
          scale.dims().size()));

  int rows, cols;
  GetRowsCols(x.shape(), &rows, &cols);

  auto grad_x = paddle::empty_like(x);
  auto grad_scale = paddle::empty_like(scale);

  auto place = x.place();
  auto dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(x.place());
  auto custom_ctx = static_cast<const phi::CustomContext *>(dev_ctx);
  auto handle = custom_kernel::GetHandleFromCTX(*custom_ctx);

  auto x_dims = phi::vectorize<int>(x.dims());
  auto scale_dims = phi::vectorize<int>(scale.dims());
  std::vector<int> invvar_dims(4, 0);

  // pad x/scale/invvar to 4D
  for (int i = x_dims.size(); i < 4; ++i) {
    x_dims.insert(x_dims.begin(), 1);
  }

  for (int i = scale_dims.size(); i < 4; ++i) {
    scale_dims.insert(scale_dims.begin(), 1);
  }

  for (int i = 0; i < x_dims.size() - 1; ++i) {
    invvar_dims[i] = x_dims[i];
  }
  invvar_dims[3] = 1;

  auto x_desc =
      custom_kernel::sdaa_ops::GetTecodnnTensorDesc(x_dims, x.dtype());
  auto scale_desc =
      custom_kernel::sdaa_ops::GetTecodnnTensorDesc(scale_dims, scale.dtype());
  auto invvar_desc = custom_kernel::sdaa_ops::GetTecodnnTensorDesc(
      invvar_dims, invvar.dtype());

  TECODNN_CHECK(tecodnnRMSNormBackward(handle,
                                       x_desc,
                                       dy.data(),
                                       x_desc,
                                       x.data(),
                                       scale_desc,
                                       scale.data(),
                                       invvar_desc,
                                       const_cast<void *>(invvar.data()),
                                       x_desc,
                                       grad_x.data(),
                                       scale_desc,
                                       grad_scale.data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(scale_desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(invvar_desc));

  return {grad_x, grad_scale};
}

PD_BUILD_OP(custom_fused_rms_norm)
    .Inputs({"x", "scale"})
    .Attrs({"epsilion: float"})
    .Outputs({"y", "invvar"})
    .SetKernelFn(PD_KERNEL(RMSLnFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(RMSLnFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RMSLnFwdInferDtype));

PD_BUILD_GRAD_OP(custom_fused_rms_norm)
    .Inputs({"x", "scale", "invvar", paddle::Grad("y")})
    .Outputs({paddle::Grad("x"), paddle::Grad("scale")})
    .Attrs({"epsilion: float"})
    .SetKernelFn(PD_KERNEL(RMSLnBwd));

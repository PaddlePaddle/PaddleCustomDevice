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

#include <algorithm>
#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/all_context.h"
#include "sdcops.h"  // NOLINT

void CHECK_CUSTOM_INPUT(const paddle::Tensor& x) {
  PD_CHECK(x.is_custom_device(), "x must be a custom Tensor.");
  PD_CHECK(x.dtype() == phi::DataType::FLOAT32, "x must be float dtype.");
  PD_CHECK(x.dims().size() == 3, "x must be a 3-D tensor.");
}

std::vector<std::vector<int64_t>> SwigluInferShape(
    const std::vector<int64_t>& x_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> SwigluInferDtype(
    const paddle::DataType& x_dtype) {
  return {x_dtype};
}

std::vector<paddle::Tensor> SwigluForward(const paddle::Tensor& x) {
  CHECK_CUSTOM_INPUT(x);
  VLOG(4) << "Call SwigluForward";
  auto x_dims = x.dims();
  int64_t seq_len = x_dims[0];
  int64_t bs = x_dims[1];
  int64_t hidden_size = x_dims[2];
  int64_t output_hidden_size = hidden_size / 2;
  auto x_out = paddle::empty(
      {seq_len, bs, output_hidden_size}, phi::DataType::FLOAT32, x.place());

  auto dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(x.place());
  auto custom_ctx = static_cast<const phi::CustomContext*>(dev_ctx);
  tecocustomHandle_t tecocustomHandle =
      custom_kernel::GetTecoCustomHandleFromCTX(*custom_ctx);

  tecocustomTensorDescriptor_t x_Desc =
      custom_kernel::sdaa_ops::GetTecocustomTensorDesc(
          phi::vectorize<int>(x.dims()),
          x.dtype(),
          custom_kernel::TensorFormat::NCHW);

  tecocustomTensorDescriptor_t dstDesc =
      custom_kernel::sdaa_ops::GetTecocustomTensorDesc(
          phi::vectorize<int>(x_out.dims()),
          x_out.dtype(),
          custom_kernel::TensorFormat::NCHW);

  TECOCUSTOM_CHECK(tecocustomSwigluActivateForward(
      tecocustomHandle, x_Desc, x.data(), dstDesc, x_out.data()));

  TECOCUSTOM_CHECK(tecocustomDestroyTensorDescriptor(x_Desc));
  TECOCUSTOM_CHECK(tecocustomDestroyTensorDescriptor(dstDesc));
  return {x_out};
}

std::vector<paddle::Tensor> SwigluBackward(const paddle::Tensor& x,
                                           const paddle::Tensor& grad_out) {
  CHECK_CUSTOM_INPUT(x);
  CHECK_CUSTOM_INPUT(grad_out);
  VLOG(4) << "Call SwigluBackward";
  auto x_dims = x.dims();
  int64_t seq_len = x_dims[0];
  int64_t bs = x_dims[1];
  int64_t hidden_size = x_dims[2];
  int64_t output_hidden_size = hidden_size / 2;

  auto dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(x.place());
  auto custom_ctx = static_cast<const phi::CustomContext*>(dev_ctx);

  auto dx = paddle::empty(
      {seq_len, bs, hidden_size}, phi::DataType::FLOAT32, x.place());
  tecocustomHandle_t tecocustomHandle =
      custom_kernel::GetTecoCustomHandleFromCTX(*custom_ctx);

  tecocustomTensorDescriptor_t srcDesc =
      custom_kernel::sdaa_ops::GetTecocustomTensorDesc(
          phi::vectorize<int>(x.dims()),
          x.dtype(),
          custom_kernel::TensorFormat::NCHW);

  tecocustomTensorDescriptor_t gradDstDesc =
      custom_kernel::sdaa_ops::GetTecocustomTensorDesc(
          phi::vectorize<int>(grad_out.dims()),
          grad_out.dtype(),
          custom_kernel::TensorFormat::NCHW);

  TECOCUSTOM_CHECK(tecocustomSwigluActivateBackward(tecocustomHandle,
                                                    gradDstDesc,
                                                    grad_out.data(),
                                                    srcDesc,
                                                    x.data(),
                                                    srcDesc,
                                                    dx.data()));
  TECOCUSTOM_CHECK(tecocustomDestroyTensorDescriptor(srcDesc));
  TECOCUSTOM_CHECK(tecocustomDestroyTensorDescriptor(gradDstDesc));
  return {dx};
}

PD_BUILD_OP(custom_swiglu)
    .Inputs({"x"})
    .Outputs({"x_out"})
    .SetKernelFn(PD_KERNEL(SwigluForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SwigluInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SwigluInferDtype));

PD_BUILD_GRAD_OP(custom_swiglu)
    .Inputs({"x", paddle::Grad("x_out")})
    .Outputs({paddle::Grad("x")})
    .SetKernelFn(PD_KERNEL(SwigluBackward));

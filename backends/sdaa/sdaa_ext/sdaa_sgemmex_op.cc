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
#include "kernels/funcs/tblas_baseop.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/all_context.h"
#include "sdcops.h"           // NOLINT
#include "tecoblas_custom.h"  // NOLINT

#define CHECK_CUSTOM_INPUT(x) \
  PD_CHECK(x.is_custom_device(), #x " must be a custom Tensor.")

std::vector<std::vector<int64_t>> SGemmEXInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& y_shape,
    bool use_AP,
    bool use_BP,
    bool use_CP) {
  std::vector<int64_t> x_dims = x_shape;
  std::vector<int64_t> y_dims = y_shape;

  std::vector<int64_t> out_shape = {x_dims[0], x_dims[1], y_dims[1]};
  return {out_shape};
}

std::vector<paddle::DataType> GemmPvcInferDtype(
    const paddle::DataType& x_dtype, const paddle::DataType& y_dtype) {
  return {x_dtype};
}

paddle::Tensor sgemmex(const paddle::Tensor& x,
                       const paddle::Tensor& y,
                       tblasOperation_t trans_x,
                       tblasOperation_t trans_y) {
  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();

  int x_row = x_dims[0];
  int x_col = x_dims[1];
  int y_row = y_dims[0];
  int y_col = y_dims[1];
  int m = x_row;
  int k = x_col;
  int n = y_col;
  float alpha = 1.0;
  float beta = 0.0;
  int lda = k;
  int ldb = n;
  int ldc = n;

  if (trans_x == TBLAS_OP_T) {
    m = x_col;
    k = x_row;
  }
  if (trans_y == TBLAS_OP_T) {
    n = y_row;
    ldc = n;
  }

  std::vector<int64_t> out_shape = {m, n};
  auto out_type = phi::DataType::FLOAT32;
  auto output = paddle::empty(out_shape, out_type, x.place());

  auto dev_ctx =
      paddle::experimental::DeviceContextPool::Instance().Get(x.place());
  auto custom_ctx = static_cast<const phi::CustomContext*>(dev_ctx);

  custom_kernel::tblas_ops::TecoBlas<float>::Gemm(*custom_ctx,
                                                  trans_x,
                                                  trans_y,
                                                  m,
                                                  n,
                                                  k,
                                                  alpha,
                                                  x.data<phi::dtype::float16>(),
                                                  lda,
                                                  y.data<phi::dtype::float16>(),
                                                  ldb,
                                                  beta,
                                                  output.data<float>(),
                                                  ldc);
  return output;
}

std::vector<paddle::Tensor> SGemmEXForward(const paddle::Tensor& x,
                                           const paddle::Tensor& y) {
  CHECK_CUSTOM_INPUT(x);
  CHECK_CUSTOM_INPUT(y);

  std::vector<int64_t> x_dims = phi::vectorize(x.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y.dims());
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();

  // check dims
  PADDLE_ENFORCE_EQ(x_ndim == 2 || x_ndim == 3,
                    true,
                    phi::errors::InvalidArgument("dim of x should be 2 or 3."));
  PADDLE_ENFORCE_EQ(
      y_ndim == 2, true, phi::errors::InvalidArgument("dim of y should be 2."));

  paddle::Tensor x_tmp(x);
  if (x_ndim == 3) {
    x_tmp.reshape({x_dims[0] * x_dims[1], x_dims[2]});
  }

  paddle::Tensor x_half, y_half;
  if (x.dtype() != phi::DataType::FLOAT16) {
    x_half = paddle::empty_like(x, phi::DataType::FLOAT16, x.place());
    x_half = x.cast(phi::DataType::FLOAT16);
  } else {
    x_half = x;
  }
  if (y.dtype() != phi::DataType::FLOAT16) {
    y_half = paddle::empty_like(y, phi::DataType::FLOAT16, y.place());
    y_half = y.cast(phi::DataType::FLOAT16);
  } else {
    y_half = y;
  }
  auto ret = sgemmex(x_half, y_half, TBLAS_OP_N, TBLAS_OP_N);

  if (x_ndim == 3) {
    ret.reshape({x_dims[0], x_dims[1], y_dims[1]});
    x_tmp.reshape(x_dims);
    x_half.reshape(x_dims);
  }

  return {ret, x_half, y_half};
}

std::vector<paddle::Tensor> SGemmEXBackward(const paddle::Tensor& grad_output,
                                            const paddle::Tensor& x_store,
                                            const paddle::Tensor& y_store) {
  CHECK_CUSTOM_INPUT(x_store);
  CHECK_CUSTOM_INPUT(y_store);
  CHECK_CUSTOM_INPUT(grad_output);
  std::vector<int64_t> output_dims = phi::vectorize(grad_output.dims());
  std::vector<int64_t> x_dims = phi::vectorize(x_store.dims());
  std::vector<int64_t> y_dims = phi::vectorize(y_store.dims());
  int output_ndim = output_dims.size();
  int x_ndim = x_dims.size();
  int y_ndim = y_dims.size();

  // check dtype and dims
  PADDLE_ENFORCE_EQ(x_store.dtype() == phi::DataType::FLOAT16,
                    true,
                    phi::errors::InvalidArgument(
                        "custom_matmul backword dtype should be half!!"));
  PADDLE_ENFORCE_EQ(y_store.dtype() == phi::DataType::FLOAT16,
                    true,
                    phi::errors::InvalidArgument(
                        "custom_matmul backword dtype should be half!!"));
  PADDLE_ENFORCE_EQ(grad_output.dtype() == phi::DataType::FLOAT32,
                    true,
                    phi::errors::InvalidArgument(
                        "custom_matmul output dtype should be float!!"));
  PADDLE_ENFORCE_EQ(x_ndim == 2 || x_ndim == 3,
                    true,
                    phi::errors::InvalidArgument("dim of x should be 2 or 3."));
  PADDLE_ENFORCE_EQ(
      output_ndim == 2 || output_ndim == 3,
      true,
      phi::errors::InvalidArgument("dim of output should be 2 or 3."));
  PADDLE_ENFORCE_EQ(
      y_ndim == 2, true, phi::errors::InvalidArgument("dim of y should be 2."));

  paddle::Tensor x_tmp(x_store);
  paddle::Tensor dout_half = grad_output.cast(phi::DataType::FLOAT16);
  if (output_ndim == 3) {
    dout_half.reshape({output_dims[0] * output_dims[1], output_dims[2]});
    x_tmp.reshape({x_dims[0] * x_dims[1], x_dims[2]});
  }

  auto dx = sgemmex(dout_half, y_store, TBLAS_OP_N, TBLAS_OP_T);
  auto dy = sgemmex(x_store, dout_half, TBLAS_OP_T, TBLAS_OP_N);

  if (x_ndim == 3) dx.reshape(x_dims);

  return {dx, dy};
}

PD_BUILD_OP(custom_sgemmex)
    .Inputs({"X", "Y"})
    .Outputs({"Out", "X_half", "Y_half"})
    .SetKernelFn(PD_KERNEL(SGemmEXForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SGemmEXInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GemmPvcInferDtype));

PD_BUILD_GRAD_OP(custom_sgemmex)
    .Inputs({paddle::Grad("Out"), "X_half", "Y_half"})
    .Outputs({paddle::Grad("X"), paddle::Grad("Y")})
    .SetKernelFn(PD_KERNEL(SGemmEXBackward));

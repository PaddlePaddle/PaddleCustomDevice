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

#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/all_context.h"
#include "sdcops.h"  // NOLINT

inline void FCOutputSize(const phi::DDim& in_dims,
                         const phi::DDim& w_dims,
                         std::vector<int64_t>& out_dims,  // NOLINT
                         int in_num_col_dims,
                         bool padding_weights) {
  auto in_mat_dims = phi::flatten_to_2d(in_dims, in_num_col_dims);
  auto w_dims0 = padding_weights ? w_dims[0] - 4 : w_dims[0];
  auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];
  PADDLE_ENFORCE_EQ(
      in_mat_dims[1],
      w_dims0,
      phi::errors::InvalidArgument(
          "The input's second dimension and weight's first dimension is "
          "expected to be the same. But received input's second dimension is "
          "%d, input's shape is %s; weight's first dimension is %d, weight's "
          "shape is %s.",
          in_mat_dims[1],
          in_mat_dims,
          w_dims0,
          phi::make_ddim({w_dims0, w_dims1})));

  out_dims.reserve(static_cast<size_t>(in_num_col_dims + 1));
  for (int i = 0; i < in_num_col_dims; ++i) {
    out_dims.push_back(in_dims[i]);
  }
  out_dims.push_back(w_dims1);
}

std::vector<paddle::Tensor> CustomFcOp(const paddle::Tensor& input,
                                       const paddle::Tensor& w,
                                       const paddle::Tensor& bias,
                                       int in_num_col_dims,
                                       const std::string& activation_type,
                                       bool padding_weights) {
  bool with_relu = activation_type == "relu" ? true : false;

  PADDLE_ENFORCE_EQ(with_relu, false, "not suport with relu now.");
  PADDLE_ENFORCE_EQ(padding_weights,
                    false,
                    phi::errors::PermissionDenied(
                        "Weight padding in fc can not be used in sdaa scope."));

  paddle::Tensor out = paddle::experimental::matmul(input, w);

  out = paddle::experimental::add(out, bias);
  return {out};
}

std::vector<paddle::DataType> CustomFcInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& y_dtype,
    const paddle::DataType& z_dtype) {
  return {x_dtype};
}

std::vector<std::vector<int64_t>> CustomFcInferShape(
    const std::vector<int64_t>& input,
    const std::vector<int64_t>& w,
    const std::vector<int64_t>& bias,
    int in_num_col_dims,
    const std::string& activation_type,
    bool padding_weights) {
  PADDLE_ENFORCE_EQ(
      w.size(),
      2,
      phi::errors::InvalidArgument(
          "The input Weight of fc is expected to be a 2-D tensor. "
          "But received the number of Weight's dimensions is %d, "
          "Weight's shape is %s.",
          w.size(),
          phi::make_ddim(w)));

  auto w_dims1 = padding_weights ? w[1] - 4 : w[1];

  PADDLE_ENFORCE_LE(
      bias.size(),
      2,
      phi::errors::InvalidArgument(
          "The input Bias of fc is expected to be a 1-D or 2-D tensor. But "
          "received the number of Bias's dimensions is %d, "
          "Bias's shape is %s.",
          bias.size(),
          phi::make_ddim(bias)));

  PADDLE_ENFORCE_EQ(
      bias[bias.size() - 1],
      w_dims1,
      phi::errors::InvalidArgument(
          "The last dimension of input Bias is expected be equal "
          "to the actual width of input Weight. But received the last "
          "dimension of Bias is %d, Bias's shape is %s; "
          "the actual width of Weight is %d, Weight's shape is %s.",
          bias[bias.size() - 1],
          phi::make_ddim(bias),
          w_dims1,
          phi::make_ddim(w)));

  if (bias.size() == 2) {
    PADDLE_ENFORCE_EQ(
        bias[0],
        1,
        phi::errors::InvalidArgument(
            "The first dimension of input Bias is expected to be 1, "
            "but received %d, Bias's shape is %s.",
            bias[0],
            phi::make_ddim(bias)));
  }

  PADDLE_ENFORCE_LT(
      in_num_col_dims,
      input.size(),
      phi::errors::InvalidArgument(
          "The attribute in_num_col_dims used to flatten Input to "
          "a 2-D tensor, is expected to be less than the number of "
          "Input's dimensions. But received in_num_col_dims is %d, "
          "the number of Input's dimensions is %d, Input's shape is %s.",
          in_num_col_dims,
          input.size(),
          phi::make_ddim(input)));

  if (!activation_type.empty()) {
    PADDLE_ENFORCE_EQ(activation_type,
                      "relu",
                      phi::errors::InvalidArgument(
                          "The attribute activation_type of fc is expected "
                          "to be \"relu\", but received %s.",
                          activation_type.c_str()));
  }

  std::vector<int64_t> output_dims;
  FCOutputSize(phi::make_ddim(input),
               phi::make_ddim(w),
               output_dims,
               in_num_col_dims,
               padding_weights);

  return {output_dims};
}

PD_BUILD_OP(custom_fc)
    .Inputs({"Input", "W", "Bias"})
    .Attrs({"in_num_col_dims: int",
            "activation_type: std::string",
            "padding_weights: bool"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomFcOp))
    .SetInferShapeFn(PD_INFER_SHAPE(CustomFcInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(CustomFcInferDtype));

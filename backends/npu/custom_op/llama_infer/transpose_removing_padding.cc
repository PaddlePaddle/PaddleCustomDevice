// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/extension.h"

std::vector<paddle::Tensor> ApplyTransposeRemovingPadding(
    const paddle::Tensor& input,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& padding_offset) {
  auto input_shape = input.shape();
  const int num_head = input_shape[1];
  const int dim_head = input_shape[3];
  const int token_num = padding_offset.shape()[0];
  auto out = paddle::full(
      {token_num, num_head * dim_head}, 0, input.dtype(), input.place());
  return {out};
}

std::vector<std::vector<int64_t>> ApplyTransposeRemovingPaddingInferShape(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& seq_lens_shape,
    const std::vector<int64_t>& padding_offset_shape) {
  return {{padding_offset_shape[0], input_shape[1] * input_shape[3]}};
}

std::vector<paddle::DataType> ApplyTransposeRemovingPaddingInferDtype(
    const paddle::DataType& input_dtype,
    const paddle::DataType& seq_lens_dtype,
    const paddle::DataType& padding_offset_dtype) {
  return {input_dtype};
}

PD_BUILD_OP(transpose_remove_padding)
    .Inputs({"input", "seq_lens", "padding_offset"})
    .Outputs({"fmha_out"})
    .SetKernelFn(PD_KERNEL(ApplyTransposeRemovingPadding))
    .SetInferShapeFn(PD_INFER_SHAPE(ApplyTransposeRemovingPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ApplyTransposeRemovingPaddingInferDtype));

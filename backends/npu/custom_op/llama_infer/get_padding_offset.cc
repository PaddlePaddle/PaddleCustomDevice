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

std::vector<paddle::Tensor> GetPaddingOffset(const paddle::Tensor& input_ids,
                                             const paddle::Tensor& cum_offsets,
                                             const paddle::Tensor& token_num,
                                             const paddle::Tensor& seq_len) {
  std::vector<int64_t> input_ids_shape = input_ids.shape();
  auto cum_offsets_out = cum_offsets.copy_to(cum_offsets.place(), false);
  auto cpu_token_num = token_num.copy_to(paddle::CPUPlace(), true);
  const int token_num_data = cpu_token_num.data<int64_t>()[0];
  auto x_remove_padding = paddle::full(
      {token_num_data}, 0, paddle::DataType::INT64, input_ids.place());
  auto padding_offset = paddle::full(
      {token_num_data}, 0, paddle::DataType::INT32, input_ids.place());
  return {x_remove_padding,
          cum_offsets_out,
          padding_offset};  // , enc_token_num, dec_token_num};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetInferShape(
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& token_num_shape,
    const std::vector<int64_t>& seq_len_shape) {
  int64_t bsz = input_ids_shape[0];
  int64_t seq_len = input_ids_shape[1];
  return {{-1}, {bsz}, {-1}};
}

std::vector<paddle::DataType> GetPaddingOffsetInferDtype(
    const paddle::DataType& input_ids_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& token_num_dtype,
    const paddle::DataType& seq_len_dtype) {
  return {input_ids_dtype, seq_len_dtype, seq_len_dtype};
}

PD_BUILD_OP(get_padding_offset)
    .Inputs({"input_ids", "cum_offsets", "token_num", "seq_len"})
    .Outputs({"x_remove_padding", "cum_offsets_out", "padding_offset"})
    .SetKernelFn(PD_KERNEL(GetPaddingOffset))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetInferDtype));

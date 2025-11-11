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

std::vector<paddle::Tensor> RebuildPadding(const paddle::Tensor& tmp_out,
                                           const paddle::Tensor& padding_offset,
                                           const paddle::Tensor& seq_lens,
                                           const paddle::Tensor& input_ids) {
  std::vector<int64_t> tmp_out_shape = tmp_out.shape();
  const int dim_embed = tmp_out_shape[1];
  const int bsz = seq_lens.shape()[0];
  auto out =
      paddle::full({bsz, dim_embed}, 0, tmp_out.dtype(), tmp_out.place());
  return {out};
}

std::vector<std::vector<int64_t>> RebuildPaddingInferShape(
    const std::vector<int64_t>& tmp_out_shape,
    const std::vector<int64_t>& padding_offset_shape,
    const std::vector<int64_t>& seq_lens_shape,
    const std::vector<int64_t>& input_ids_shape) {
  int64_t bsz = seq_lens_shape[0];
  int64_t dim_embed = tmp_out_shape[1];
  return {{bsz, dim_embed}};
}

std::vector<paddle::DataType> RebuildPaddingInferDtype(
    const paddle::DataType& tmp_out_dtype,
    const paddle::DataType& padding_offset_dtype,
    const paddle::DataType& seq_lens_dtype,
    const paddle::DataType& input_ids_dtype) {
  return {tmp_out_dtype};
}

PD_BUILD_OP(rebuild_padding)
    .Inputs({"tmp_out", "padding_offset", "seq_lens", "input_ids"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(RebuildPadding))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildPaddingInferDtype));

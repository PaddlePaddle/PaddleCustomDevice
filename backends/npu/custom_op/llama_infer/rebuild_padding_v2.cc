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

#include <iostream>
#include <vector>

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"

std::vector<paddle::Tensor> RebuildPaddingV2(
    const paddle::Tensor& tmp_out,      // [token_num, dim_embed]
    const paddle::Tensor& cum_offsets,  // [bsz, 1]
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_encoder,
    int max_input_length) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(tmp_out.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  auto tmp_out_tensor = static_cast<phi::DenseTensor*>(tmp_out.impl().get());
  auto cum_offsets_tensor =
      static_cast<const phi::DenseTensor*>(cum_offsets.impl().get());
  auto seq_lens_decoder_tensor =
      static_cast<const phi::DenseTensor*>(seq_lens_decoder.impl().get());
  auto seq_lens_encoder_tensor =
      static_cast<const phi::DenseTensor*>(seq_lens_encoder.impl().get());

  const int dim_embed = tmp_out.shape().back();
  const int bsz = cum_offsets.shape()[0];

  if (tmp_out.shape().size() == 3) {  // 需要[token_num, dim_embed]的shape
    tmp_out_tensor->Resize(
        {tmp_out.shape()[0] * tmp_out.shape()[1], tmp_out.shape()[2]});
  }

  std::shared_ptr<phi::DenseTensor> out = std::make_shared<phi::DenseTensor>();
  out->Resize({bsz, dim_embed});
  dev_ctx->Alloc(out.get(), tmp_out_tensor->dtype());

  const auto& runner = NpuOpRunner("RebuildPadding",
                                   {*tmp_out_tensor,
                                    *cum_offsets_tensor,
                                    *seq_lens_decoder_tensor,
                                    *seq_lens_encoder_tensor},
                                   {*out},
                                   {{"max_input_length", max_input_length}});
  runner.Run(stream);

  return {paddle::Tensor(out)};
}

std::vector<std::vector<int64_t>> RebuildPaddingV2InferShape(
    const std::vector<int64_t>& tmp_out_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape) {
  int64_t bsz = cum_offsets_shape[0];
  int64_t dim_embed = tmp_out_shape[1];
  return {{bsz, dim_embed}};
}

std::vector<paddle::DataType> RebuildPaddingV2InferDtype(
    const paddle::DataType& tmp_out_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_encoder_dtype) {
  return {tmp_out_dtype};
}

PD_BUILD_OP(rebuild_padding_v2)
    .Inputs({"tmp_out", "cum_offsets", "seq_lens_decoder", "seq_lens_encoder"})
    .Outputs({"out"})
    .Attrs({"max_input_length: int"})
    .SetKernelFn(PD_KERNEL(RebuildPaddingV2))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildPaddingV2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildPaddingV2InferDtype));

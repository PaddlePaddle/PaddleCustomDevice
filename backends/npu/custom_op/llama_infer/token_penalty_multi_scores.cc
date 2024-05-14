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

std::vector<paddle::Tensor> TokenPenaltyMultiScores(
    const paddle::Tensor& pre_ids,
    const paddle::Tensor& logits,
    const paddle::Tensor& penalty_scores,
    const paddle::Tensor& frequency_scores,
    const paddle::Tensor& presence_scores,
    const paddle::Tensor& cur_len,
    const paddle::Tensor& min_len,
    const paddle::Tensor& eos_token_id) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(pre_ids.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto cur_len_tensor =
      static_cast<const phi::DenseTensor*>(cur_len.impl().get());
  auto eos_token_id_tensor =
      static_cast<const phi::DenseTensor*>(eos_token_id.impl().get());
  auto frequency_scores_tensor =
      static_cast<const phi::DenseTensor*>(frequency_scores.impl().get());
  auto logits_tensor =
      static_cast<const phi::DenseTensor*>(logits.impl().get());
  auto min_len_tensor =
      static_cast<const phi::DenseTensor*>(min_len.impl().get());
  auto penalty_scores_tensor =
      static_cast<const phi::DenseTensor*>(penalty_scores.impl().get());
  auto pre_ids_tensor =
      static_cast<const phi::DenseTensor*>(pre_ids.impl().get());
  auto presence_scores_tensor =
      static_cast<const phi::DenseTensor*>(presence_scores.impl().get());

  phi::DenseTensor repeat_times;
  repeat_times.Resize(phi::make_ddim(logits.shape()));
  dev_ctx->template Alloc<int32_t>(&repeat_times);

  static phi::DenseTensor zeros_like_repeat_times;
  if (zeros_like_repeat_times.numel() != repeat_times.numel()) {
    zeros_like_repeat_times.Resize(phi::make_ddim(logits.shape()));
    dev_ctx->template Alloc<int32_t>(&zeros_like_repeat_times);
    ACL_CHECK(
        aclrtMemsetAsync(zeros_like_repeat_times.data(),
                         zeros_like_repeat_times.numel() * sizeof(int32_t),
                         0,
                         zeros_like_repeat_times.numel() * sizeof(int32_t),
                         stream));
  }
  ACL_CHECK(aclrtMemcpyAsync(repeat_times.data(),
                             repeat_times.numel() * sizeof(int32_t),
                             zeros_like_repeat_times.data(),
                             repeat_times.numel() * sizeof(int32_t),
                             ACL_MEMCPY_DEVICE_TO_DEVICE,
                             stream));

  auto logits_out = logits.copy_to(logits.place(), false);
  auto logits_out_tensor =
      static_cast<const phi::DenseTensor*>(logits_out.impl().get());

  std::vector<phi::DenseTensor> inputs = {*pre_ids_tensor,
                                          *logits_out_tensor,
                                          repeat_times,
                                          *penalty_scores_tensor,
                                          *frequency_scores_tensor,
                                          *presence_scores_tensor,
                                          *cur_len_tensor,
                                          *min_len_tensor,
                                          *eos_token_id_tensor};

  std::vector<phi::DenseTensor> outputs = {*logits_out_tensor};

  const auto& runner = NpuOpRunner("TokenPenaltyMultiScores", inputs, outputs);
  runner.Run(stream);

  return {paddle::Tensor(logits_out)};
}

std::vector<std::vector<int64_t>> TokenPenaltyMultiScoresInferShape(
    const std::vector<int64_t>& pre_ids_shape,
    const std::vector<int64_t>& logits_shape,
    const std::vector<int64_t>& penalty_scores_shape,
    const std::vector<int64_t>& frequency_scores_shape,
    const std::vector<int64_t>& presence_scores_shape,
    const std::vector<int64_t>& cur_len_shape,
    const std::vector<int64_t>& min_len_shape,
    const std::vector<int64_t>& eos_token_id_shape) {
  return {logits_shape};
}

std::vector<paddle::DataType> TokenPenaltyMultiScoresInferDtype(
    const paddle::DataType& pre_ids_dtype,
    const paddle::DataType& logits_dtype,
    const paddle::DataType& penalty_scores_dtype,
    const paddle::DataType& frequency_scores_dtype,
    const paddle::DataType& presence_scores_dtype,
    const paddle::DataType& cur_len_dtype,
    const paddle::DataType& min_len_dtype,
    const paddle::DataType& eos_token_id_dtype) {
  return {logits_dtype};
}

PD_BUILD_OP(get_token_penalty_multi_scores)
    .Inputs({"pre_ids",
             "logits",
             "penalty_scores",
             "frequency_scores",
             "presence_scores",
             "cur_len",
             "min_len",
             "eos_token_id"})
    .Outputs({"logits_out"})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScores))
    .SetInferShapeFn(PD_INFER_SHAPE(TokenPenaltyMultiScoresInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TokenPenaltyMultiScoresInferDtype));

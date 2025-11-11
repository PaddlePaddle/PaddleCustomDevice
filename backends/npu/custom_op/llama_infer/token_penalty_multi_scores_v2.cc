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

void TokenPenaltyMultiScoresV2(const paddle::Tensor& pre_ids,
                               const paddle::Tensor& logits,
                               const paddle::Tensor& penalty_scores,
                               const paddle::Tensor& frequency_scores,
                               const paddle::Tensor& presence_scores,
                               const paddle::Tensor& temperatures,
                               const paddle::Tensor& bad_tokens,
                               const paddle::Tensor& cur_len,
                               const paddle::Tensor& min_len,
                               const paddle::Tensor& eos_token_id) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(pre_ids.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto pre_ids_tensor =
      static_cast<const phi::DenseTensor*>(pre_ids.impl().get());
  auto logits_tensor =
      static_cast<const phi::DenseTensor*>(logits.impl().get());

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

  auto penalty_scores_tensor =
      static_cast<const phi::DenseTensor*>(penalty_scores.impl().get());
  auto frequency_scores_tensor =
      static_cast<const phi::DenseTensor*>(frequency_scores.impl().get());
  auto presence_scores_tensor =
      static_cast<const phi::DenseTensor*>(presence_scores.impl().get());
  auto temperatures_tensor =
      static_cast<const phi::DenseTensor*>(temperatures.impl().get());
  auto bad_tokens_tensor =
      static_cast<const phi::DenseTensor*>(bad_tokens.impl().get());
  auto cur_len_tensor =
      static_cast<const phi::DenseTensor*>(cur_len.impl().get());
  auto min_len_tensor =
      static_cast<const phi::DenseTensor*>(min_len.impl().get());
  auto eos_token_id_tensor =
      static_cast<const phi::DenseTensor*>(eos_token_id.impl().get());

  std::shared_ptr<phi::DenseTensor> logits_out =
      std::make_shared<phi::DenseTensor>();
  logits_out->Resize(logits_tensor->dims());
  dev_ctx->Alloc(logits_out.get(), logits_tensor->dtype());

  const auto& runner = NpuOpRunner("TokenPenaltyMultiScoresV2",
                                   {*pre_ids_tensor,
                                    *logits_tensor,
                                    repeat_times,
                                    *penalty_scores_tensor,
                                    *frequency_scores_tensor,
                                    *presence_scores_tensor,
                                    *temperatures_tensor,
                                    *bad_tokens_tensor,
                                    *cur_len_tensor,
                                    *min_len_tensor,
                                    *eos_token_id_tensor},
                                   {*logits_out},
                                   {});
  runner.Run(stream);
}

PD_BUILD_OP(get_token_penalty_multi_scores_v2)
    .Inputs({"pre_ids",
             "logits",
             "penalty_scores",
             "frequency_scores",
             "presence_scores",
             "temperatures",
             "bad_tokens",
             "cur_len",
             "min_len",
             "eos_token_id"})
    .Outputs({"logits_out"})
    .SetInplaceMap({{"logits", "logits_out"}})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScoresV2));

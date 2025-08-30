// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "custom_op/custom_op_common.h"

void min_length_logits_process(float *logits,
                               const int64_t *cur_len,
                               const int64_t *min_len,
                               const int64_t *eos_token_id,
                               const int64_t bs,
                               const int64_t length,
                               const int64_t end_length) {
  for (int bi = 0; bi < bs; ++bi) {
    if (cur_len[bi] < 0) {
      continue;
    }
    if (cur_len[bi] < min_len[bi]) {
      for (int i = 0; i < end_length; ++i) {
        logits[bi * length + eos_token_id[i]] = -1e10;
      }
    }
  }
}

void update_repeat_times(const int64_t *pre_ids,
                         const int64_t *cur_len,
                         int *repeat_times,
                         const int64_t bs,
                         const int64_t length,
                         const int64_t length_id) {
  for (int bi = 0; bi < bs; ++bi) {
    if (cur_len[bi] < 0) {
      continue;
    }
    const int64_t *pre_ids_now = pre_ids + bi * length_id;
    int *repeat_times_now = repeat_times + bi * length;
    for (int i = 0; i < length_id; i++) {
      int64_t id = pre_ids_now[i];
      if (id < 0) {
        break;
      }
      repeat_times_now[id] += 1;
    }
  }
}

void update_value_by_repeat_times(const int *repeat_times,
                                  const float *penalty_scores,
                                  const float *frequency_score,
                                  const float *presence_score,
                                  const float *temperatures,
                                  float *logits,
                                  const int64_t bs,
                                  const int64_t length) {
  for (int bi = 0; bi < bs; ++bi) {
    float *logits_now = logits + bi * length;
    const int *repeat_times_now = repeat_times + bi * length;
    float alpha = static_cast<float>(penalty_scores[bi]);
    float beta = static_cast<float>(frequency_score[bi]);
    float gamma = static_cast<float>(presence_score[bi]);
    for (int i = 0; i < length; ++i) {
      int times = repeat_times_now[i];
      float logit_now = static_cast<float>(logits_now[i]);
      if (times != 0) {
        logit_now = logit_now < 0 ? logit_now * alpha : logit_now / alpha;
        logit_now = logit_now - times * beta - gamma;
      }
      logits_now[i] = static_cast<float>(logit_now / temperatures[bi]);
      //   if (times == 0) {
      //     logits_now[i] = static_cast<float>(logit_now / temperatures[bi]);
      //   }
      //   logit_now = logit_now < 0 ? logit_now * alpha : logit_now / alpha;
      //   logits_now[i] = static_cast<float>(logit_now - times * beta - gamma);
    }
  }
}

void ban_bad_words(float *logits,
                   const int64_t *bad_words_list,
                   const int64_t bs,
                   const int64_t length,
                   const int64_t bad_words_length) {
  for (int bi = 0; bi < bs; ++bi) {
    float *logits_now = logits + bi * length;
    for (int bwid = 0; bwid < bad_words_length; ++bwid) {
      const int64_t bad_words_token_id = bad_words_list[bwid];
      if (bad_words_token_id >= length || bad_words_token_id < 0) continue;
      logits_now[bad_words_token_id] = -1e10;
    }
  }
}

template <paddle::DataType D>
void token_penalty_multi_scores_kernel(const paddle::Tensor &pre_ids,
                                       const paddle::Tensor &logits,
                                       const paddle::Tensor &penalty_scores,
                                       const paddle::Tensor &frequency_score,
                                       const paddle::Tensor &presence_score,
                                       const paddle::Tensor &temperatures,
                                       const paddle::Tensor &bad_tokens,
                                       const paddle::Tensor &cur_len,
                                       const paddle::Tensor &min_len,
                                       const paddle::Tensor &eos_token_id) {
  std::vector<int64_t> shape = logits.shape();
  auto repeat_times =
      paddle::full(shape, 0, paddle::DataType::INT32, paddle::CPUPlace());
  int64_t bs = shape[0];
  int64_t length = shape[1];
  int64_t length_id = pre_ids.shape()[1];
  int64_t end_length = eos_token_id.shape()[0];
  int64_t length_bad_words = bad_tokens.shape()[0];

  auto pre_ids_cpu = pre_ids.copy_to(paddle::CPUPlace(), false);
  auto logits_cpu = logits.copy_to(paddle::CPUPlace(), false);
  auto penalty_scores_cpu = penalty_scores.copy_to(paddle::CPUPlace(), false);
  auto frequency_score_cpu = frequency_score.copy_to(paddle::CPUPlace(), false);
  auto presence_score_cpu = presence_score.copy_to(paddle::CPUPlace(), false);
  auto temperatures_cpu = temperatures.copy_to(paddle::CPUPlace(), false);
  auto bad_tokens_cpu = bad_tokens.copy_to(paddle::CPUPlace(), false);
  auto cur_len_cpu = cur_len.copy_to(paddle::CPUPlace(), false);
  auto min_len_cpu = min_len.copy_to(paddle::CPUPlace(), false);
  auto eos_token_id_cpu = eos_token_id.copy_to(paddle::CPUPlace(), true);

  min_length_logits_process(const_cast<float *>(logits_cpu.data<float>()),
                            cur_len_cpu.data<int64_t>(),
                            min_len_cpu.data<int64_t>(),
                            eos_token_id_cpu.data<int64_t>(),
                            bs,
                            length,
                            end_length);

  update_repeat_times(pre_ids_cpu.data<int64_t>(),
                      cur_len_cpu.data<int64_t>(),
                      repeat_times.data<int>(),
                      bs,
                      length,
                      length_id);

  update_value_by_repeat_times(repeat_times.data<int>(),
                               penalty_scores_cpu.data<float>(),
                               frequency_score_cpu.data<float>(),
                               presence_score_cpu.data<float>(),
                               temperatures_cpu.data<float>(),
                               const_cast<float *>(logits_cpu.data<float>()),
                               bs,
                               length);

  ban_bad_words(const_cast<float *>(logits_cpu.data<float>()),
                bad_tokens_cpu.data<int64_t>(),
                bs,
                length,
                length_bad_words);

  paddle::Tensor *logits_ptr = const_cast<paddle::Tensor *>(&logits);
  logits_ptr->copy_(logits_cpu, logits.place(), true);
}

void TokenPenaltyMultiScoresCPU(const paddle::Tensor &pre_ids,
                                const paddle::Tensor &logits,
                                const paddle::Tensor &penalty_scores,
                                const paddle::Tensor &frequency_scores,
                                const paddle::Tensor &presence_scores,
                                const paddle::Tensor &temperatures,
                                const paddle::Tensor &bad_tokens,
                                const paddle::Tensor &cur_len,
                                const paddle::Tensor &min_len,
                                const paddle::Tensor &eos_token_id) {
  PADDLE_GCU_KERNEL_TRACE("get_token_penalty_multi_scores_gcu");
  VLOG(6)
      << "[CUSTOM_KERNEL] Custom Operator: get_token_penalty_multi_scores_gcu";
  return token_penalty_multi_scores_kernel<paddle::DataType::FLOAT32>(
      pre_ids,
      logits,
      penalty_scores,
      frequency_scores,
      presence_scores,
      temperatures,
      bad_tokens,
      cur_len,
      min_len,
      eos_token_id);
}

void TokenPenaltyMultiScores(const paddle::Tensor &pre_ids,
                             const paddle::Tensor &prompt_ids,
                             const paddle::Tensor &prompt_len,
                             const paddle::Tensor &logits,
                             const paddle::Tensor &penalty_scores,
                             const paddle::Tensor &frequency_scores,
                             const paddle::Tensor &presence_scores,
                             const paddle::Tensor &temperatures,
                             const paddle::Tensor &bad_tokens,
                             const paddle::Tensor &cur_len,
                             const paddle::Tensor &min_len,
                             const paddle::Tensor &eos_token_id) {
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(logits.place()));

  // Inplace: in & out
  auto logits_tensor =
      static_cast<const phi::DenseTensor *>(logits.impl().get());

  // Inputs
  auto pre_ids_tensor =
      static_cast<const phi::DenseTensor *>(pre_ids.impl().get());
  auto prompt_ids_tensor =
      static_cast<const phi::DenseTensor *>(prompt_ids.impl().get());
  auto prompt_len_tensor =
      static_cast<const phi::DenseTensor *>(prompt_len.impl().get());
  auto penalty_scores_tensor =
      static_cast<const phi::DenseTensor *>(penalty_scores.impl().get());
  auto frequency_scores_tensor =
      static_cast<const phi::DenseTensor *>(frequency_scores.impl().get());
  auto presence_scores_tensor =
      static_cast<const phi::DenseTensor *>(presence_scores.impl().get());
  auto temperatures_tensor =
      static_cast<const phi::DenseTensor *>(temperatures.impl().get());
  auto bad_tokens_tensor =
      static_cast<const phi::DenseTensor *>(bad_tokens.impl().get());
  auto cur_len_tensor =
      static_cast<const phi::DenseTensor *>(cur_len.impl().get());
  auto min_len_tensor =
      static_cast<const phi::DenseTensor *>(min_len.impl().get());
  auto eos_token_id_tensor =
      static_cast<const phi::DenseTensor *>(eos_token_id.impl().get());

  // aten inputs and outputs
  auto logits_aten = custom_kernel::CreateTopsatenTensor(*logits_tensor);
  auto pre_ids_aten = custom_kernel::CreateTopsatenTensor(*pre_ids_tensor);
  auto prompt_ids_aten =
      custom_kernel::CreateTopsatenTensor(*prompt_ids_tensor);
  auto prompt_len_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *prompt_len_tensor, {prompt_len_tensor->numel()}));
  auto penalty_scores_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *penalty_scores_tensor, {penalty_scores_tensor->numel()}));
  auto frequency_scores_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *frequency_scores_tensor, {frequency_scores_tensor->numel()}));
  auto presence_scores_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *presence_scores_tensor, {presence_scores_tensor->numel()}));
  auto temperatures_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *temperatures_tensor, {temperatures_tensor->numel()}));
  auto bad_tokens_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *bad_tokens_tensor, {bad_tokens_tensor->numel()}));
  auto cur_len_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *cur_len_tensor, {cur_len_tensor->numel()}));
  auto min_len_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *min_len_tensor, {min_len_tensor->numel()}));
  auto eos_token_id_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *eos_token_id_tensor, {eos_token_id_tensor->numel()}));

  auto stream = static_cast<topsStream_t>(dev_ctx->stream());

  auto op_info = [&]() -> std::string {
    return custom_kernel::GetOpInfo("topspaddleGetTokenPenaltyMultiScores",
                                    *logits_tensor,
                                    *pre_ids_tensor,
                                    *prompt_ids_tensor,
                                    *prompt_len_tensor,
                                    *penalty_scores_tensor,
                                    *frequency_scores_tensor,
                                    *presence_scores_tensor,
                                    *temperatures_tensor,
                                    *bad_tokens_tensor,
                                    *cur_len_tensor,
                                    *min_len_tensor,
                                    *eos_token_id_tensor,
                                    stream);
  };
  auto abstract_info = [&]() -> std::string {
    return custom_kernel::GetAbstractInfo(
        "topspaddleGetTokenPenaltyMultiScores",
        *logits_tensor,
        *pre_ids_tensor,
        *prompt_ids_tensor,
        *prompt_len_tensor,
        *penalty_scores_tensor,
        *frequency_scores_tensor,
        *presence_scores_tensor,
        *temperatures_tensor,
        *bad_tokens_tensor,
        *cur_len_tensor,
        *min_len_tensor,
        *eos_token_id_tensor,
        stream);
  };

  VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();
  if (custom_kernel::ProfilerIsOn()) {
    auto abstract_info_str = abstract_info();
    GCU_AOT_KERNEL_TRACE(abstract_info_str);
  }

#if 0
  auto status =
      topspaddle::topspaddleGetTokenPenaltyMultiScores(logits_aten,
                                                       pre_ids_aten,
                                                       prompt_ids_aten,
                                                       prompt_len_aten,
                                                       penalty_scores_aten,
                                                       frequency_scores_aten,
                                                       presence_scores_aten,
                                                       temperatures_aten,
                                                       bad_tokens_aten,
                                                       cur_len_aten,
                                                       min_len_aten,
                                                       eos_token_id_aten,
                                                       stream);
  PADDLE_ENFORCE_EQ(status,
                    TOPSATEN_STATUS_SUCCESS,
                    phi::errors::Fatal(
                        "Failed to call aten op "
                        "topspaddle::topspaddleGetTokenPenaltyMultiScores, get "
                        "error: %d, details: %s",
                        status,
                        op_info().c_str()));
#endif
  VLOG(6) << "Launch tops aten op successfully, details:" << op_info();
}

void TokenPenaltyMultiScoresKernel(const paddle::Tensor &pre_ids,
                                   const paddle::Tensor &prompt_ids,
                                   const paddle::Tensor &prompt_len,
                                   const paddle::Tensor &logits,
                                   const paddle::Tensor &penalty_scores,
                                   const paddle::Tensor &frequency_scores,
                                   const paddle::Tensor &presence_scores,
                                   const paddle::Tensor &temperatures,
                                   const paddle::Tensor &bad_tokens,
                                   const paddle::Tensor &cur_len,
                                   const paddle::Tensor &min_len,
                                   const paddle::Tensor &eos_token_id) {
  PADDLE_GCU_KERNEL_TRACE("get_token_penalty_multi_scores_gcu");
  VLOG(6)
      << "[CUSTOM_KERNEL] Custom Operator: get_token_penalty_multi_scores_gcu";
  if (custom_kernel::IsScorpio()) {
    TokenPenaltyMultiScoresCPU(pre_ids,
                               logits,
                               penalty_scores,
                               frequency_scores,
                               presence_scores,
                               temperatures,
                               bad_tokens,
                               cur_len,
                               min_len,
                               eos_token_id);
  } else {
    TokenPenaltyMultiScores(pre_ids,
                            prompt_ids,
                            prompt_len,
                            logits,
                            penalty_scores,
                            frequency_scores,
                            presence_scores,
                            temperatures,
                            bad_tokens,
                            cur_len,
                            min_len,
                            eos_token_id);
  }
}

PD_BUILD_OP(get_token_penalty_multi_scores_gcu)
    .Inputs({"pre_ids",
             "prompt_ids",
             "prompt_len",
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
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScoresKernel));

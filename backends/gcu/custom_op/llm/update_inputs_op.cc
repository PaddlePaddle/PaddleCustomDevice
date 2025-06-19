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

#include "custom_op/custom_op_common.h"

void update_inputs_kernel(bool *not_need_stop,
                          int *seq_lens_this_time,
                          int *seq_lens_encoder,
                          int *seq_lens_decoder,
                          int64_t *input_ids,
                          const int64_t *stop_nums,
                          const bool *stop_flags,
                          const bool *is_block_step,
                          const int64_t *next_tokens,
                          const int bsz,
                          const int input_ids_stride) {
  int64_t stop_sum = 0;
  for (int bi = 0; bi < bsz; ++bi) {
    bool stop_flag_now = false;
    int64_t stop_flag_now_int = 0;
    stop_flag_now = stop_flags[bi];
    stop_flag_now_int = static_cast<int64_t>(stop_flag_now);
    auto seq_len_this_time = seq_lens_this_time[bi];
    auto seq_len_encoder = seq_lens_encoder[bi];
    auto seq_len_decoder = seq_lens_decoder[bi];
    seq_lens_decoder[bi] =
        stop_flag_now
            ? 0
            : (seq_len_decoder == 0 ? seq_len_encoder : seq_len_decoder + 1);
    seq_lens_this_time[bi] = stop_flag_now ? 0 : 1;
    seq_lens_encoder[bi] = 0;
    int64_t *input_ids_now = input_ids + bi * input_ids_stride;
    input_ids_now[0] = next_tokens[bi];
    stop_sum += stop_flag_now_int;
  }
  not_need_stop[0] = stop_sum < stop_nums[0];
}

void UpdateInputes(const paddle::Tensor &stop_flags,
                   const paddle::Tensor &not_need_stop,
                   const paddle::Tensor &seq_lens_this_time,
                   const paddle::Tensor &seq_lens_encoder,
                   const paddle::Tensor &seq_lens_decoder,
                   const paddle::Tensor &input_ids,
                   const paddle::Tensor &stop_nums,
                   const paddle::Tensor &next_tokens,
                   const paddle::Tensor &is_block_step) {
  PADDLE_GCU_KERNEL_TRACE("update_inputs_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: update_inputs_gcu";
  const int bsz = input_ids.shape()[0];
  const int input_ids_stride = input_ids.shape()[1];

  auto stop_flags_cpu = stop_flags.copy_to(paddle::CPUPlace(), false);
  auto not_need_stop_cpu = not_need_stop.copy_to(paddle::CPUPlace(), false);
  auto seq_lens_this_time_cpu =
      seq_lens_this_time.copy_to(paddle::CPUPlace(), false);
  auto seq_lens_encoder_cpu =
      seq_lens_encoder.copy_to(paddle::CPUPlace(), false);
  auto seq_lens_decoder_cpu =
      seq_lens_decoder.copy_to(paddle::CPUPlace(), false);
  auto input_ids_cpu = input_ids.copy_to(paddle::CPUPlace(), false);
  auto stop_nums_cpu = stop_nums.copy_to(paddle::CPUPlace(), false);
  auto next_tokens_cpu = next_tokens.copy_to(paddle::CPUPlace(), false);
  auto is_block_step_cpu = is_block_step.copy_to(paddle::CPUPlace(), true);

  update_inputs_kernel(not_need_stop_cpu.data<bool>(),
                       seq_lens_this_time_cpu.data<int>(),
                       seq_lens_encoder_cpu.data<int>(),
                       seq_lens_decoder_cpu.data<int>(),
                       input_ids_cpu.data<int64_t>(),
                       stop_nums_cpu.data<int64_t>(),
                       stop_flags_cpu.data<bool>(),
                       is_block_step_cpu.data<bool>(),
                       next_tokens_cpu.data<int64_t>(),
                       bsz,
                       input_ids_stride);

  paddle::Tensor *not_need_stop_ptr =
      const_cast<paddle::Tensor *>(&not_need_stop);
  not_need_stop_ptr->copy_(not_need_stop_cpu, not_need_stop.place(), false);

  paddle::Tensor *seq_lens_this_time_ptr =
      const_cast<paddle::Tensor *>(&seq_lens_this_time);
  seq_lens_this_time_ptr->copy_(
      seq_lens_this_time_cpu, seq_lens_this_time.place(), false);

  paddle::Tensor *seq_lens_encoder_ptr =
      const_cast<paddle::Tensor *>(&seq_lens_encoder);
  seq_lens_encoder_ptr->copy_(
      seq_lens_encoder_cpu, seq_lens_encoder.place(), false);

  paddle::Tensor *seq_lens_decoder_ptr =
      const_cast<paddle::Tensor *>(&seq_lens_decoder);
  seq_lens_decoder_ptr->copy_(
      seq_lens_decoder_cpu, seq_lens_decoder.place(), false);

  paddle::Tensor *input_ids_ptr = const_cast<paddle::Tensor *>(&input_ids);
  input_ids_ptr->copy_(input_ids_cpu, input_ids.place(), true);
}

PD_BUILD_OP(update_inputs_gcu)
    .Inputs({"stop_flags",
             "not_need_stop",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "input_ids",
             "stop_nums",
             "next_tokens",
             "is_block_step"})
    .Outputs({"not_need_stop_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "input_ids_out"})
    .SetInplaceMap({{"not_need_stop", "not_need_stop_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"input_ids", "input_ids_out"}})
    .SetKernelFn(PD_KERNEL(UpdateInputes));

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
                          const int now_bsz,
                          const int max_bsz,
                          const int input_ids_stride) {
  int64_t stop_sum = 0;
  for (int bi = 0; bi < max_bsz; ++bi) {
    int64_t stop_flag_now_int = 0;
    if (bi < now_bsz) {
      bool stop_flag_now = false;
      stop_flag_now = stop_flags[bi];
      if (is_block_step[bi]) {
        stop_flag_now_int = 0;
      } else {
        stop_flag_now_int = static_cast<int64_t>(stop_flag_now);
      }
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
    } else {
      stop_flag_now_int = 1;
    }

    stop_sum += stop_flag_now_int;
  }
  not_need_stop[0] = stop_sum < stop_nums[0];
}

void UpdateInputesCPU(const paddle::Tensor &stop_flags,
                      const paddle::Tensor &not_need_stop,
                      const paddle::Tensor &seq_lens_this_time,
                      const paddle::Tensor &seq_lens_encoder,
                      const paddle::Tensor &seq_lens_decoder,
                      const paddle::Tensor &input_ids,
                      const paddle::Tensor &stop_nums,
                      const paddle::Tensor &next_tokens,
                      const paddle::Tensor &is_block_step) {
  PADDLE_GCU_KERNEL_TRACE("update_inputs_gcu_cpu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: update_inputs_gcu_cpu";
  const int max_bsz = stop_flags.shape()[0];
  const int now_bsz = seq_lens_this_time.shape()[0];
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
                       now_bsz,
                       max_bsz,
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
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          input_ids.place()));

  // Inplace: in & out
  auto not_need_stop_gcu = not_need_stop.copy_to(stop_flags.place(), false);
  auto not_need_stop_tensor =
      static_cast<const phi::DenseTensor *>(not_need_stop_gcu.impl().get());
  auto seq_lens_this_time_tensor =
      static_cast<const phi::DenseTensor *>(seq_lens_this_time.impl().get());
  auto seq_lens_encoder_tensor =
      static_cast<const phi::DenseTensor *>(seq_lens_encoder.impl().get());
  auto seq_lens_decoder_tensor =
      static_cast<const phi::DenseTensor *>(seq_lens_decoder.impl().get());
  auto input_ids_tensor =
      static_cast<const phi::DenseTensor *>(input_ids.impl().get());

  // Inputs
  auto stop_flags_tensor =
      static_cast<const phi::DenseTensor *>(stop_flags.impl().get());
  auto stop_nums_tensor =
      static_cast<const phi::DenseTensor *>(stop_nums.impl().get());
  auto next_tokens_tensor =
      static_cast<const phi::DenseTensor *>(next_tokens.impl().get());
  auto is_block_step_tensor =
      static_cast<const phi::DenseTensor *>(is_block_step.impl().get());

  // aten inputs and outputs
  auto not_need_stop_aten =
      custom_kernel::CreateTopsatenTensor(*not_need_stop_tensor);
  auto seq_lens_this_time_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *seq_lens_this_time_tensor, {seq_lens_this_time_tensor->numel()}));
  auto seq_lens_encoder_aten =
      custom_kernel::CreateTopsatenTensor(*seq_lens_encoder_tensor);
  auto seq_lens_decoder_aten =
      custom_kernel::CreateTopsatenTensor(*seq_lens_decoder_tensor);
  auto input_ids_aten = custom_kernel::CreateTopsatenTensor(*input_ids_tensor);

  auto stop_flags_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *stop_flags_tensor, {stop_flags_tensor->numel()}));
  auto stop_nums_aten = custom_kernel::CreateTopsatenTensor(*stop_nums_tensor);
  auto next_tokens_aten =
      custom_kernel::CreateTopsatenTensor(custom_kernel::ReshapeWithoutCopy(
          *next_tokens_tensor, {next_tokens_tensor->numel()}));
  auto is_block_step_aten =
      custom_kernel::CreateTopsatenTensor(*is_block_step_tensor);

  auto stream = static_cast<topsStream_t>(dev_ctx->stream());

  auto op_info = [&]() -> std::string {
    return custom_kernel::GetOpInfo("topspaddleUpdateInputs",
                                    *not_need_stop_tensor,
                                    *seq_lens_this_time_tensor,
                                    *seq_lens_encoder_tensor,
                                    *seq_lens_decoder_tensor,
                                    *input_ids_tensor,
                                    *stop_flags_tensor,
                                    *stop_nums_tensor,
                                    *next_tokens_tensor,
                                    *is_block_step_tensor,
                                    stream);
  };
  auto abstract_info = [&]() -> std::string {
    return custom_kernel::GetAbstractInfo("topspaddleUpdateInputs",
                                          *not_need_stop_tensor,
                                          *seq_lens_this_time_tensor,
                                          *seq_lens_encoder_tensor,
                                          *seq_lens_decoder_tensor,
                                          *input_ids_tensor,
                                          *stop_flags_tensor,
                                          *stop_nums_tensor,
                                          *next_tokens_tensor,
                                          *is_block_step_tensor,
                                          stream);
  };

  VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();
  if (custom_kernel::ProfilerIsOn()) {
    auto abstract_info_str = abstract_info();
    GCU_AOT_KERNEL_TRACE(abstract_info_str);
  }

#if 0
  auto status = topspaddle::topspaddleUpdateInputs(not_need_stop_aten,
                                                   seq_lens_this_time_aten,
                                                   seq_lens_encoder_aten,
                                                   seq_lens_decoder_aten,
                                                   input_ids_aten,
                                                   stop_flags_aten,
                                                   stop_nums_aten,
                                                   next_tokens_aten,
                                                   is_block_step_aten,
                                                   stream);
  PADDLE_ENFORCE_EQ(
      status,
      TOPSATEN_STATUS_SUCCESS,
      phi::errors::Fatal("Failed to call aten op "
                         "topspaddle::topspaddleUpdateInputs, get "
                         "error: %d, details: %s",
                         status,
                         op_info().c_str()));
#endif

  VLOG(6) << "Launch tops aten op successfully, details:" << op_info();

  paddle::Tensor *not_need_stop_ptr =
      const_cast<paddle::Tensor *>(&not_need_stop);
  not_need_stop_ptr->copy_(not_need_stop_gcu, not_need_stop.place(), true);
}

void UpdateInputesKernel(const paddle::Tensor &stop_flags,
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
  if (custom_kernel::IsScorpio()) {
    UpdateInputesCPU(stop_flags,
                     not_need_stop,
                     seq_lens_this_time,
                     seq_lens_encoder,
                     seq_lens_decoder,
                     input_ids,
                     stop_nums,
                     next_tokens,
                     is_block_step);
  } else {
    UpdateInputes(stop_flags,
                  not_need_stop,
                  seq_lens_this_time,
                  seq_lens_encoder,
                  seq_lens_decoder,
                  input_ids,
                  stop_nums,
                  next_tokens,
                  is_block_step);
  }
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
    .SetKernelFn(PD_KERNEL(UpdateInputesKernel));

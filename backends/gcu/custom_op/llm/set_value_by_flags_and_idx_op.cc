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

void set_value_by_flag_and_id(const bool *stop_flags,
                              int64_t *pre_ids_all,
                              const int64_t *input_ids,
                              const int *seq_lens_encoder,
                              const int *seq_lens_decoder,
                              const int64_t *step_idx,
                              int bs,
                              int length,
                              int length_input_ids) {
  for (int bi = 0; bi < bs; bi++) {
    if (!stop_flags[bi]) {
      const int seq_len_dec = seq_lens_decoder[bi];
      const int seq_len_enc = seq_lens_encoder[bi];
      int64_t *pre_ids_all_now = pre_ids_all + bi * length;
      const int64_t *input_ids_now = input_ids + bi * length_input_ids;
      if (seq_len_dec == 0) {
        pre_ids_all_now[step_idx[bi]] = input_ids_now[seq_len_enc - 1];
      } else {
        pre_ids_all_now[step_idx[bi]] = input_ids_now[0];
      }
    }
  }
}

void SetValueByFlagsAndIdx(const paddle::Tensor &pre_ids_all,
                           const paddle::Tensor &input_ids,
                           const paddle::Tensor &seq_lens_this_time,
                           const paddle::Tensor &seq_lens_encoder,
                           const paddle::Tensor &seq_lens_decoder,
                           const paddle::Tensor &step_idx,
                           const paddle::Tensor &stop_flags) {
  PADDLE_GCU_KERNEL_TRACE("set_value_by_flags_and_idx_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: set_value_by_flags_and_idx_gcu";
  std::vector<int64_t> pre_ids_all_shape = pre_ids_all.shape();
  int bs = seq_lens_this_time.shape()[0];
  int length = pre_ids_all_shape[1];
  int length_input_ids = input_ids.shape()[1];

  auto pre_ids_all_cpu = pre_ids_all.copy_to(paddle::CPUPlace(), false);
  auto input_ids_cpu = input_ids.copy_to(paddle::CPUPlace(), false);
  auto seq_lens_this_time_cpu =
      seq_lens_this_time.copy_to(paddle::CPUPlace(), false);
  auto seq_lens_encoder_cpu =
      seq_lens_encoder.copy_to(paddle::CPUPlace(), false);
  auto seq_lens_decoder_cpu =
      seq_lens_decoder.copy_to(paddle::CPUPlace(), false);
  auto step_idx_cpu = step_idx.copy_to(paddle::CPUPlace(), false);
  auto stop_flags_cpu = stop_flags.copy_to(paddle::CPUPlace(), true);

  set_value_by_flag_and_id(
      stop_flags_cpu.data<bool>(),
      const_cast<int64_t *>(pre_ids_all_cpu.data<int64_t>()),
      input_ids_cpu.data<int64_t>(),
      seq_lens_encoder_cpu.data<int>(),
      seq_lens_decoder_cpu.data<int>(),
      step_idx_cpu.data<int64_t>(),
      bs,
      length,
      length_input_ids);

  paddle::Tensor *pre_ids_all_ptr = const_cast<paddle::Tensor *>(&pre_ids_all);
  pre_ids_all_ptr->copy_(pre_ids_all_cpu, pre_ids_all.place(), true);
}

PD_BUILD_OP(set_value_by_flags_and_idx_gcu)
    .Inputs({"pre_ids_all",
             "input_ids",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx",
             "stop_flags"})
    .Outputs({"pre_ids_all_out"})
    .SetInplaceMap({{"pre_ids_all", "pre_ids_all_out"}})
    .SetKernelFn(PD_KERNEL(SetValueByFlagsAndIdx));

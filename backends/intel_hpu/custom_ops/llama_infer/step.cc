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
#include <dlfcn.h>
#include <omp.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

bool in_need_block_list(const int &qid,
                        int *need_block_list,
                        const int &need_block_len) {
  bool res = false;
  for (int i = 0; i < need_block_len; i++) {
    if (qid == need_block_list[i]) {
      res = true;
      need_block_list[i] = -1;
      break;
    }
  }
  return res;
}

void free_and_dispatch_block(bool *stop_flags,
                             int *seq_lens_this_time,
                             int *seq_lens_decoder,
                             int *block_tables,
                             int *encoder_block_lens,
                             bool *is_block_step,
                             int *step_block_list,
                             int *step_len,
                             int *recover_block_list,
                             int *recover_len,
                             int *need_block_list,
                             int *need_block_len,
                             int *used_list_len,
                             int *free_list,
                             int *free_list_len,
                             int64_t *first_token_ids,
                             const int bsz,
                             const int block_size,
                             const int block_num_per_seq,
                             const int max_decoder_block_num) {
#pragma omp parallel for num_threads(OMP_THREAD_NUM)
  for (int bid = 0; bid < bsz; bid++) {
    int *block_table_now = block_tables + bid * block_num_per_seq;
    if (stop_flags[bid] && !is_block_step[bid]) {
      first_token_ids[bid] = -1;
      const int encoder_block_len = encoder_block_lens[bid];
      const int decoder_used_len = used_list_len[bid];
      if (decoder_used_len > 0) {
        int ori_free_list_len = 0;
#pragma omp critical
        {
          ori_free_list_len = *free_list_len;
          *free_list_len = *free_list_len + decoder_used_len;
        }
        for (int i = 0; i < decoder_used_len; i++) {
          free_list[ori_free_list_len + i] =
              block_table_now[encoder_block_len + i];
          block_table_now[encoder_block_len + i] = -1;
        }
        encoder_block_lens[bid] = 0;
        used_list_len[bid] = 0;
      }
    } else if (seq_lens_this_time[bid] != 0 &&
               block_table_now[seq_lens_decoder[bid] / block_size] == -1) {
#pragma omp atomic
      need_block_len[0]++;
      need_block_list[need_block_len[0] - 1] = bid;
    }
  }

  int in_need_block_list_len = 0;
  bool step_max_block_flag = false;
  while (need_block_len[0] > free_list_len[0]) {
    int max_bid = -1;
    int max_used_block_num = 0;
#pragma omp parallel for num_threads(OMP_THREAD_NUM)
    for (int i = 0; i < bsz; i++) {
      if (!is_block_step[i] &&
          (step_max_block_flag || used_list_len[i] != max_decoder_block_num)) {
        int used_block_num = used_list_len[i];
#pragma omp critical
        {
          if (used_block_num > max_used_block_num) {
            max_used_block_num = used_block_num;
            max_bid = i;
          }
        }
      }
    }

    if (max_used_block_num == 0) {
      step_max_block_flag = true;
    } else {
      const int encoder_block_len = encoder_block_lens[max_bid];
      int *block_table_now = block_tables + max_bid * block_num_per_seq;
      for (int i = 0; i < max_used_block_num; i++) {
        free_list[free_list_len[0] + i] =
            block_table_now[encoder_block_len + i];
        block_table_now[encoder_block_len + i] = -1;
      }
      step_block_list[step_len[0]] = max_bid;
      if (in_need_block_list(max_bid,
                             need_block_list,
                             need_block_len[0] + in_need_block_list_len)) {
        need_block_len[0] -= 1;
        in_need_block_list_len += 1;
      }
      step_len[0] += 1;
      free_list_len[0] += max_used_block_num;
      stop_flags[max_bid] = true;
      is_block_step[max_bid] = true;
      seq_lens_this_time[max_bid] = 0;
      seq_lens_decoder[max_bid] = 0;
    }
  }

#pragma omp parallel for num_threads(OMP_THREAD_NUM)
  for (int bid = 0; bid < need_block_len[0] + in_need_block_list_len; bid++) {
    const int need_block_id = need_block_list[bid];
    if (need_block_id != -1 && !stop_flags[need_block_id]) {
      used_list_len[need_block_id]++;
      int ori_free_list_len = 0;
#pragma omp critical
      {
        ori_free_list_len = *free_list_len;
        *free_list_len = *free_list_len - 1;
      }
      int *block_table_now = block_tables + need_block_id * block_num_per_seq;
      block_table_now[seq_lens_decoder[need_block_id] / block_size] =
          free_list[ori_free_list_len - 1];
      need_block_list[bid] = -1;
    }
  }

  int ori_free_list_len = free_list_len[0];
  int ori_step_len = step_len[0];
  while (ori_step_len > 0) {
    int ori_step_block_id = step_block_list[ori_step_len - 1];
    int tmp_used_len = used_list_len[ori_step_block_id];
    const int max_decoder_block_num_this_seq =
        max_decoder_block_num - encoder_block_lens[ori_step_block_id];
    int used_len = tmp_used_len + 1 < max_decoder_block_num_this_seq
                       ? tmp_used_len + 1
                       : max_decoder_block_num_this_seq;

    if (ori_free_list_len >= used_len) {
      recover_block_list[recover_len[0]] = ori_step_block_id;
      recover_len[0] += 1;
      is_block_step[ori_step_block_id] = false;
      used_list_len[ori_step_block_id] = used_len;
      ori_free_list_len -= used_len;
      step_block_list[ori_step_len - 1] = -1;
      step_len[0] -= 1;
      ori_step_len = step_len[0];
    } else {
      break;
    }
  }
  need_block_len[0] = 0;
}

void copy_tensor(const phi::CustomContext *dev_ctx,
                 const paddle::Tensor &src,
                 const paddle::Tensor &dst) {
  auto place = dst.place();
  auto src_dt = dynamic_cast<phi::DenseTensor *>(src.impl().get());
  auto dst_dt = dynamic_cast<phi::DenseTensor *>(dst.impl().get());
  custom_kernel::TensorCopy(*dev_ctx, *src_dt, true, dst_dt, place);
}

void StepPaddle(const paddle::Tensor &stop_flags,
                const paddle::Tensor &seq_lens_this_time,
                const paddle::Tensor &seq_lens_encoder,
                const paddle::Tensor &seq_lens_decoder,
                const paddle::Tensor &block_tables,  // [bsz, block_num_per_seq]
                const paddle::Tensor &encoder_block_lens,
                const paddle::Tensor &is_block_step,
                const paddle::Tensor &step_block_list,
                const paddle::Tensor &step_lens,
                const paddle::Tensor &recover_block_list,
                const paddle::Tensor &recover_lens,
                const paddle::Tensor &need_block_list,
                const paddle::Tensor &need_block_len,
                const paddle::Tensor &used_list_len,
                const paddle::Tensor &free_list,
                const paddle::Tensor &free_list_len,
                const paddle::Tensor &first_token_ids,
                const int block_size,
                const int max_seq_len) {
  auto stop_flags_cpu = stop_flags.copy_to(paddle::CPUPlace(), true);
  auto seq_lens_this_time_cpu =
      seq_lens_this_time.copy_to(paddle::CPUPlace(), true);
  auto seq_lens_encoder_cpu =
      seq_lens_encoder.copy_to(paddle::CPUPlace(), true);
  auto seq_lens_decoder_cpu =
      seq_lens_decoder.copy_to(paddle::CPUPlace(), true);
  auto first_token_ids_cpu = first_token_ids.copy_to(paddle::CPUPlace(), true);

  int bsz = seq_lens_this_time_cpu.shape()[0];
  int block_num_per_seq = block_tables.shape()[1];
  int max_decoder_block_num = max_seq_len / block_size;
  free_and_dispatch_block(stop_flags_cpu.data<bool>(),
                          seq_lens_this_time_cpu.data<int>(),
                          seq_lens_decoder_cpu.data<int>(),
                          const_cast<int *>(block_tables.data<int>()),
                          const_cast<int *>(encoder_block_lens.data<int>()),
                          const_cast<bool *>(is_block_step.data<bool>()),
                          const_cast<int *>(step_block_list.data<int>()),
                          const_cast<int *>(step_lens.data<int>()),
                          const_cast<int *>(recover_block_list.data<int>()),
                          const_cast<int *>(recover_lens.data<int>()),
                          const_cast<int *>(need_block_list.data<int>()),
                          const_cast<int *>(need_block_len.data<int>()),
                          const_cast<int *>(used_list_len.data<int>()),
                          const_cast<int *>(free_list.data<int>()),
                          const_cast<int *>(free_list_len.data<int>()),
                          first_token_ids_cpu.data<int64_t>(),
                          bsz,
                          block_size,
                          block_num_per_seq,
                          max_decoder_block_num);

  // Copy the results back to the original tensors
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          stop_flags.place()));

  copy_tensor(dev_ctx, stop_flags_cpu, stop_flags);
  copy_tensor(dev_ctx, seq_lens_this_time_cpu, seq_lens_this_time);
  copy_tensor(dev_ctx, seq_lens_encoder_cpu, seq_lens_encoder);
  copy_tensor(dev_ctx, seq_lens_decoder_cpu, seq_lens_decoder);
  copy_tensor(dev_ctx, first_token_ids_cpu, first_token_ids);
}

PD_BUILD_OP(step_paddle)
    .Inputs({"stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "block_tables",
             "encoder_block_lens",
             "is_block_step",
             "step_block_list",
             "step_lens",
             "recover_block_list",
             "recover_lens",
             "need_block_list",
             "need_block_len",
             "used_list_len",
             "free_list",
             "free_list_len",
             "first_token_ids"})
    .Attrs({"block_size: int", "max_seq_len: int"})
    .Outputs({"stop_flags_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "block_tables_out",
              "encoder_block_lens_out",
              "is_block_step_out",
              "step_block_list_out",
              "step_lens_out",
              "recover_block_list_out",
              "recover_lens_out",
              "need_block_list_out",
              "need_block_len_out",
              "used_list_len_out",
              "free_list_out",
              "free_list_len_out",
              "first_token_ids_out"})
    .SetInplaceMap({{"stop_flags", "stop_flags_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"block_tables", "block_tables_out"},
                    {"encoder_block_lens", "encoder_block_lens_out"},
                    {"is_block_step", "is_block_step_out"},
                    {"step_block_list", "step_block_list_out"},
                    {"step_lens", "step_lens_out"},
                    {"recover_block_list", "recover_block_list_out"},
                    {"recover_lens", "recover_lens_out"},
                    {"need_block_list", "need_block_list_out"},
                    {"need_block_len", "need_block_len_out"},
                    {"used_list_len", "used_list_len_out"},
                    {"free_list", "free_list_out"},
                    {"free_list_len", "free_list_len_out"},
                    {"first_token_ids", "first_token_ids_out"}})
    .SetKernelFn(PD_KERNEL(StepPaddle));

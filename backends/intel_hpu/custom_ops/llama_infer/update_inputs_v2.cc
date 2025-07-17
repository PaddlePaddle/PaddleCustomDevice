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
#include <omp.h>

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

bool is_in_end_v3(const int64_t id, const int64_t* end_ids, int length) {
  for (int i = 0; i < length; i++) {
    if (id == end_ids[i]) {
      return true;
    }
  }
  return false;
}

void cpu_wrapper(bool* not_need_stop,
                 int64_t* step_idx,
                 bool* stop_flags,
                 int* seq_lens_this_time,
                 int* seq_lens_encoder,
                 int* seq_lens_decoder,
                 int64_t* next_tokens,
                 int64_t* kwargs_next_tokens,
                 int64_t* input_ids,
                 const int64_t* end_ids,
                 const int64_t* stop_nums,
                 const bool* is_block_step,
                 const int64_t* max_dec_len,
                 int bsz,
                 int max_bsz,
                 int input_ids_stride,
                 int end_length) {
#pragma omp parallel for num_threads(OMP_THREAD_NUM)
  for (int i = 0; i < max_bsz; i++) {
    bool stop_flag = stop_flags[i];
    if (!stop_flag) {
      step_idx[i] += 1;
    }
    if (step_idx[i] >= max_dec_len[i]) {
      stop_flags[i] = true;
    }
  }

#pragma omp parallel for num_threads(OMP_THREAD_NUM)
  for (int i = 0; i < bsz; i++) {
    if (stop_flags[i]) {
      if (seq_lens_this_time[i] == 0) {
        next_tokens[i] = -1;
      } else {
        next_tokens[i] = end_ids[0];
        kwargs_next_tokens[i] = end_ids[0];
      }
    } else {
      kwargs_next_tokens[i] = next_tokens[i];
    }
    if (is_in_end_v3(next_tokens[i], end_ids, end_length)) {
      stop_flags[i] = true;
    }
  }

  std::vector<int64_t> stop_flag_now_int(max_bsz, 1);
#pragma omp parallel for num_threads(OMP_THREAD_NUM)
  for (int i = 0; i < bsz; i++) {
    bool stop_flags_now = stop_flags[i];
    stop_flag_now_int[i] = is_block_step[i] ? 0 : (stop_flags_now ? 1 : 0);
    const int seq_len_encoder = seq_lens_encoder[i];
    const int seq_len_decoder = seq_lens_decoder[i];

    seq_lens_decoder[i] =
        stop_flags_now
            ? 0
            : (seq_len_encoder > 0 ? (seq_len_encoder + seq_len_decoder)
                                   : seq_len_decoder + 1);

    seq_lens_this_time[i] = stop_flags[i] ? 0 : 1;
    seq_lens_encoder[i] = 0;
    int64_t* input_ids_now = input_ids + i * input_ids_stride;
    input_ids_now[0] = next_tokens[i];
  }
  int64_t stop_sum = 0;
  for (size_t i = 0; i < stop_flag_now_int.size(); i++) {
    stop_sum += stop_flag_now_int[i];
  }

  not_need_stop[0] = stop_sum < stop_nums[0];
}

void update_inputs_v2(bool* not_need_stop,
                      int64_t* step_idx,
                      bool* stop_flags,
                      int* seq_lens_this_time,
                      int* seq_lens_encoder,
                      int* seq_lens_decoder,
                      int64_t* next_tokens,
                      int64_t* kwargs_next_tokens,
                      int64_t* input_ids,
                      const int64_t* end_ids,
                      const int64_t* stop_nums,
                      const bool* is_block_step,
                      const int64_t* max_dec_len,
                      int now_bsz,
                      int max_bsz,
                      int input_ids_stride,
                      int end_length) {
  PD_CHECK(max_bsz <= 1024,
           "Max supported batch size is 1024. Now received ",
           max_bsz);
  PD_CHECK(now_bsz <= max_bsz,
           "input batch size is larger than max batch size");
  cpu_wrapper(not_need_stop,
              step_idx,
              stop_flags,
              seq_lens_this_time,
              seq_lens_encoder,
              seq_lens_decoder,
              next_tokens,
              kwargs_next_tokens,
              input_ids,
              end_ids,
              stop_nums,
              is_block_step,
              max_dec_len,
              now_bsz,
              max_bsz,
              input_ids_stride,
              end_length);
}

void UpdateInputesV2(const paddle::Tensor& stop_flags,
                     const paddle::Tensor& step_idx,
                     const paddle::Tensor& not_need_stop,  // cpu
                     const paddle::Tensor& seq_lens_this_time,
                     const paddle::Tensor& seq_lens_encoder,
                     const paddle::Tensor& seq_lens_decoder,
                     const paddle::Tensor& max_dec_len,
                     const paddle::Tensor& input_ids,
                     const paddle::Tensor& stop_nums,
                     const paddle::Tensor& next_tokens,
                     const paddle::Tensor& is_block_step,
                     const paddle::Tensor& end_ids,
                     const paddle::Tensor& kwargs_next_tokens) {
  auto stop_flags_cpu = stop_flags.copy_to(paddle::CPUPlace(), true);
  auto step_idx_cpu = step_idx.copy_to(paddle::CPUPlace(), true);
  auto not_need_stop_cpu = not_need_stop.copy_to(paddle::CPUPlace(), true);
  auto seq_lens_this_time_cpu =
      seq_lens_this_time.copy_to(paddle::CPUPlace(), true);
  auto seq_lens_encoder_cpu =
      seq_lens_encoder.copy_to(paddle::CPUPlace(), true);
  auto seq_lens_decoder_cpu =
      seq_lens_decoder.copy_to(paddle::CPUPlace(), true);
  auto max_dec_len_cpu = max_dec_len.copy_to(paddle::CPUPlace(), true);
  auto input_ids_cpu = input_ids.copy_to(paddle::CPUPlace(), true);
  auto stop_nums_cpu = stop_nums.copy_to(paddle::CPUPlace(), true);
  auto next_tokens_cpu = next_tokens.copy_to(paddle::CPUPlace(), true);
  auto is_block_step_cpu = is_block_step.copy_to(paddle::CPUPlace(), true);
  auto end_ids_cpu = end_ids.copy_to(paddle::CPUPlace(), true);
  auto kwargs_next_tokens_cpu =
      kwargs_next_tokens.copy_to(paddle::CPUPlace(), true);

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          stop_flags.place()));

  const int max_bsz = stop_flags.shape()[0];
  const int now_bsz = seq_lens_this_time.shape()[0];
  const int input_ids_stride = input_ids.shape()[1];
  const int end_length = end_ids.shape()[0];
  update_inputs_v2(const_cast<bool*>(not_need_stop_cpu.data<bool>()),
                   const_cast<int64_t*>(step_idx_cpu.data<int64_t>()),
                   const_cast<bool*>(stop_flags_cpu.data<bool>()),
                   const_cast<int*>(seq_lens_this_time_cpu.data<int>()),
                   const_cast<int*>(seq_lens_encoder_cpu.data<int>()),
                   const_cast<int*>(seq_lens_decoder_cpu.data<int>()),
                   const_cast<int64_t*>(next_tokens_cpu.data<int64_t>()),
                   const_cast<int64_t*>(kwargs_next_tokens_cpu.data<int64_t>()),
                   const_cast<int64_t*>(input_ids_cpu.data<int64_t>()),
                   end_ids_cpu.data<int64_t>(),
                   stop_nums_cpu.data<int64_t>(),
                   is_block_step_cpu.data<bool>(),
                   max_dec_len_cpu.data<int64_t>(),
                   now_bsz,
                   max_bsz,
                   input_ids_stride,
                   end_length);

  custom_kernel::copy_tensor_wrapper(dev_ctx, not_need_stop_cpu, not_need_stop);
  custom_kernel::copy_tensor_wrapper(dev_ctx, stop_flags_cpu, stop_flags);
  custom_kernel::copy_tensor_wrapper(dev_ctx, step_idx_cpu, step_idx);
  custom_kernel::copy_tensor_wrapper(
      dev_ctx, seq_lens_this_time_cpu, seq_lens_this_time);
  custom_kernel::copy_tensor_wrapper(
      dev_ctx, seq_lens_encoder_cpu, seq_lens_encoder);
  custom_kernel::copy_tensor_wrapper(
      dev_ctx, seq_lens_decoder_cpu, seq_lens_decoder);
  custom_kernel::copy_tensor_wrapper(dev_ctx, input_ids_cpu, input_ids);
  custom_kernel::copy_tensor_wrapper(dev_ctx, next_tokens_cpu, next_tokens);
  custom_kernel::copy_tensor_wrapper(
      dev_ctx, kwargs_next_tokens_cpu, kwargs_next_tokens);
}

std::vector<std::vector<int64_t>> UpdateInputsV2InferShape(
    const std::vector<int64_t>& stop_flags_shape,
    const std::vector<int64_t>& step_idx_shape,
    const std::vector<int64_t>& not_need_stop_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& max_dec_len_shape,
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& stop_nums_shape,
    const std::vector<int64_t>& next_tokens_shape,
    const std::vector<int64_t>& is_block_step_shape,
    const std::vector<int64_t>& end_ids_shape,
    const std::vector<int64_t>& kwargs_next_tokens_shape) {
  return {stop_flags_shape};
}

std::vector<paddle::DataType> UpdateInputsV2InferDtype(
    const paddle::DataType& stop_flags_dtype,
    const paddle::DataType& step_idx_dtype,
    const paddle::DataType& not_need_stop_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& max_dec_len_dtype,
    const paddle::DataType& input_ids_dtype,
    const paddle::DataType& stop_nums_dtype,
    const paddle::DataType& next_tokens_dtype,
    const paddle::DataType& is_block_step_dtype,
    const paddle::DataType& end_ids_dtype,
    const paddle::DataType& kwargs_next_tokens_dtype) {
  return {stop_flags_dtype};
}

PD_BUILD_OP(update_inputs_v2)
    .Inputs({"stop_flags",
             "step_idx",
             "not_need_stop",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "max_dec_len",
             "input_ids",
             "stop_nums",
             "next_tokens",
             "is_block_step",
             "end_ids",
             "kwargs_next_tokens"})
    .Outputs({"stop_flags_out",
              "not_need_stop_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "input_ids_out",
              "next_tokens_out",
              "kwargs_next_tokens_out",
              "step_idx_out"})
    .SetInplaceMap({{"stop_flags", "stop_flags_out"},
                    {"not_need_stop", "not_need_stop_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"input_ids", "input_ids_out"},
                    {"next_tokens", "next_tokens_out"},
                    {"kwargs_next_tokens", "kwargs_next_tokens_out"},
                    {"step_idx", "step_idx_out"}})
    .SetKernelFn(PD_KERNEL(UpdateInputesV2))
    .SetInferShapeFn(PD_INFER_SHAPE(UpdateInputsV2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(UpdateInputsV2InferDtype));

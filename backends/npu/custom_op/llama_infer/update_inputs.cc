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

void UpdateInputes(const paddle::Tensor& stop_flags,
                   const paddle::Tensor& not_need_stop,  // cpu
                   const paddle::Tensor& seq_lens_this_time,
                   const paddle::Tensor& seq_lens_encoder,
                   const paddle::Tensor& seq_lens_decoder,
                   const paddle::Tensor& input_ids,
                   const paddle::Tensor& stop_nums,
                   const paddle::Tensor& next_tokens,
                   const paddle::Tensor& is_block_step) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          stop_flags.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto stop_flags_tensor =
      static_cast<const phi::DenseTensor*>(stop_flags.impl().get());
  auto seq_lens_this_time_tensor =
      static_cast<const phi::DenseTensor*>(seq_lens_this_time.impl().get());
  auto seq_lens_encoder_tensor =
      static_cast<const phi::DenseTensor*>(seq_lens_encoder.impl().get());
  auto seq_lens_decoder_tensor =
      static_cast<const phi::DenseTensor*>(seq_lens_decoder.impl().get());
  auto input_ids_tensor =
      static_cast<const phi::DenseTensor*>(input_ids.impl().get());
  auto stop_nums_tensor =
      static_cast<const phi::DenseTensor*>(stop_nums.impl().get());
  auto next_tokens_tensor =
      static_cast<const phi::DenseTensor*>(next_tokens.impl().get());
  auto is_block_step_tensor =
      static_cast<const phi::DenseTensor*>(is_block_step.impl().get());

  bool not_need_stop_on_host =
      not_need_stop.place().GetType() == phi::AllocationType::CPU;
  if (not_need_stop_on_host) {
    auto not_need_stop_npu = not_need_stop.copy_to(stop_flags.place(), false);
    auto not_need_stop_tensor =
        static_cast<const phi::DenseTensor*>(not_need_stop_npu.impl().get());
    const auto& runner = NpuOpRunner("UpdateInputs",
                                     {*stop_flags_tensor,
                                      *not_need_stop_tensor,
                                      *seq_lens_this_time_tensor,
                                      *seq_lens_encoder_tensor,
                                      *seq_lens_decoder_tensor,
                                      *input_ids_tensor,
                                      *stop_nums_tensor,
                                      *next_tokens_tensor,
                                      *is_block_step_tensor},
                                     {*not_need_stop_tensor,
                                      *seq_lens_this_time_tensor,
                                      *seq_lens_encoder_tensor,
                                      *seq_lens_decoder_tensor,
                                      *input_ids_tensor},
                                     {});
    runner.Run(stream);
    auto not_need_stop_cpu =
        not_need_stop_npu.copy_to(not_need_stop.place(), true);
    bool* not_need_stop_data = const_cast<bool*>(not_need_stop.data<bool>());
    not_need_stop_data[0] = not_need_stop_cpu.data<bool>()[0];
  } else {
    auto not_need_stop_tensor =
        static_cast<const phi::DenseTensor*>(not_need_stop.impl().get());
    const auto& runner = NpuOpRunner("UpdateInputs",
                                     {*stop_flags_tensor,
                                      *not_need_stop_tensor,
                                      *seq_lens_this_time_tensor,
                                      *seq_lens_encoder_tensor,
                                      *seq_lens_decoder_tensor,
                                      *input_ids_tensor,
                                      *stop_nums_tensor,
                                      *next_tokens_tensor,
                                      *is_block_step_tensor},
                                     {*not_need_stop_tensor,
                                      *seq_lens_this_time_tensor,
                                      *seq_lens_encoder_tensor,
                                      *seq_lens_decoder_tensor,
                                      *input_ids_tensor},
                                     {});
    runner.Run(stream);
  }
}

PD_BUILD_OP(update_inputs)
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

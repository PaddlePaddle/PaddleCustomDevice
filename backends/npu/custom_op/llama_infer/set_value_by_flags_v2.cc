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

void SetValueByFlagsAndIdxV2(const paddle::Tensor& pre_ids_all,
                             const paddle::Tensor& input_ids,
                             const paddle::Tensor& seq_lens_this_time,
                             const paddle::Tensor& seq_lens_encoder,
                             const paddle::Tensor& seq_lens_decoder,
                             const paddle::Tensor& step_idx,
                             const paddle::Tensor& stop_flags) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          pre_ids_all.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto pre_ids_all_tensor =
      static_cast<const phi::DenseTensor*>(pre_ids_all.impl().get());
  auto input_ids_tensor =
      static_cast<const phi::DenseTensor*>(input_ids.impl().get());
  auto seq_lens_this_time_tensor =
      static_cast<const phi::DenseTensor*>(seq_lens_this_time.impl().get());
  auto seq_lens_encoder_tensor =
      static_cast<const phi::DenseTensor*>(seq_lens_encoder.impl().get());
  auto seq_lens_decoder_tensor =
      static_cast<const phi::DenseTensor*>(seq_lens_decoder.impl().get());
  auto step_idx_tensor =
      static_cast<const phi::DenseTensor*>(step_idx.impl().get());
  auto stop_flags_tensor =
      static_cast<const phi::DenseTensor*>(stop_flags.impl().get());

  const auto& runner = NpuOpRunner("SetValueByFlagsAndIdxV2",
                                   {*pre_ids_all_tensor,
                                    *input_ids_tensor,
                                    *seq_lens_this_time_tensor,
                                    *seq_lens_encoder_tensor,
                                    *seq_lens_decoder_tensor,
                                    *step_idx_tensor,
                                    *stop_flags_tensor},
                                   {*pre_ids_all_tensor},
                                   {});
  runner.Run(stream);
}

PD_BUILD_OP(set_value_by_flags_and_idx_v2)
    .Inputs({"pre_ids_all",
             "input_ids",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx",
             "stop_flags"})
    .Outputs({"pre_ids_all_out"})
    .SetInplaceMap({{"pre_ids_all", "pre_ids_all_out"}})
    .SetKernelFn(PD_KERNEL(SetValueByFlagsAndIdxV2));

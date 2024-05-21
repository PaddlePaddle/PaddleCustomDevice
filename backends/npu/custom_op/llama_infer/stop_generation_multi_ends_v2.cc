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

void GetStopFlagsMultiV2(const paddle::Tensor& topk_ids,
                         const paddle::Tensor& stop_flags,
                         const paddle::Tensor& seq_lens,
                         const paddle::Tensor& end_ids,
                         const paddle::Tensor& next_tokens) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          topk_ids.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto topk_ids_tensor =
      static_cast<const phi::DenseTensor*>(topk_ids.impl().get());
  auto stop_flags_tensor =
      static_cast<const phi::DenseTensor*>(stop_flags.impl().get());
  auto seq_lens_tensor =
      static_cast<const phi::DenseTensor*>(seq_lens.impl().get());
  auto end_ids_tensor =
      static_cast<const phi::DenseTensor*>(end_ids.impl().get());
  auto next_tokens_tensor =
      static_cast<const phi::DenseTensor*>(next_tokens.impl().get());

  std::shared_ptr<phi::DenseTensor> topk_ids_out =
      std::make_shared<phi::DenseTensor>();
  topk_ids_out->Resize(topk_ids_tensor->dims());
  dev_ctx->Alloc(topk_ids_out.get(), topk_ids_tensor->dtype());

  std::shared_ptr<phi::DenseTensor> stop_flags_out =
      std::make_shared<phi::DenseTensor>();
  stop_flags_out->Resize(stop_flags_tensor->dims());
  dev_ctx->Alloc(stop_flags_out.get(), stop_flags_tensor->dtype());

  std::shared_ptr<phi::DenseTensor> next_tokens_out =
      std::make_shared<phi::DenseTensor>();
  next_tokens_out->Resize(next_tokens_tensor->dims());
  dev_ctx->Alloc(next_tokens_out.get(), next_tokens_tensor->dtype());

  const auto& runner =
      NpuOpRunner("SetStopValueMultiEndsV2",
                  {*topk_ids_tensor,
                   *stop_flags_tensor,
                   *seq_lens_tensor,
                   *end_ids_tensor,
                   *next_tokens_tensor},
                  {*topk_ids_out, *stop_flags_out, *next_tokens_out},
                  {});
  runner.Run(stream);
}

PD_BUILD_OP(set_stop_value_multi_ends_v2)
    .Inputs({"topk_ids", "stop_flags", "seq_lens", "end_ids", "next_tokens"})
    .Outputs({"topk_ids_out", "stop_flags_out", "next_tokens_out"})
    .SetInplaceMap({{"topk_ids", "topk_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMultiV2));

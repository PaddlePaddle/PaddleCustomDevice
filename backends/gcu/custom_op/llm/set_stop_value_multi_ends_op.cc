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

bool is_in_end(const int64_t id, const int64_t *end_ids, int length) {
  bool flag = false;
  for (int i = 0; i < length; i++) {
    if (id == end_ids[i]) {
      return true;
    }
  }
  return flag;
}

void set_value_by_flags(bool *stop_flags,
                        int64_t *topk_ids,
                        int64_t *next_tokens,
                        const int64_t *end_ids,
                        const int *seq_lens,
                        const int bs,
                        const int end_length,
                        bool beam_search) {
  for (int bi = 0; bi < bs; bi++) {
    if (stop_flags[bi]) {
      if ((seq_lens[bi] == 0)) {
        topk_ids[bi] = -1;
      } else {
        topk_ids[bi] = end_ids[0];
        next_tokens[bi] = end_ids[0];
      }
    } else {
      next_tokens[bi] = topk_ids[bi];
    }
    if (!beam_search && is_in_end(topk_ids[bi], end_ids, end_length)) {
      stop_flags[bi] = true;
    }
  }
}

void GetStopFlagsMultiCPU(const paddle::Tensor &topk_ids,
                          const paddle::Tensor &stop_flags,
                          const paddle::Tensor &seq_lens,
                          const paddle::Tensor &end_ids,
                          const paddle::Tensor &next_tokens,
                          const bool beam_search) {
  std::vector<int64_t> shape = topk_ids.shape();
  int64_t bs_now = shape[0];
  int64_t end_length = end_ids.shape()[0];

  auto topk_ids_cpu = topk_ids.copy_to(paddle::CPUPlace(), false);
  auto stop_flags_cpu = stop_flags.copy_to(paddle::CPUPlace(), false);
  auto seq_lens_cpu = seq_lens.copy_to(paddle::CPUPlace(), false);
  auto end_ids_cpu = end_ids.copy_to(paddle::CPUPlace(), false);
  auto next_tokens_cpu = next_tokens.copy_to(paddle::CPUPlace(), true);

  set_value_by_flags(stop_flags_cpu.data<bool>(),
                     topk_ids_cpu.data<int64_t>(),
                     next_tokens_cpu.data<int64_t>(),
                     end_ids_cpu.data<int64_t>(),
                     seq_lens_cpu.data<int>(),
                     bs_now,
                     end_length,
                     false);

  paddle::Tensor *topk_ids_ptr = const_cast<paddle::Tensor *>(&topk_ids);
  topk_ids_ptr->copy_(topk_ids_cpu, topk_ids.place(), false);

  paddle::Tensor *stop_flags_ptr = const_cast<paddle::Tensor *>(&stop_flags);
  stop_flags_ptr->copy_(stop_flags_cpu, stop_flags.place(), false);

  paddle::Tensor *next_tokens_ptr = const_cast<paddle::Tensor *>(&next_tokens);
  next_tokens_ptr->copy_(next_tokens_cpu, next_tokens.place(), true);
}

void GetStopFlagsMulti(const paddle::Tensor &topk_ids,
                       const paddle::Tensor &stop_flags,
                       const paddle::Tensor &seq_lens,
                       const paddle::Tensor &end_ids,
                       const paddle::Tensor &next_tokens,
                       const bool beam_search) {
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          topk_ids.place()));

  // Inplace: in & out
  auto topk_ids_tensor =
      static_cast<const phi::DenseTensor *>(topk_ids.impl().get());
  auto stop_flags_tensor =
      static_cast<const phi::DenseTensor *>(stop_flags.impl().get());
  auto next_tokens_tensor =
      static_cast<const phi::DenseTensor *>(next_tokens.impl().get());

  // Inputs
  auto seq_lens_tensor =
      static_cast<const phi::DenseTensor *>(seq_lens.impl().get());
  auto end_ids_tensor =
      static_cast<const phi::DenseTensor *>(end_ids.impl().get());

  // aten inputs and outputs
  auto stop_flags_aten =
      custom_kernel::CreateTopsatenTensor(*stop_flags_tensor);
  auto topk_ids_aten = custom_kernel::CreateTopsatenTensor(*topk_ids_tensor);
  auto next_tokens_aten =
      custom_kernel::CreateTopsatenTensor(*next_tokens_tensor);
  auto end_ids_aten = custom_kernel::CreateTopsatenTensor(*end_ids_tensor);
  auto seq_lens_aten = custom_kernel::CreateTopsatenTensor(*seq_lens_tensor);

  auto stream = static_cast<topsStream_t>(dev_ctx->stream());

  auto op_info = [&]() -> std::string {
    return custom_kernel::GetOpInfo("topspaddleGetStopFlagsMulti",
                                    *stop_flags_tensor,
                                    *topk_ids_tensor,
                                    *next_tokens_tensor,
                                    *end_ids_tensor,
                                    *seq_lens_tensor,
                                    beam_search,
                                    stream);
  };
  auto abstract_info = [&]() -> std::string {
    return custom_kernel::GetAbstractInfo("topspaddleGetStopFlagsMulti",
                                          *stop_flags_tensor,
                                          *topk_ids_tensor,
                                          *next_tokens_tensor,
                                          *end_ids_tensor,
                                          *seq_lens_tensor,
                                          beam_search,
                                          stream);
  };

  VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();
  if (custom_kernel::ProfilerIsOn()) {
    auto abstract_info_str = abstract_info();
    GCU_AOT_KERNEL_TRACE(abstract_info_str);
  }

#if 0
  auto status = topspaddle::topspaddleGetStopFlagsMulti(stop_flags_aten,
                                                        topk_ids_aten,
                                                        next_tokens_aten,
                                                        end_ids_aten,
                                                        seq_lens_aten,
                                                        beam_search,
                                                        stream);
  PADDLE_ENFORCE_EQ(
      status,
      TOPSATEN_STATUS_SUCCESS,
      phi::errors::Fatal("Failed to call aten op "
                         "topspaddle::topspaddleGetStopFlagsMulti, get "
                         "error: %d, details: %s",
                         status,
                         op_info().c_str()));
#endif

  VLOG(6) << "Launch tops aten op successfully, details:" << op_info();
}

void GetStopFlagsMultiKernel(const paddle::Tensor &topk_ids,
                             const paddle::Tensor &stop_flags,
                             const paddle::Tensor &seq_lens,
                             const paddle::Tensor &end_ids,
                             const paddle::Tensor &next_tokens,
                             const bool beam_search) {
  PADDLE_GCU_KERNEL_TRACE("set_stop_value_multi_ends_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: set_stop_value_multi_ends_gcu";
  if (custom_kernel::IsScorpio() || custom_kernel::IsLibra()) {
    GetStopFlagsMultiCPU(
        topk_ids, stop_flags, seq_lens, end_ids, next_tokens, beam_search);
  } else {
    GetStopFlagsMulti(
        topk_ids, stop_flags, seq_lens, end_ids, next_tokens, beam_search);
  }
}

PD_BUILD_OP(set_stop_value_multi_ends_gcu)
    .Inputs({"topk_ids", "stop_flags", "seq_lens", "end_ids", "next_tokens"})
    .Attrs({"beam_search: bool"})
    .Outputs({"topk_ids_out", "stop_flags_out", "next_tokens_out"})
    .SetInplaceMap({{"topk_ids", "topk_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMultiKernel));

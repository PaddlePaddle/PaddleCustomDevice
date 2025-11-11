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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void MMHAKernel(const Context &dev_ctx,
                const phi::DenseTensor &x,
                const phi::DenseTensor &cache_kv,
                const paddle::optional<phi::DenseTensor> &bias,
                const paddle::optional<phi::DenseTensor> &src_mask,
                const paddle::optional<phi::DenseTensor> &cum_offsets,
                const paddle::optional<phi::DenseTensor> &sequence_lengths,
                const paddle::optional<phi::DenseTensor> &rotary_tensor,
                const paddle::optional<phi::DenseTensor> &beam_cache_offset,
                const paddle::optional<phi::DenseTensor> &qkv_out_scale,
                const paddle::optional<phi::DenseTensor> &out_shift,
                const paddle::optional<phi::DenseTensor> &out_smooth,
                int seq_len,
                int rotary_emb_dims,
                const bool use_neox_rotary_style,
                const std::string &compute_dtype,
                const float out_scale,
                const int quant_round_type,
                const float quant_max_bound,
                const float quant_min_bound,
                phi::DenseTensor *out,
                phi::DenseTensor *cache_kv_out,
                phi::DenseTensor *beam_cache_offset_out) {
  PADDLE_THROW(phi::errors::Unimplemented("Only supports model export"));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(masked_multihead_attention,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::MMHAKernel,
                          phi::float16,
                          int32_t) {}

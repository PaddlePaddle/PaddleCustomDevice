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
void MultiHeadAttentionVariableForwardKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& query,
    const phi::DenseTensor& key,
    const phi::DenseTensor& value,
    const phi::DenseTensor& seq_lens,
    const phi::DenseTensor& kv_seq_lens,
    const paddle::optional<phi::DenseTensor>& mask,
    const float scale,
    const bool causal,
    const int pre_cache_length,
    phi::DenseTensor* output) {
  PADDLE_THROW(phi::errors::Unimplemented("Only supports model export"));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    variable_length_memory_efficient_attention,
    npu,
    ALL_LAYOUT,
    custom_kernel::MultiHeadAttentionVariableForwardKernel,
    phi::float16) {
  kernel->InputAt(3).SetDataType(phi::DataType::INT32);
}

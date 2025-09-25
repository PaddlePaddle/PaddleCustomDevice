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

void ReshapeAndCacheFlash(const paddle::Tensor& key_cache,
                          const paddle::Tensor& value_cache,
                          const paddle::Tensor& key,
                          const paddle::Tensor& value,
                          const paddle::Tensor& slot_mapping) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(key.place()));

  // [num_blocks, block_size, num_kv_heads, head_size]
  auto key_cache_tensor =
      static_cast<const phi::DenseTensor*>(key_cache.impl().get());
  // [num_blocks, block_size, num_kv_heads, head_size]
  auto value_cache_tensor =
      static_cast<const phi::DenseTensor*>(value_cache.impl().get());
  // [num_tokens, num_kv_heads, head_size] or [num_tokens, hidden_size_kv]
  auto key_tensor = static_cast<const phi::DenseTensor*>(key.impl().get());
  // [num_tokens, num_kv_heads, head_size] or [num_tokens, hidden_size_kv]
  auto value_tensor = static_cast<const phi::DenseTensor*>(value.impl().get());
  // [num_tokens]
  auto slot_mapping_tensor =
      static_cast<const phi::DenseTensor*>(slot_mapping.impl().get());

  PADDLE_ENFORCE_EQ(
      key_tensor->dims(),
      value_tensor->dims(),
      phi::errors::InvalidArgument("key_dims must be same as value_dims, but "
                                   "get key_dims: %s, value_dims: %s.",
                                   key_tensor->dims().to_str(),
                                   value_tensor->dims().to_str()));

  phi::DenseTensor key_reshape = *key_tensor;
  phi::DenseTensor value_reshape = *value_tensor;
  if (key_tensor->dims().size() == 2) {
    int64_t num_tokens = key_tensor->dims().at(0);

    auto kv_cache_dims = key_cache_tensor->dims();
    int64_t num_kv_heads = kv_cache_dims.at(2);
    int64_t head_size = kv_cache_dims.at(3);

    key_reshape = custom_kernel::ReshapeWithoutCopy(
        *key_tensor, {num_tokens, num_kv_heads, head_size});
    value_reshape = custom_kernel::ReshapeWithoutCopy(
        *value_tensor, {num_tokens, num_kv_heads, head_size});
  }

  auto key_cache_aten = custom_kernel::CreateTopsatenTensor(*key_cache_tensor);
  auto value_cache_aten =
      custom_kernel::CreateTopsatenTensor(*value_cache_tensor);
  auto key_aten = custom_kernel::CreateTopsatenTensor(key_reshape);
  auto value_aten = custom_kernel::CreateTopsatenTensor(value_reshape);
  auto slot_mapping_aten =
      custom_kernel::CreateTopsatenTensor(*slot_mapping_tensor);

  auto stream = static_cast<topsStream_t>(dev_ctx->stream());

  auto op_info = [&]() -> std::string {
    return custom_kernel::GetOpInfo("topsvllmReshapeAndCacheFlash",
                                    *key_cache_tensor,
                                    *value_cache_tensor,
                                    key_reshape,
                                    value_reshape,
                                    *slot_mapping_tensor,
                                    stream);
  };
  auto abstract_info = [&]() -> std::string {
    return custom_kernel::GetAbstractInfo("topsvllmReshapeAndCacheFlash",
                                          *key_cache_tensor,
                                          *value_cache_tensor,
                                          key_reshape,
                                          value_reshape,
                                          *slot_mapping_tensor,
                                          stream);
  };

  VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();
  if (custom_kernel::ProfilerIsOn()) {
    auto abstract_info_str = abstract_info();
    GCU_AOT_KERNEL_TRACE(abstract_info_str);
  }
  auto status = topsvllm::topsvllmReshapeAndCacheFlash(key_cache_aten,
                                                       value_cache_aten,
                                                       key_aten,
                                                       value_aten,
                                                       slot_mapping_aten,
                                                       stream);
  PADDLE_ENFORCE_EQ(
      status,
      TOPSATEN_STATUS_SUCCESS,
      phi::errors::Fatal(
          "Failed to call aten op "
          "topsvllm::topsvllmReshapeAndCacheFlash, get error: %d, details: %s",
          status,
          op_info().c_str()));
  VLOG(6) << "Launch tops aten op successfully, details:" << op_info();
}

void ReshapeAndCacheFlashNative(const paddle::Tensor& key_cache,
                                const paddle::Tensor& value_cache,
                                const paddle::Tensor& key,
                                const paddle::Tensor& value,
                                const paddle::Tensor& slot_mapping) {
  // key_cache: [num_blocks, block_size, num_kv_heads, head_size]
  // value_cache: [num_blocks, block_size, num_kv_heads, head_size]
  // key: [num_tokens, num_kv_heads, head_size] or [num_tokens, hidden_size_kv]
  // value: [num_tokens, num_kv_heads, head_size] or [num_tokens,
  // hidden_size_kv] slot_mapping: [num_tokens]
  auto kv_cache_dims = key_cache.dims();
  int64_t num_kv_heads = kv_cache_dims[2];
  int64_t head_size = kv_cache_dims[3];
  auto key_cache_inter =
      paddle::experimental::reshape(key_cache, {-1, num_kv_heads, head_size});
  auto value_cache_inter =
      paddle::experimental::reshape(value_cache, {-1, num_kv_heads, head_size});
  auto key_inter =
      paddle::experimental::reshape(key, {-1, num_kv_heads, head_size});
  auto value_inter =
      paddle::experimental::reshape(value, {-1, num_kv_heads, head_size});
  paddle::experimental::index_put_(
      key_cache_inter, {slot_mapping}, key_inter, false);
  paddle::experimental::index_put_(
      value_cache_inter, {slot_mapping}, value_inter, false);
}

void ReshapeAndCacheFlashKernel(const paddle::Tensor& key_cache,
                                const paddle::Tensor& value_cache,
                                const paddle::Tensor& key,
                                const paddle::Tensor& value,
                                const paddle::Tensor& slot_mapping) {
  PADDLE_GCU_KERNEL_TRACE("reshape_and_cache_flash_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: reshape_and_cache_flash_gcu";
  if (custom_kernel::IsScorpio()) {
    ReshapeAndCacheFlashNative(
        key_cache, value_cache, key, value, slot_mapping);
  } else {
    ReshapeAndCacheFlash(key_cache, value_cache, key, value, slot_mapping);
  }
}

PD_BUILD_OP(reshape_and_cache_flash_gcu)
    .Inputs({"key", "value", "key_cache", "value_cache", "slot_mapping"})
    .Outputs({"key_cache_out", "value_cache_out"})
    .SetInplaceMap({{"key_cache", "key_cache_out"},
                    {"value_cache", "value_cache_out"}})
    .SetKernelFn(PD_KERNEL(ReshapeAndCacheFlashKernel));

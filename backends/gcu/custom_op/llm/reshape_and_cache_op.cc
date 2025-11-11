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

std::vector<std::vector<int64_t>> ReshapeAndCacheQuantInferShape(
    std::vector<int64_t> key_shape,
    std::vector<int64_t> value_shape,
    std::vector<int64_t> k_cache_shape,
    std::vector<int64_t> v_cache_shape,
    std::vector<int64_t> slot_mapping_shape) {
  return {k_cache_shape, v_cache_shape};
}

std::vector<paddle::DataType> ReshapeAndCacheQuantInferDtype(
    const paddle::DataType& key_dtype,
    const paddle::DataType& value_dtype,
    const paddle::DataType& k_cache_dtype,
    const paddle::DataType& v_cache_dtype,
    const paddle::DataType& slot_mapping_dtype) {
  return {k_cache_dtype, v_cache_dtype};
}

std::vector<paddle::Tensor> ReshapeAndCacheQuantKernel(
    const paddle::Tensor& key,
    const paddle::Tensor& value,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    const float k_scale = 1.0f,
    const float k_zero = 0.0f,
    const float v_scale = 1.0f,
    const float v_zero = 0.0f) {
  PADDLE_GCU_KERNEL_TRACE("reshape_and_cache_quant_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: reshape_and_cache_quant_gcu";

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(key.place()));

  // =============================================================================
  //
  // Input key_cache:
  // shape :  [num_blocks, num_kv_heads, head_size/x, block_size, x]
  // strides: [num_kv_heads * head_size * block_size, head_size * block_size,
  // block_size * x, x, 1]
  //
  // Output key_cache:
  // shape :  [num_blocks, num_kv_heads, head_size/x, block_size, x]
  // strides: [num_kv_heads * head_size * block_size, head_size * block_size, x,
  // head_size, 1]
  //
  // =============================================================================
  //
  // Input value_cache:
  // shape :  [num_blocks, num_kv_heads, head_size, block_size]
  // strides: [num_kv_heads * head_size * block_size, head_size * block_size,
  // block_size, 1]
  //
  // Output value_cache:
  // shape :  [num_blocks, num_kv_heads, head_size, block_size]
  // strides: [num_kv_heads * head_size * block_size, head_size * block_size, 1,
  // head_size]
  //
  // =============================================================================

  // [num_tokens, num_kv_heads, head_size]
  auto key_tensor = static_cast<const phi::DenseTensor*>(key.impl().get());
  // [num_tokens, num_kv_heads, head_size]
  auto value_tensor = static_cast<const phi::DenseTensor*>(value.impl().get());
  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  auto key_cache_tensor =
      static_cast<const phi::DenseTensor*>(key_cache.impl().get());
  // [num_blocks, num_kv_heads, head_size, block_size]
  auto value_cache_tensor =
      static_cast<const phi::DenseTensor*>(value_cache.impl().get());
  // [num_tokens]
  auto slot_mapping_tensor =
      static_cast<const phi::DenseTensor*>(slot_mapping.impl().get());

  VLOG(3) << "ReshapeAndCacheQuantKernel key shape: [" << key_tensor->dims()
          << "], value shape: [" << value_tensor->dims() << "].";
  VLOG(3) << "ReshapeAndCacheQuantKernel input key_cache shape: ["
          << key_cache_tensor->dims() << "], strides: ["
          << key_cache_tensor->strides() << "].";
  VLOG(3) << "ReshapeAndCacheQuantKernel input value_cache shape: ["
          << value_cache_tensor->dims() << "], strides: ["
          << value_cache_tensor->strides() << "].";

  // key_cache_out
  std::shared_ptr<phi::DenseTensor> key_cache_out =
      std::make_shared<phi::DenseTensor>(*key_cache_tensor);
  // value_cache_out
  std::shared_ptr<phi::DenseTensor> value_cache_out =
      std::make_shared<phi::DenseTensor>(*value_cache_tensor);

  const char* kv_dtype = kv_cache_dtype.c_str();

  auto key_aten = custom_kernel::CreateTopsatenTensor(*key_tensor);
  auto value_aten = custom_kernel::CreateTopsatenTensor(*value_tensor);
  auto key_cache_aten = custom_kernel::CreateTopsatenTensor(*key_cache_tensor);
  auto value_cache_aten =
      custom_kernel::CreateTopsatenTensor(*value_cache_tensor);
  auto slot_mapping_aten =
      custom_kernel::CreateTopsatenTensor(*slot_mapping_tensor);

  auto stream = static_cast<topsStream_t>(dev_ctx->stream());
  auto status = topsvllm::topsvllmReshapeAndCacheQuant(key_cache_aten,
                                                       value_cache_aten,
                                                       key_aten,
                                                       value_aten,
                                                       slot_mapping_aten,
                                                       kv_dtype,
                                                       k_scale,
                                                       k_zero,
                                                       v_scale,
                                                       v_zero,
                                                       stream);
  PADDLE_ENFORCE_EQ(status,
                    TOPSATEN_STATUS_SUCCESS,
                    phi::errors::Fatal(
                        "Failed to call aten op "
                        "topsvllm::topsvllmReshapeAndCacheQuant, get error: %d",
                        status));

  //   topsatenSize_t key_cache_shape = key_cache_aten.GetTensorShape();
  //   std::vector<int64_t> key_cache_out_shape(
  //       key_cache_shape.data, key_cache_shape.data + key_cache_shape.len);

  //   topsatenSize_t key_cache_strides = key_cache_aten.GetTensorStrides();
  //   std::vector<int64_t> key_cache_out_strides(
  //       key_cache_strides.data, key_cache_strides.data +
  //       key_cache_strides.len);

  //   topsatenSize_t value_cache_shape = value_cache_aten.GetTensorShape();
  //   std::vector<int64_t> value_cache_out_shape(
  //       value_cache_shape.data, value_cache_shape.data +
  //       value_cache_shape.len);

  //   topsatenSize_t value_cache_strides = value_cache_aten.GetTensorStrides();
  //   std::vector<int64_t> value_cache_out_strides(
  //       value_cache_strides.data,
  //       value_cache_strides.data + value_cache_strides.len);

  //   auto key_cache_meta = key_cache_out->meta();
  //   key_cache_meta.dims = common::make_ddim(key_cache_out_shape);
  //   key_cache_meta.strides = common::make_ddim(key_cache_out_strides);
  //   key_cache_out->set_meta(key_cache_meta);

  //   auto value_cache_meta = value_cache_out->meta();
  //   value_cache_meta.dims = common::make_ddim(value_cache_out_shape);
  //   value_cache_meta.strides = common::make_ddim(value_cache_out_strides);
  //   value_cache_out->set_meta(value_cache_meta);

  //   VLOG(3) << "ReshapeAndCacheQuantKernel output key_cache shape: ["
  //           << key_cache_out->dims() << "], strides: ["
  //           << key_cache_out->strides() << "].";
  //   VLOG(3) << "ReshapeAndCacheQuantKernel output value_cache shape: ["
  //           << value_cache_out->dims() << "], strides: ["
  //           << value_cache_out->strides() << "].";

  return {paddle::Tensor(key_cache_out), paddle::Tensor(value_cache_out)};
}

PD_BUILD_OP(reshape_and_cache_gcu)
    .Inputs({"key", "value", "key_cache", "value_cache", "slot_mapping"})
    .Outputs({"key_cache_out", "value_cache_out"})
    .Attrs({
        "kv_cache_dtype: std::string",
        "k_scale: float",
        "k_zero: float",
        "v_scale: float",
        "v_zero: float",
    })
    .SetKernelFn(PD_KERNEL(ReshapeAndCacheQuantKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(ReshapeAndCacheQuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ReshapeAndCacheQuantInferDtype));

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

std::vector<std::vector<int64_t>> MoeAlignBlockSizeInferShape(
    std::vector<int64_t> sorted_token_ids,
    std::vector<int64_t> experts_ids,
    std::vector<int64_t> num_tokens_post_pad,
    std::vector<int64_t> topk_ids,
    int num_experts,
    int block_size) {
  return {sorted_token_ids, experts_ids, num_tokens_post_pad};
}

std::vector<paddle::DataType> MoeAlignBlockSizeInferDtype(
    const paddle::DataType& sorted_token_ids,
    const paddle::DataType& experts_ids,
    const paddle::DataType& num_tokens_post_pad,
    const paddle::DataType& topk_ids,
    int num_experts,
    int block_size) {
  return {sorted_token_ids, experts_ids, num_tokens_post_pad};
}

std::vector<paddle::Tensor> MoeAlignBlockSizeKernel(
    const paddle::Tensor& sorted_token_ids,
    const paddle::Tensor& experts_ids,
    const paddle::Tensor& num_tokens_post_pad,
    const paddle::Tensor& topk_ids,
    int num_experts,
    int block_size) {
  PADDLE_GCU_KERNEL_TRACE("moe_align_block_size_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: moe_align_block_size_gcu";
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          topk_ids.place()));

  auto sorted_token_ids_tensor =
      static_cast<const phi::DenseTensor*>(sorted_token_ids.impl().get());
  auto experts_ids_tensor =
      static_cast<const phi::DenseTensor*>(experts_ids.impl().get());
  auto num_tokens_post_pad_tensor =
      static_cast<const phi::DenseTensor*>(num_tokens_post_pad.impl().get());

  auto topk_ids_tensor =
      static_cast<const phi::DenseTensor*>(topk_ids.impl().get());

  auto op_info = [&]() -> std::string {
    return custom_kernel::GetOpInfo(
        "topsvllmMoeAlignBlockSize",
        sorted_token_ids_tensor,
        experts_ids_tensor,
        num_tokens_post_pad_tensor,
        topk_ids_tensor,
        num_experts,
        block_size,
        static_cast<topsStream_t>(dev_ctx->stream()));
  };

  VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();
  std::string abstract_info =
      custom_kernel::GetAbstractInfo("topsvllmMoeAlignBlockSize",
                                     sorted_token_ids_tensor,
                                     experts_ids_tensor,
                                     num_tokens_post_pad_tensor,
                                     topk_ids_tensor,
                                     num_experts,
                                     block_size);

  // sorted_token_ids_out
  std::shared_ptr<phi::DenseTensor> sorted_token_ids_out =
      std::make_shared<phi::DenseTensor>(*sorted_token_ids_tensor);
  // experts_ids_out
  std::shared_ptr<phi::DenseTensor> experts_ids_out =
      std::make_shared<phi::DenseTensor>(*experts_ids_tensor);
  // num_tokens_post_pad_out
  std::shared_ptr<phi::DenseTensor> num_tokens_post_pad_out =
      std::make_shared<phi::DenseTensor>(*num_tokens_post_pad_tensor);

  auto sorted_token_ids_tensor_aten =
      custom_kernel::CreateTopsatenTensor(*sorted_token_ids_tensor);
  auto experts_ids_tensor_aten =
      custom_kernel::CreateTopsatenTensor(*experts_ids_tensor);
  auto num_tokens_post_pad_tensor_aten =
      custom_kernel::CreateTopsatenTensor(*num_tokens_post_pad_tensor);
  auto topk_ids_tensor_aten =
      custom_kernel::CreateTopsatenTensor(*topk_ids_tensor);

  GCU_AOT_KERNEL_TRACE(abstract_info);
  auto stream = static_cast<topsStream_t>(dev_ctx->stream());
  auto status =
      topsvllm::topsvllmMoeAlignBlockSize(sorted_token_ids_tensor_aten,
                                          experts_ids_tensor_aten,
                                          num_tokens_post_pad_tensor_aten,
                                          topk_ids_tensor_aten,
                                          num_experts,
                                          block_size,
                                          stream);
  custom_kernel::GcuOpMaybeStreamSync(*dev_ctx);
  PADDLE_ENFORCE_EQ(
      status,
      TOPSATEN_STATUS_SUCCESS,
      phi::errors::Fatal("Failed to call aten op, get error: %d, details: %s",
                         status,
                         op_info().c_str()));

  VLOG(6) << "Launch tops aten op successfully, details:" << op_info();

  return {paddle::Tensor(sorted_token_ids_out),
          paddle::Tensor(experts_ids_out),
          paddle::Tensor(num_tokens_post_pad_out)};
}

PD_BUILD_OP(moe_align_block_size_gcu)
    .Inputs(
        {"sorted_token_ids", "experts_ids", "num_tokens_post_pad", "topk_ids"})
    .Outputs({"sorted_token_ids_out",
              "experts_ids_out",
              "num_tokens_post_pad_out"})
    .Attrs({
        "num_experts: int",
        "block_size: int",
    })
    .SetKernelFn(PD_KERNEL(MoeAlignBlockSizeKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeAlignBlockSizeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeAlignBlockSizeInferDtype));

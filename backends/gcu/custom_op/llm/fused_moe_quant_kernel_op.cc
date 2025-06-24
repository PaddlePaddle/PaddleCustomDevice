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

std::vector<std::vector<int64_t>> FusedMoeQuantInferShape(
    std::vector<int64_t> C_shape,
    std::vector<int64_t> A_shape,
    std::vector<int64_t> B_shape) {
  return {C_shape};
}

std::vector<paddle::DataType> FusedMoeQuantInferDtype(
    const paddle::DataType& C_dtype,
    const paddle::DataType& A_dtype,
    const paddle::DataType& B_dtype) {
  return {C_dtype};
}

std::vector<paddle::Tensor> FusedMoeQuantKernel(
    const paddle::Tensor& C,
    const paddle::Tensor& A,
    const paddle::Tensor& B,
    const paddle::optional<paddle::Tensor>& A_scale,
    const paddle::Tensor& B_scale,
    const paddle::optional<paddle::Tensor>& B_zp,
    const paddle::Tensor& topk_weights,
    const paddle::Tensor& topk_ids,
    const paddle::Tensor& sorted_token_ids,
    const paddle::Tensor& experts_ids,
    const paddle::Tensor& num_tokens_post_pad,
    int64_t gs,
    bool mul_routed_weight,
    int64_t topk,
    int64_t block_size) {
  PADDLE_GCU_KERNEL_TRACE("fused_moe_quant_kernel_kernel");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: fused_moe_quant_kernel_kernel";
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(A.place()));

  auto c_tensor = static_cast<const phi::DenseTensor*>(C.impl().get());
  auto a_tensor = static_cast<const phi::DenseTensor*>(A.impl().get());
  auto b_tensor = static_cast<const phi::DenseTensor*>(B.impl().get());

  phi::DenseTensor tmp;
  phi::DenseTensor* A_scale_tensor = &tmp;
  if (A_scale.is_initialized()) {
    A_scale_tensor = static_cast<phi::DenseTensor*>(A_scale.get().impl().get());
  }

  phi::DenseTensor* B_zp_tensor = &tmp;
  if (B_zp.is_initialized()) {
    B_zp_tensor = static_cast<phi::DenseTensor*>(B_zp.get().impl().get());
  }

  auto B_scale_tensor =
      static_cast<const phi::DenseTensor*>(B_scale.impl().get());
  auto topk_weights_tensor =
      static_cast<const phi::DenseTensor*>(topk_weights.impl().get());
  auto topk_ids_tensor =
      static_cast<const phi::DenseTensor*>(topk_ids.impl().get());
  auto sorted_token_ids_tensor =
      static_cast<const phi::DenseTensor*>(sorted_token_ids.impl().get());
  auto experts_ids_tensor =
      static_cast<const phi::DenseTensor*>(experts_ids.impl().get());
  auto num_tokens_post_pad_tensor =
      static_cast<const phi::DenseTensor*>(num_tokens_post_pad.impl().get());

  // C_out
  std::shared_ptr<phi::DenseTensor> C_out =
      std::make_shared<phi::DenseTensor>(*c_tensor);

  auto c_tensor_aten = custom_kernel::CreateTopsatenTensor(*c_tensor);
  auto a_tensor_aten = custom_kernel::CreateTopsatenTensor(*a_tensor);
  auto b_tensor_aten = custom_kernel::CreateTopsatenTensor(*b_tensor);
  auto A_scale_tensor_aten =
      custom_kernel::CreateTopsatenTensor(*A_scale_tensor);
  auto B_zp_tensor_aten = custom_kernel::CreateTopsatenTensor(*B_zp_tensor);

  auto B_scale_tensor_aten =
      custom_kernel::CreateTopsatenTensor(*B_scale_tensor);
  auto topk_weights_tensor_aten =
      custom_kernel::CreateTopsatenTensor(*topk_weights_tensor);
  auto topk_ids_tensor_aten =
      custom_kernel::CreateTopsatenTensor(*topk_ids_tensor);
  auto sorted_token_ids_tensor_aten =
      custom_kernel::CreateTopsatenTensor(*sorted_token_ids_tensor);
  auto experts_ids_tensor_aten =
      custom_kernel::CreateTopsatenTensor(*experts_ids_tensor);
  auto num_tokens_post_pad_tensor_aten =
      custom_kernel::CreateTopsatenTensor(*num_tokens_post_pad_tensor);

  auto stream = static_cast<topsStream_t>(dev_ctx->stream());
  std::string op_info;
  std::string abstract_info;
  topsatenStatus_t status;
  if (A_scale.is_initialized()) {  // w8a8
    auto op_info = [&]() -> std::string {
      return custom_kernel::GetOpInfo(
          "topsvllmInvokeFusedMoeNonGatherQuantKernel",
          c_tensor,
          a_tensor,
          b_tensor,
          A_scale_tensor,
          B_scale_tensor,
          topk_weights_tensor,
          topk_ids_tensor,
          sorted_token_ids_tensor,
          experts_ids_tensor,
          num_tokens_post_pad_tensor,
          mul_routed_weight,
          topk,
          block_size,
          stream);
    };

    abstract_info = custom_kernel::GetAbstractInfo(
        "topsvllmInvokeFusedMoeNonGatherQuantKernel",
        *c_tensor,
        *a_tensor,
        *b_tensor,
        *A_scale_tensor,
        *B_scale_tensor,
        *topk_weights_tensor,
        *topk_ids_tensor,
        *sorted_token_ids_tensor,
        *experts_ids_tensor,
        *num_tokens_post_pad_tensor,
        mul_routed_weight,
        topk,
        block_size);
    VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();
    GCU_AOT_KERNEL_TRACE(abstract_info);
    status = topsvllm::topsvllmInvokeFusedMoeNonGatherQuantKernel(
        c_tensor_aten,
        a_tensor_aten,
        b_tensor_aten,
        A_scale_tensor_aten,
        B_scale_tensor_aten,
        topk_weights_tensor_aten,
        topk_ids_tensor_aten,
        sorted_token_ids_tensor_aten,
        experts_ids_tensor_aten,
        num_tokens_post_pad_tensor_aten,
        mul_routed_weight,
        topk,
        block_size,
        stream);
  } else {
    auto op_info = [&]() -> std::string {
      return custom_kernel::GetOpInfo(
          "topsvllmInvokeFusedMoeNonGatherQuantKernel",
          c_tensor,
          a_tensor,
          b_tensor,
          B_scale_tensor,
          gs,
          B_zp_tensor,
          topk_weights_tensor,
          topk_ids_tensor,
          sorted_token_ids_tensor,
          experts_ids_tensor,
          num_tokens_post_pad_tensor,
          mul_routed_weight,
          topk,
          block_size,
          stream);
    };

    abstract_info = custom_kernel::GetAbstractInfo(
        "topsvllmInvokeFusedMoeNonGatherQuantKernel",
        *c_tensor,
        *a_tensor,
        *b_tensor,
        *B_scale_tensor,
        gs,
        *B_zp_tensor,
        *topk_weights_tensor,
        *topk_ids_tensor,
        *sorted_token_ids_tensor,
        *experts_ids_tensor,
        *num_tokens_post_pad_tensor,
        mul_routed_weight,
        topk,
        block_size);
    VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();
    GCU_AOT_KERNEL_TRACE(abstract_info);
    status = topsvllm::topsvllmInvokeFusedMoeNonGatherQuantKernel(
        c_tensor_aten,
        a_tensor_aten,
        b_tensor_aten,
        B_scale_tensor_aten,
        gs,
        B_zp_tensor_aten,
        topk_weights_tensor_aten,
        topk_ids_tensor_aten,
        sorted_token_ids_tensor_aten,
        experts_ids_tensor_aten,
        num_tokens_post_pad_tensor_aten,
        mul_routed_weight,
        topk,
        block_size,
        stream);
  }

  custom_kernel::GcuOpMaybeStreamSync(*dev_ctx);
  PADDLE_ENFORCE_EQ(
      status,
      TOPSATEN_STATUS_SUCCESS,
      phi::errors::Fatal("Failed to call aten op, get error: %d, details: %s",
                         status,
                         op_info.c_str()));

  VLOG(6) << "Launch tops aten op successfully, details:" << op_info;
  return {paddle::Tensor(C_out)};
}

PD_BUILD_OP(fused_moe_quant_kernel)
    .Inputs({
        "C",
        "A",
        "B",
        paddle::Optional("A_scale"),
        "B_scale",
        paddle::Optional("B_zp"),
        "topk_weights",
        "topk_ids",
        "sorted_token_ids",
        "experts_ids",
        "num_tokens_post_pad",
    })
    .Outputs({"C_out"})
    .Attrs({
        "gs: int64_t",
        "mul_routed_weight: bool",
        "topk: int64_t",
        "block_size: int64_t",
    })
    .SetKernelFn(PD_KERNEL(FusedMoeQuantKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedMoeQuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedMoeQuantInferDtype));

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

inline bool UseLegacy() {
  static const char* profiler_env =
      std::getenv("PADDLE_GCU_FUSED_MOE_USE_LEGACY");
  static bool use_legacy =
      (profiler_env != nullptr && (std::string(profiler_env) == "true" ||
                                   std::string(profiler_env) == "1"));
  return use_legacy;
}

std::vector<std::vector<int64_t>> FusedMoeInferShape(
    std::vector<int64_t> C_shape,
    std::vector<int64_t> A_shape,
    std::vector<int64_t> B_shape) {
  return {C_shape};
}

std::vector<paddle::DataType> FusedMoeInferDtype(
    const paddle::DataType& C_dtype,
    const paddle::DataType& A_dtype,
    const paddle::DataType& B_dtype) {
  return {C_dtype};
}

std::vector<paddle::Tensor> FusedMoeKernel(
    const paddle::Tensor& C,
    const paddle::Tensor& A,
    const paddle::Tensor& B,
    const paddle::optional<paddle::Tensor>& bias,
    const paddle::Tensor& topk_weights,
    const paddle::Tensor& topk_ids,
    const paddle::Tensor& sorted_token_ids,
    const paddle::Tensor& experts_ids,
    const paddle::Tensor& num_tokens_post_pad,
    const bool mul_routed_weight,
    const int64_t topk,
    const int64_t block_size) {
  PADDLE_GCU_KERNEL_TRACE("invoke_fused_moe_biased_kernel");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: invoke_fused_moe_biased_kernel";
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(A.place()));

  auto c_tensor = static_cast<const phi::DenseTensor*>(C.impl().get());
  auto a_tensor = static_cast<const phi::DenseTensor*>(A.impl().get());
  auto b_tensor = static_cast<const phi::DenseTensor*>(B.impl().get());

  phi::DenseTensor tmp;
  phi::DenseTensor* bias_tensor = &tmp;
  if (bias.is_initialized()) {
    bias_tensor = static_cast<phi::DenseTensor*>(bias.get().impl().get());
  }

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
  auto bias_tensor_aten = custom_kernel::CreateTopsatenTensor(*bias_tensor);
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
  auto use_legacy = UseLegacy();
  if (use_legacy) {
    if (bias.is_initialized()) {
      auto op_info = [&]() -> std::string {
        return custom_kernel::GetOpInfo(
            "topsvllmInvokeFusedMoeKernel",
            c_tensor,
            a_tensor,
            b_tensor,
            bias_tensor,
            topk_weights_tensor,
            topk_ids_tensor,
            sorted_token_ids_tensor,
            experts_ids_tensor,
            num_tokens_post_pad_tensor,
            mul_routed_weight,
            topk,
            block_size,
            static_cast<topsStream_t>(dev_ctx->stream()));
      };

      VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();

      std::string abstract_info =
          custom_kernel::GetAbstractInfo("topsvllmInvokeFusedMoeKernel",
                                         *c_tensor,
                                         *a_tensor,
                                         *b_tensor,
                                         *bias_tensor,
                                         *topk_weights_tensor,
                                         *topk_ids_tensor,
                                         *sorted_token_ids_tensor,
                                         *experts_ids_tensor,
                                         *num_tokens_post_pad_tensor,
                                         mul_routed_weight,
                                         topk,
                                         block_size);

      GCU_AOT_KERNEL_TRACE(abstract_info);

      status = topsvllm::topsvllmInvokeFusedMoeKernel(
          c_tensor_aten,
          a_tensor_aten,
          b_tensor_aten,
          bias_tensor_aten,
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
            "topsvllmInvokeFusedMoeKernel",
            c_tensor,
            a_tensor,
            b_tensor,
            topk_weights_tensor,
            topk_ids_tensor,
            sorted_token_ids_tensor,
            experts_ids_tensor,
            num_tokens_post_pad_tensor,
            mul_routed_weight,
            topk,
            block_size,
            static_cast<topsStream_t>(dev_ctx->stream()));
      };

      VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();

      std::string abstract_info =
          custom_kernel::GetAbstractInfo("topsvllmInvokeFusedMoeKernel",
                                         *c_tensor,
                                         *a_tensor,
                                         *b_tensor,
                                         *topk_weights_tensor,
                                         *topk_ids_tensor,
                                         *sorted_token_ids_tensor,
                                         *experts_ids_tensor,
                                         *num_tokens_post_pad_tensor,
                                         mul_routed_weight,
                                         topk,
                                         block_size);

      GCU_AOT_KERNEL_TRACE(abstract_info);

      status = topsvllm::topsvllmInvokeFusedMoeKernel(
          c_tensor_aten,
          a_tensor_aten,
          b_tensor_aten,
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
  } else {
    if (bias.is_initialized()) {
      auto op_info = [&]() -> std::string {
        return custom_kernel::GetOpInfo(
            "topsvllmInvokeFusedMoeNonGatherKernel",
            c_tensor,
            a_tensor,
            b_tensor,
            bias_tensor,
            topk_weights_tensor,
            topk_ids_tensor,
            sorted_token_ids_tensor,
            experts_ids_tensor,
            num_tokens_post_pad_tensor,
            mul_routed_weight,
            topk,
            block_size,
            static_cast<topsStream_t>(dev_ctx->stream()));
      };

      VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();

      std::string abstract_info = custom_kernel::GetAbstractInfo(
          "topsvllmInvokeFusedMoeNonGatherKernel",
          *c_tensor,
          *a_tensor,
          *b_tensor,
          *bias_tensor,
          *topk_weights_tensor,
          *topk_ids_tensor,
          *sorted_token_ids_tensor,
          *experts_ids_tensor,
          *num_tokens_post_pad_tensor,
          mul_routed_weight,
          topk,
          block_size);

      GCU_AOT_KERNEL_TRACE(abstract_info);

      status = topsvllm::topsvllmInvokeFusedMoeNonGatherKernel(
          c_tensor_aten,
          a_tensor_aten,
          b_tensor_aten,
          bias_tensor_aten,
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
            "topsvllmInvokeFusedMoeNonGatherKernel",
            c_tensor,
            a_tensor,
            b_tensor,
            bias_tensor,
            topk_weights_tensor,
            topk_ids_tensor,
            sorted_token_ids_tensor,
            experts_ids_tensor,
            num_tokens_post_pad_tensor,
            mul_routed_weight,
            topk,
            block_size,
            static_cast<topsStream_t>(dev_ctx->stream()));
      };

      VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();

      std::string abstract_info = custom_kernel::GetAbstractInfo(
          "topsvllmInvokeFusedMoeNonGatherKernel",
          *c_tensor,
          *a_tensor,
          *b_tensor,
          *topk_weights_tensor,
          *topk_ids_tensor,
          *sorted_token_ids_tensor,
          *experts_ids_tensor,
          *num_tokens_post_pad_tensor,
          mul_routed_weight,
          topk,
          block_size);

      GCU_AOT_KERNEL_TRACE(abstract_info);

      status = topsvllm::topsvllmInvokeFusedMoeNonGatherKernel(
          c_tensor_aten,
          a_tensor_aten,
          b_tensor_aten,
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

PD_BUILD_OP(fused_moe_kernel_gcu)
    .Inputs({"C",
             "A",
             "B",
             paddle::Optional("bias"),
             "topk_weights",
             "topk_ids",
             "sorted_token_ids",
             "experts_ids",
             "num_tokens_post_pad"})
    .Outputs({"C_out"})
    .Attrs({
        "mul_routed_weight: bool",
        "topk: int64_t",
        "block_size: int64_t",
    })
    .SetKernelFn(PD_KERNEL(FusedMoeKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedMoeInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedMoeInferDtype));

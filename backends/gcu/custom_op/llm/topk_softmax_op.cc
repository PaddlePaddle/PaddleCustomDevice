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

std::vector<std::vector<int64_t>> TopkSoftmaxInferShape(
    std::vector<int64_t> topk_weights_shape,
    std::vector<int64_t> topk_indices_shape,
    std::vector<int64_t> token_expert_indices_shape,
    std::vector<int64_t> gating_output_shape) {
  return {topk_weights_shape, topk_indices_shape, token_expert_indices_shape};
}

std::vector<paddle::DataType> TopkSoftmaxInferDtype(
    const paddle::DataType& topk_weights_dtype,
    const paddle::DataType& topk_indices_dtype,
    const paddle::DataType& token_expert_indices_dtype,
    const paddle::DataType& gating_output_dtype) {
  return {topk_weights_dtype, topk_indices_dtype, token_expert_indices_dtype};
}

std::vector<paddle::Tensor> TopkSoftmaxKernel(
    const paddle::Tensor& topk_weights,
    const paddle::Tensor& topk_indices,
    const paddle::Tensor& token_expert_indices,
    const paddle::Tensor& gating_output,
    const bool norm_topk_prob = false) {
  PADDLE_GCU_KERNEL_TRACE("topk_softmax_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: topk_softmax_gcu";
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          gating_output.place()));

  // [num_tokens, topk]
  auto topk_weights_tensor =
      static_cast<const phi::DenseTensor*>(topk_weights.impl().get());
  // [num_tokens, topk]
  auto topk_indices_tensor =
      static_cast<const phi::DenseTensor*>(topk_indices.impl().get());
  // [num_tokens, topk]
  auto token_expert_indices_tensor =
      static_cast<const phi::DenseTensor*>(token_expert_indices.impl().get());
  // [num_tokens, num_experts]
  auto gating_output_tensor =
      static_cast<const phi::DenseTensor*>(gating_output.impl().get());

  auto op_info = [&]() -> std::string {
    return custom_kernel::GetOpInfo(
        "topsvllmTopkSoftmax",
        *topk_weights_tensor,
        *topk_indices_tensor,
        *token_expert_indices_tensor,
        *gating_output_tensor,
        norm_topk_prob,
        static_cast<topsStream_t>(dev_ctx->stream()));
  };

  VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();
  std::string abstract_info =
      custom_kernel::GetAbstractInfo("topsvllmTopkSoftmax",
                                     *topk_weights_tensor,
                                     *topk_indices_tensor,
                                     *token_expert_indices_tensor,
                                     *gating_output_tensor,
                                     norm_topk_prob);

  // topk_weights_out
  std::shared_ptr<phi::DenseTensor> topk_weights_out =
      std::make_shared<phi::DenseTensor>(*topk_weights_tensor);
  // topk_indices_out
  std::shared_ptr<phi::DenseTensor> topk_indices_out =
      std::make_shared<phi::DenseTensor>(*topk_indices_tensor);
  // token_expert_indices_out
  std::shared_ptr<phi::DenseTensor> token_expert_indices_out =
      std::make_shared<phi::DenseTensor>(*token_expert_indices_tensor);

  auto topk_weights_aten =
      custom_kernel::CreateTopsatenTensor(*topk_weights_tensor);
  auto topk_indices_aten =
      custom_kernel::CreateTopsatenTensor(*topk_indices_tensor);
  auto token_expert_indices_aten =
      custom_kernel::CreateTopsatenTensor(*token_expert_indices_tensor);
  auto gating_output_aten =
      custom_kernel::CreateTopsatenTensor(*gating_output_tensor);

  GCU_AOT_KERNEL_TRACE(abstract_info);
  auto stream = static_cast<topsStream_t>(dev_ctx->stream());
  auto status = topsvllm::topsvllmTopkSoftmax(topk_weights_aten,
                                              topk_indices_aten,
                                              token_expert_indices_aten,
                                              gating_output_aten,
                                              norm_topk_prob,
                                              stream);
  custom_kernel::GcuOpMaybeStreamSync(*dev_ctx);
  PADDLE_ENFORCE_EQ(
      status,
      TOPSATEN_STATUS_SUCCESS,
      phi::errors::Fatal("Failed to call aten op, get error: %d, details: %s",
                         status,
                         op_info().c_str()));
  VLOG(6) << "Launch tops aten op successfully, details:" << op_info();

  return {paddle::Tensor(topk_weights_out),
          paddle::Tensor(topk_indices_out),
          paddle::Tensor(token_expert_indices_out)};
}

PD_BUILD_OP(topk_softmax_gcu)
    .Inputs({"topk_weights",
             "topk_indices",
             "token_expert_indices",
             "gating_output"})
    .Outputs({"topk_weights_out",
              "topk_indices_out",
              "token_expert_indices_out"})
    .Attrs({
        "norm_topk_prob: bool",
    })
    .SetKernelFn(PD_KERNEL(TopkSoftmaxKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(TopkSoftmaxInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopkSoftmaxInferDtype));

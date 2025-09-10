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

std::vector<std::vector<int64_t>> TopPSamplingOpInferShape(
    std::vector<int64_t> probs_shape, std::vector<int64_t> top_p_shape) {
  int64_t bs = probs_shape[0];
  std::vector<int64_t> ret_shape = {bs, 1};
  return {ret_shape, ret_shape};
}

std::vector<paddle::DataType> TopPSamplingOpInferDtype(
    const paddle::DataType& probs_dtype, const paddle::DataType& top_p_dtype) {
  return {probs_dtype, probs_dtype};
}

std::vector<paddle::Tensor> TopPSamplingOpKernel(const paddle::Tensor& probs,
                                                 const paddle::Tensor& top_p) {
  PADDLE_GCU_KERNEL_TRACE("top_p_sampling_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: top_p_sampling_gcu";
  return custom_op_common::TopPSampling(probs, top_p);
}

PD_BUILD_OP(top_p_sampling_gcu)
    .Inputs({"probs", "top_p"})
    .Outputs({"next_scores", "next_tokens"})
    .SetKernelFn(PD_KERNEL(TopPSamplingOpKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(TopPSamplingOpInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopPSamplingOpInferDtype));

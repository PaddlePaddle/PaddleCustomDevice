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

#include "custom_op/custom_op_common.h"

std::vector<std::vector<int64_t>> FusedAddRmsNormInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& residual_shape,
    const std::vector<int64_t>& weight_shape,
    float epsilon) {
  return {x_shape, residual_shape};
}

std::vector<paddle::Tensor> FusedAddRmsNormKernel(
    const paddle::Tensor& x,
    const paddle::Tensor& residual,
    const paddle::Tensor& weight,
    float epsilon) {
  PADDLE_GCU_KERNEL_TRACE("fused_add_rms_norm_op");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: fused_add_rms_norm_op";
  return custom_op_common::FusedAddRmsNorm(x, residual, weight, epsilon);
}

PD_BUILD_OP(fused_add_rms_norm_op)
    .Inputs({"x", "residual", "weight"})
    .Outputs({"out", "residual_out"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(FusedAddRmsNormKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(
        FusedAddRmsNormInferShape));  // neccessary if the op has muti_inputs

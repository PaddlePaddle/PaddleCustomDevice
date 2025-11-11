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

std::vector<std::vector<int64_t>> RmsNormInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& weight_shape,
    float epsilon) {
  return {x_shape};
}

std::vector<paddle::Tensor> RmsNormKernel(const paddle::Tensor& x,
                                          const paddle::Tensor& weight,
                                          float epsilon) {
  PADDLE_GCU_KERNEL_TRACE("rms_norm_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: rms_norm_gcu";
  return custom_op_common::RmsNorm(x, weight, epsilon);
}

PD_BUILD_OP(rms_norm_gcu)
    .Inputs({"x", "weight"})
    .Outputs({"out"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(RmsNormKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(
        RmsNormInferShape));  // neccessary if the op has muti_inputs

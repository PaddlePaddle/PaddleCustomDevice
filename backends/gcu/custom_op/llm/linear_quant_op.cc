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

std::vector<std::vector<int64_t>> LinearQuantInferShape(
    std::vector<int64_t> lhs_shape, std::vector<int64_t> rhs_shape) {
  std::vector<int64_t> out_shape = lhs_shape;
  if (lhs_shape.at(lhs_shape.size() - 1) == rhs_shape.at(0)) {
    out_shape.at(lhs_shape.size() - 1) = rhs_shape.at(rhs_shape.size() - 1);
  } else {
    out_shape.at(lhs_shape.size() - 1) = rhs_shape.at(0);
  }
  return {out_shape};
}

std::vector<paddle::DataType> LinearQuantInferDtype(
    const paddle::DataType &lhs_dtype, const paddle::DataType &rhs_dtype) {
  return {lhs_dtype};
}

std::vector<paddle::Tensor> LinearQuantKernel(
    const paddle::Tensor &lhs,
    const paddle::Tensor &rhs,
    const paddle::Tensor &scale,
    const paddle::optional<paddle::Tensor> &bias,
    int group_size = -1) {
  PADDLE_GCU_KERNEL_TRACE("linear_quant_gcu");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: linear_quant_gcu";
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(lhs.place()));

  auto lhs_tensor = static_cast<const phi::DenseTensor *>(lhs.impl().get());
  auto rhs_tensor = static_cast<const phi::DenseTensor *>(rhs.impl().get());
  auto scale_tensor = static_cast<const phi::DenseTensor *>(scale.impl().get());

  phi::DenseTensor tmp;
  phi::DenseTensor *bias_tensor = &tmp;
  if (bias.is_initialized()) {
    bias_tensor = static_cast<phi::DenseTensor *>(bias.get().impl().get());
  }

  auto out_dims = lhs_tensor->dims();
  auto rhs_dims = rhs_tensor->dims();
  if (out_dims.at(out_dims.size() - 1) == rhs_dims.at(0)) {
    out_dims.at(out_dims.size() - 1) = rhs_dims.at(rhs_dims.size() - 1);
  } else {
    out_dims.at(out_dims.size() - 1) = rhs_dims.at(0);
  }

  // linear_out
  std::shared_ptr<phi::DenseTensor> linear_out =
      std::make_shared<phi::DenseTensor>();
  linear_out->Resize(out_dims);
  dev_ctx->Alloc(linear_out.get(), lhs_tensor->dtype());

  LAUNCH_TOPSATENOP(topsatenLinearQuant,
                    (*dev_ctx),
                    *linear_out,
                    *lhs_tensor,
                    *rhs_tensor,
                    *bias_tensor,
                    *scale_tensor,
                    group_size);

  return {paddle::Tensor(linear_out)};
}

PD_BUILD_OP(linear_quant_gcu)
    .Inputs({"lhs", "rhs", "scale", paddle::Optional("bias")})
    .Outputs({"linear_out"})
    .Attrs({
        "group_size: int",
    })
    .SetKernelFn(PD_KERNEL(LinearQuantKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(LinearQuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LinearQuantInferDtype));

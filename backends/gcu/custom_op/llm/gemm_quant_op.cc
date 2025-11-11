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

std::vector<std::vector<int64_t>> AwqGemmInferShape(
    std::vector<int64_t> lhs_shape, std::vector<int64_t> rhs_shape) {
  lhs_shape.at(lhs_shape.size() - 1) = rhs_shape.at(rhs_shape.size() - 1);
  return {lhs_shape};
}

std::vector<paddle::DataType> AwqGemmInferDtype(
    const paddle::DataType &lhs_dtype, const paddle::DataType &rhs_dtype) {
  return {lhs_dtype};
}

std::vector<paddle::Tensor> AwqGemmKernel(
    const paddle::Tensor &lhs,
    const paddle::Tensor &rhs,
    const paddle::Tensor &scale,
    const paddle::Tensor &zeros,
    const paddle::optional<paddle::Tensor> &bias,
    int group_size) {
  PADDLE_GCU_KERNEL_TRACE("awq_gemm_kernel");
  VLOG(6) << "[CUSTOM_KERNEL] Custom Operator: awq_gemm_kernel";
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(lhs.place()));

  auto lhs_tensor = static_cast<const phi::DenseTensor *>(lhs.impl().get());
  auto rhs_tensor = static_cast<const phi::DenseTensor *>(rhs.impl().get());
  auto scale_tensor = static_cast<const phi::DenseTensor *>(scale.impl().get());
  auto zeros_tensor = static_cast<const phi::DenseTensor *>(zeros.impl().get());

  phi::DenseTensor tmp;
  phi::DenseTensor *bias_tensor = &tmp;
  if (bias.is_initialized()) {
    bias_tensor = static_cast<phi::DenseTensor *>(bias.get().impl().get());
  }

  auto out_dims = lhs_tensor->dims();
  auto rhs_dims = rhs_tensor->dims();
  out_dims.at(out_dims.size() - 1) = rhs_dims.at(rhs_dims.size() - 1);
  // linear_out
  std::shared_ptr<phi::DenseTensor> linear_out =
      std::make_shared<phi::DenseTensor>();
  linear_out->Resize(out_dims);
  dev_ctx->Alloc(linear_out.get(), lhs_tensor->dtype());

  auto stream = static_cast<topsStream_t>(dev_ctx->stream());

  auto op_info = [&]() -> std::string {
    return custom_kernel::GetOpInfo(
        "topsatenGemmQuant",
        linear_out,
        lhs_tensor,
        rhs_tensor,
        scale_tensor,
        zeros_tensor,
        group_size,
        bias_tensor,
        static_cast<topsStream_t>(dev_ctx->stream()));
  };

  VLOG(6) << "[AOT_KERNEL] Start to launch tops aten op, " << op_info();
  std::string abstract_info =
      custom_kernel::GetAbstractInfo("topsatenGemmQuant",
                                     linear_out,
                                     lhs_tensor,
                                     rhs_tensor,
                                     scale_tensor,
                                     zeros_tensor,
                                     bias_tensor,
                                     group_size);

  auto linear_out_aten = custom_kernel::CreateTopsatenTensor(*linear_out);
  auto lhs_tensor_aten = custom_kernel::CreateTopsatenTensor(*lhs_tensor);
  auto rhs_tensor_aten = custom_kernel::CreateTopsatenTensor(*rhs_tensor);
  auto scale_tensor_aten = custom_kernel::CreateTopsatenTensor(*scale_tensor);
  auto zeros_tensor_aten = custom_kernel::CreateTopsatenTensor(*zeros_tensor);
  auto bias_tensor_aten = custom_kernel::CreateTopsatenTensor(*bias_tensor);

  GCU_AOT_KERNEL_TRACE(abstract_info);

  auto status = topsaten::topsatenGemmQuant(linear_out_aten,
                                            lhs_tensor_aten,
                                            rhs_tensor_aten,
                                            scale_tensor_aten,
                                            zeros_tensor_aten,
                                            bias_tensor_aten,
                                            group_size,
                                            stream);
  custom_kernel::GcuOpMaybeStreamSync(*dev_ctx);
  PADDLE_ENFORCE_EQ(
      status,
      TOPSATEN_STATUS_SUCCESS,
      phi::errors::Fatal("Failed to call aten op, get error: %d, details: %s",
                         status,
                         op_info().c_str()));

  VLOG(6) << "Launch tops aten op successfully, details:" << op_info();

  return {paddle::Tensor(linear_out)};
}

PD_BUILD_OP(awq_gemm_gcu)
    .Inputs({"lhs", "rhs", "scale", "zeros", paddle::Optional("bias")})
    .Outputs({"linear_out"})
    .Attrs({
        "group_size: int",
    })
    .SetKernelFn(PD_KERNEL(AwqGemmKernel))
    .SetInferShapeFn(PD_INFER_SHAPE(AwqGemmInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AwqGemmInferDtype));

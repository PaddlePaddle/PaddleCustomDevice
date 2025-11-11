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

#include <iostream>
#include <vector>

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"

std::vector<paddle::Tensor> SetValueByFlagsAndIdx(
    const paddle::Tensor& pre_ids_all,
    const paddle::Tensor& pre_ids_now,
    const paddle::Tensor& step_idx,
    const paddle::Tensor& stop_flags) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          stop_flags.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto pre_ids_all_tensor =
      static_cast<const phi::DenseTensor*>(pre_ids_all.impl().get());
  auto pre_ids_now_tensor =
      static_cast<const phi::DenseTensor*>(pre_ids_now.impl().get());
  auto step_idx_tensor =
      static_cast<const phi::DenseTensor*>(step_idx.impl().get());
  auto stop_flags_tensor =
      static_cast<const phi::DenseTensor*>(stop_flags.impl().get());

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(stop_flags_tensor->dims());
  dev_ctx->Alloc(out_tensor.get(), stop_flags_tensor->dtype());

  const auto& runner = NpuOpRunner("SetValueByFlagsAndIdx",
                                   {*pre_ids_all_tensor,
                                    *pre_ids_now_tensor,
                                    *step_idx_tensor,
                                    *stop_flags_tensor},
                                   {*out_tensor},
                                   {});
  runner.Run(stream);

  return {paddle::Tensor(out_tensor)};
}

std::vector<std::vector<int64_t>> SetValueByFlagsAndIdxInferShape(
    const std::vector<int64_t>& pre_ids_all_shape,
    const std::vector<int64_t>& pre_ids_now_shape,
    const std::vector<int64_t>& step_idx_shape,
    const std::vector<int64_t>& stop_flags_shape) {
  return {stop_flags_shape};
}

std::vector<paddle::DataType> SetValueByFlagsAndIdxInferDtype(
    const paddle::DataType& pre_ids_all_dtype,
    const paddle::DataType& pre_ids_now_dtype,
    const paddle::DataType& step_idx_dtype,
    const paddle::DataType& stop_flags_dtype) {
  return {stop_flags_dtype};
}

PD_BUILD_OP(set_value_by_flags_and_idx)
    .Inputs({"pre_ids_all", "pre_ids_now", "step_idx", "stop_flags"})
    .Outputs({"stop_flags_out"})
    .SetKernelFn(PD_KERNEL(SetValueByFlagsAndIdx))
    .SetInferShapeFn(PD_INFER_SHAPE(SetValueByFlagsAndIdxInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SetValueByFlagsAndIdxInferDtype));

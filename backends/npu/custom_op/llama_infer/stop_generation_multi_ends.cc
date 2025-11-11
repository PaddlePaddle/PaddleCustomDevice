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

std::vector<paddle::Tensor> GetStopFlagsMulti(const paddle::Tensor& topk_ids,
                                              const paddle::Tensor& stop_flags,
                                              const paddle::Tensor& end_ids,
                                              int64_t mode) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(end_ids.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  auto end_ids_tensor =
      static_cast<const phi::DenseTensor*>(end_ids.impl().get());
  auto stop_flags_tensor =
      static_cast<const phi::DenseTensor*>(stop_flags.impl().get());
  auto topk_ids_tensor =
      static_cast<const phi::DenseTensor*>(topk_ids.impl().get());

  std::shared_ptr<phi::DenseTensor> stop_flags_out =
      std::make_shared<phi::DenseTensor>();
  stop_flags_out->Resize(stop_flags_tensor->dims());
  dev_ctx->Alloc(stop_flags_out.get(), stop_flags_tensor->dtype());

  auto topk_ids_out = topk_ids.copy_to(topk_ids.place(), false);
  auto topk_ids_out_tensor =
      static_cast<const phi::DenseTensor*>(topk_ids_out.impl().get());

  int32_t attr_mode = mode;
  NPUAttributeMap attr_input = {{"mode", attr_mode}};

  const auto& runner =
      NpuOpRunner("SetStopValueMultiEnds",
                  {*topk_ids_out_tensor, *stop_flags_tensor, *end_ids_tensor},
                  {*topk_ids_out_tensor, *stop_flags_out},
                  attr_input);
  runner.Run(stream);
  return {paddle::Tensor(topk_ids_out), paddle::Tensor(stop_flags_out)};
}

std::vector<std::vector<int64_t>> GetStopFlagsMultiInferShape(
    const std::vector<int64_t>& topk_ids_shape,
    const std::vector<int64_t>& stop_flags_shape,
    const std::vector<int64_t>& end_ids_shape) {
  return {topk_ids_shape, stop_flags_shape};
}

std::vector<paddle::DataType> GetStopFlagsMultiInferDtype(
    const paddle::DataType& topk_ids_dtype,
    const paddle::DataType& stop_flags_dtype,
    const paddle::DataType& end_ids_dtype) {
  return {topk_ids_dtype, stop_flags_dtype};
}

PD_BUILD_OP(set_stop_value_multi_ends)
    .Inputs({"topk_ids", "stop_flags", "end_ids"})
    .Outputs({"topk_ids_out", "stop_flags_out"})
    .Attrs({"mode: int64_t"})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMulti))
    .SetInferShapeFn(PD_INFER_SHAPE(GetStopFlagsMultiInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetStopFlagsMultiInferDtype));

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

const phi::DDim get_output_size_gather_mm(const paddle::Tensor& x1,
                                          const paddle::Tensor& x2,
                                          int64_t world_size,
                                          int64_t gather_index) {
  auto out_x = gather_index == 0 ? x1.dims()[0] * world_size : x1.dims()[0];
  auto out_y = x2.dims()[1];
  return {out_x, out_y};
}

const phi::DDim get_output_size_gather(const paddle::Tensor& x1,
                                       const paddle::Tensor& x2,
                                       int64_t world_size,
                                       int64_t gather_index) {
  const paddle::Tensor& gather_out = gather_index == 0 ? x1 : x2;
  return {gather_out.dims()[0] * world_size, gather_out.dims()[1]};
}

std::vector<paddle::Tensor> npu_allgather_mm(
    const paddle::Tensor& x1,
    const paddle::Tensor& x2,
    const paddle::optional<paddle::Tensor>& bias,
    std::string hcom,
    int64_t world_size,
    int64_t gather_index,
    bool gather_output,
    int64_t comm_turn) {
  PD_CHECK(
      x1.dims().size() == 2 && x2.dims().size() == 2,
      "Both inputs of mm are required to be 2D, but the actual inputs are ",
      x1.dims().size(),
      "D and ",
      x2.dims().size(),
      "D");
  PD_CHECK(x1.dims()[1] == x2.dims()[0],
           "The K-axis in the two inputs of Matmul must be equal, but in "
           "reality, the K-axis of x1 is ",
           x1.dims()[1],
           " and the K-axis of x2 is ",
           x2.dims()[0]);

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x1.place()));
  auto out_gather_mm_size =
      get_output_size_gather_mm(x1, x2, world_size, gather_index);
  auto out_gather_size =
      get_output_size_gather(x1, x2, world_size, gather_index);

  std::shared_ptr<phi::DenseTensor> out_gather_mm =
      std::make_shared<phi::DenseTensor>();
  out_gather_mm->Resize(out_gather_mm_size);
  dev_ctx->Alloc(out_gather_mm.get(), x1.dtype());

  std::shared_ptr<phi::DenseTensor> out_gather =
      std::make_shared<phi::DenseTensor>();
  if (gather_output) {
    out_gather->Resize(out_gather_size);
  } else {
    out_gather->Resize({0});
  }
  dev_ctx->Alloc(out_gather.get(), x1.dtype());

  phi::DenseTensor* bias_real = nullptr;
  if (bias) {
    auto bias_ptr = *(bias.get_ptr());
    bias_real = static_cast<phi::DenseTensor*>(bias_ptr.impl().get());
  } else {
    bias_real = new phi::DenseTensor();
  }

  auto x1_tensor = *(static_cast<const phi::DenseTensor*>(x1.impl().get()));
  auto x2_tensor = *(static_cast<const phi::DenseTensor*>(x2.impl().get()));
  char* hcom_ptr = const_cast<char*>(hcom.data());
#if (CANN_VERSION_CODE >= 700000)
  int64_t stream_mode = ACL_STOP_ON_FAILURE;
  EXEC_NPU_CMD(aclnnAllGatherMatmul,
               *dev_ctx,
               x1_tensor,
               x2_tensor,
               *bias_real,
               hcom_ptr,
               gather_index,
               comm_turn,
               stream_mode,
               *out_gather_mm,
               *out_gather);
#endif
  return {paddle::Tensor(out_gather_mm), paddle::Tensor(out_gather)};
}

std::vector<std::vector<int64_t>> FusedAllgatherMMInferShape(
    std::vector<int64_t> x1_shape, std::vector<int64_t> x2_shape) {
  return {x1_shape, x2_shape};
}

PD_BUILD_OP(fused_allgather_mm)
    .Inputs({"x1", "x2", paddle::Optional("bias")})
    .Outputs({"out_gather_mm", "out_gather"})
    .Attrs({"hcom:std::string",
            "world_size:int64_t",
            "gather_index:int64_t",
            "gather_output:bool",
            "comm_turn:int64_t"})
    .SetKernelFn(PD_KERNEL(npu_allgather_mm))
    .SetInferShapeFn(PD_INFER_SHAPE(
        FusedAllgatherMMInferShape));  // neccessary if the op has muti_inputs

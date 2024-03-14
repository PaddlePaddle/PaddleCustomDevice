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

std::vector<paddle::Tensor> npu_mm_reduce_scatter(
    const paddle::Tensor& x1,
    const paddle::Tensor& x2,
    const paddle::optional<paddle::Tensor>& bias,
    std::string hcom,
    int64_t world_size,
    std::string reduce_op,
    int64_t comm_turn) {
  PD_CHECK(world_size == 2 || world_size == 4 || world_size == 8,
           "world_size should be 2 or 4 or 8, but the actual value is ",
           world_size);
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
  PD_CHECK(
      x1.dims()[0] % world_size == 0,
      "The M-axis in input of Matmul should be be divisible by world_size");

  auto output_size = {x1.dims()[0] / world_size, x2.dims()[1]};
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(x1.place()));
  auto x1_tensor = *(static_cast<const phi::DenseTensor*>(x1.impl().get()));
  auto x2_tensor = *(static_cast<const phi::DenseTensor*>(x2.impl().get()));

  std::shared_ptr<phi::DenseTensor> result =
      std::make_shared<phi::DenseTensor>();
  result->Resize(output_size);
  dev_ctx->Alloc(result.get(), x1.dtype());

  char* reduce_op_ptr = const_cast<char*>(reduce_op.data());
  char* hcom_ptr = const_cast<char*>(hcom.data());
  phi::DenseTensor* bias_real = nullptr;
  if (bias) {
    auto bias_ptr = *(bias.get_ptr());
    bias_real = static_cast<phi::DenseTensor*>(bias_ptr.impl().get());
  } else {
    bias_real = new phi::DenseTensor();
  }
  int64_t stream_mode = ACL_STOP_ON_FAILURE;
  EXEC_NPU_CMD(aclnnMatmulReduceScatter,
               *dev_ctx,
               x1_tensor,
               x2_tensor,
               *bias_real,
               hcom_ptr,
               reduce_op_ptr,
               comm_turn,
               stream_mode,
               *result);

  return {paddle::Tensor(result)};
}

std::vector<std::vector<int64_t>> FusedMMReduceScatterInferShape(
    std::vector<int64_t> x1_shape, std::vector<int64_t> x2_shape) {
  return {x1_shape, x2_shape};
}

PD_BUILD_OP(fused_mm_reduce_scatter)
    .Inputs({"x1", "x2", paddle::Optional("bias")})
    .Outputs({"result"})
    .Attrs({"hcom:std::string",
            "world_size:int64_t",
            "reduce_op:std::string",
            "comm_turn:int64_t"})
    .SetKernelFn(PD_KERNEL(npu_mm_reduce_scatter))
    .SetInferShapeFn(
        PD_INFER_SHAPE(FusedMMReduceScatterInferShape));  // neccessary if the
                                                          // op has muti_inputs

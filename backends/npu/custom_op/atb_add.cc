// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include <acl/acl.h>
#include "atb_layer_base.h"

#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

std::shared_ptr<PpAscendAtbOpBase> g_atbAddOp;

std::vector<std::vector<int64_t>> AtbAddInferShape(
    const std::vector<int64_t> &x_shape) {

  return {x_shape};
}

std::vector<paddle::Tensor> AtbAddOp(const paddle::Tensor &x, const paddle::Tensor &y) {
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  if (!g_atbAddOp) {
    std::cout << "Run In AtbAddOp " << std::endl;
    g_atbAddOp.reset(new PpAscendAtbOpBase("AtbAddOp"));

    atb::Operation *op = nullptr;
    atb::infer::ElewiseParam mlpResidualAddParam;
    mlpResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(mlpResidualAddParam, &op);

    g_atbAddOp->operation_.reset(op);
  }
  std::vector<const phi::DenseTensor *> inputs;

  inputs.push_back(static_cast<const phi::DenseTensor *>(x.impl().get()));
  inputs.push_back(static_cast<const phi::DenseTensor *>(y.impl().get()));

  std::vector<const phi::DenseTensor *> outputs = {static_cast<const phi::DenseTensor *>(x.impl().get())};

  g_atbAddOp->Execute(stream, inputs, outputs);

  return {x};
}

PD_BUILD_OP(atb_add)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AtbAddOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        AtbAddInferShape)); // neccessary if the op has muti_inputs
#endif

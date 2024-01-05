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
#include <atb/atb_infer.h>

#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

std::shared_ptr<PpAscendAtbOpBase> g_atbMulsOp;

std::vector<paddle::Tensor> AtbMulsOp(const paddle::Tensor& x,
                                      float varAttr) {

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  std::vector<const phi::DenseTensor *> inputs;
  inputs.push_back(static_cast<const phi::DenseTensor *>(x.impl().get()));

  std::vector<const phi::DenseTensor *> outputs;
  outputs.push_back(static_cast<const phi::DenseTensor *>(x.impl().get()));

  if (!g_atbMulsOp) {
    std::cout << "Run In AtbMulsOp: " << std::endl;
    g_atbMulsOp.reset(new PpAscendAtbOpBase("AtbMulsOp"));

    atb::Operation *op = nullptr;
    atb::infer::ElewiseParam mulsParam;
    mulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
    mulsParam.mulsParam.varAttr = varAttr;
    atb::CreateOperation(mulsParam, &op);
    g_atbMulsOp->operation_.reset(op);
  }

  g_atbMulsOp->Execute(stream, inputs, outputs, dev_ctx);

  return {x};
}

std::vector<std::vector<int64_t>> AtbMulsOpInferShape(
    const std::vector<int64_t> &x_shape) {

    return {x_shape};
}

PD_BUILD_OP(atb_muls)
    .Inputs({"X"})
    .Outputs({"Y"})
    .Attrs({"varAttr: float"})
    .SetKernelFn(PD_KERNEL(AtbMulsOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        AtbMulsOpInferShape)); // neccessary if the op has muti_inputs
#endif

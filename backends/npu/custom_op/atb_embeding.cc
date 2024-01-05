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
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>
#include "atb_layer_base.h"
#include "llama_layer/llama_lmhead_operation.h"

#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

std::shared_ptr<PpAscendAtbOpBase> g_atbEmbedingOp;

void PerpareEmbedingInputs(
    const paddle::Tensor &x,
    const paddle::Tensor &y,
    std::vector<const phi::DenseTensor *> &inputs) {
  auto x_tensor = static_cast<const phi::DenseTensor *>(x.impl().get());
  auto y_tensor = static_cast<const phi::DenseTensor *>(y.impl().get());
  // 调整顺序 加速库参数权重在前 input id在后
  inputs.push_back(y_tensor);
  inputs.push_back(x_tensor);
}

std::vector<std::vector<int64_t>> AtbEmbedingInferShape(
    const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape) {
  std::vector<int64_t> output_shape = {x_shape[0], x_shape[1], y_shape[1]};
  return {output_shape};
}

std::vector<paddle::Tensor> AtbEmbedingOp(const paddle::Tensor &x, const paddle::Tensor &y) {
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  std::vector<int64_t> output_shape = {x.shape()[0], x.shape()[1], y.shape()[1]};

  g_atbEmbedingOp->output_->Resize(phi::make_ddim(output_shape));
  if (!g_atbEmbedingOp) {
    std::cout << "Run In AtbEmbedingOp " << std::endl;
    g_atbEmbedingOp.reset(new PpAscendAtbOpBase("AtbEmbedingOp"));

    atb::Operation *op = nullptr;
    atb::infer::GatherParam embedingParam;
    atb::CreateOperation(embedingParam, &op);

    g_atbEmbedingOp->operation_.reset(op);
    dev_ctx->Alloc(g_atbEmbedingOp->output_.get(), 
        static_cast<const phi::DenseTensor *>(y.impl().get())->dtype());  
  }
  std::vector<const phi::DenseTensor *> inputs;
  PerpareEmbedingInputs(x,
                    y,
                    inputs);

  std::vector<const phi::DenseTensor *> outputs;
  outputs.push_back(g_atbEmbedingOp->output_.get());

  g_atbEmbedingOp->Execute(stream, inputs, outputs, dev_ctx);

  return {paddle::Tensor(g_atbEmbedingOp->output_)};
}

PD_BUILD_OP(atb_embeding)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AtbEmbedingOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        AtbEmbedingInferShape)); // neccessary if the op has muti_inputs
#endif

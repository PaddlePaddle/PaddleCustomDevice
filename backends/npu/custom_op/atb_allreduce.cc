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

std::shared_ptr<PpAscendAtbOpBase> g_atbAllreduceOp;

std::vector<std::vector<int64_t>> AtbAllreduceInferShape(
    const std::vector<int64_t> &x_shape) {
  return {x_shape};
}

std::vector<paddle::Tensor> AtbAllreduceOp(const paddle::Tensor &x) {
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  auto comm = reinterpret_cast<HcclComm>(phi::detail::GetCCLComm(x.place(), 0));
  auto x_tensor = static_cast<const phi::DenseTensor *>(x.impl().get());

  g_atbAllreduceOp->output_->Resize(phi::make_ddim(x.shape()));
  if (!g_atbAllreduceOp) {
    std::cout << "Run In AtbAllreduceOp " << std::endl;
    g_atbAllreduceOp.reset(new PpAscendAtbOpBase("AtbAllreduceOp"));

    std::string device_id_str = getenv("FLAGS_selected_npus");
    int device_id = stoi(device_id_str);

    atb::Operation *op = nullptr;
    atb::infer::AllReduceParam allReduceParam = { device_id, 8, 0, "sum", "lccl", comm };
    atb::CreateOperation(allReduceParam, &op);

    g_atbAllreduceOp->operation_.reset(op);
    dev_ctx->Alloc(g_atbAllreduceOp->output_.get(), x_tensor->dtype());  
  }
  std::vector<const phi::DenseTensor *> inputs;

  inputs.push_back(x_tensor);

  std::vector<const phi::DenseTensor *> outputs;
  outputs.push_back(g_atbAllreduceOp->output_.get());

  g_atbAllreduceOp->Execute(stream, inputs, outputs, dev_ctx);

  return {paddle::Tensor(g_atbAllreduceOp->output_)};
}

PD_BUILD_OP(atb_allreduce)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AtbAllreduceOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        AtbAllreduceInferShape)); // neccessary if the op has muti_inputs
#endif

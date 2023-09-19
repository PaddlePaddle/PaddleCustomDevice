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
#include <atb/atb_infer.h>
#include "llama_lmhead.h"

#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

std::shared_ptr<PpAtbLlaMaLmHeadOp> g_llaMaLmheadOp;

PpAtbLlaMaLmHeadOp::PpAtbLlaMaLmHeadOp(const std::string &modelName) : PpAscendAtbOpBase(modelName) {}

PpAtbLlaMaLmHeadOp::~PpAtbLlaMaLmHeadOp() {}

std::vector<paddle::Tensor> LlaMaLmHeadOp(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    // const paddle::Tensor &matmul_weight,
    float rmsNormEps) {

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(hidden.place()));

  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  auto comm = reinterpret_cast<HcclComm>(phi::detail::GetCCLComm(hidden.place(), 0));

  std::vector<int64_t> hidden_shape = hidden.shape();
  std::vector<int64_t> key_shape;
  std::vector<int64_t> value_shape;

  auto data_type = static_cast<const phi::DenseTensor *>(hidden.impl().get())->dtype();

  std::shared_ptr<phi::DenseTensor> layerout_tensor = std::make_shared<phi::DenseTensor>();
  layerout_tensor->Resize(phi::make_ddim(hidden.shape()));
  dev_ctx->Alloc(layerout_tensor.get(), data_type);

  std::vector<const phi::DenseTensor *> inputs;
  inputs.push_back(static_cast<const phi::DenseTensor *>(hidden.impl().get()));

  std::vector<const phi::DenseTensor *> outputs;
  outputs.push_back(layerout_tensor.get());

  if (!g_llaMaLmheadOp) {
    std::cout << "Run In LlaMaLmHeadOp: " << std::endl;
    g_llaMaLmheadOp.reset(new PpAtbLlaMaLmHeadOp("LlaMaLmHeadOp"));

    atb::Operation *op = nullptr;
    atb::infer::RmsNormParam param;
    param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    param.normParam.epsilon = rmsNormEps;
    atb::CreateOperation(param, &op);
    g_llaMaLmheadOp->operation_.reset(op);
  }

  g_llaMaLmheadOp->Execute(stream, inputs, outputs);

  return {paddle::Tensor(layerout_tensor)};
}

std::vector<std::vector<int64_t>> LlaMaLmHeadOpInferShape(
    const std::vector<int64_t> &hidden_shape,
    const std::vector<int64_t> &norm_weight_shape,
    // const std::vector<int64_t> &matmul_weight_shape,
    float rmsNormEps) {
  // std::vector<int64_t> shape;

  // shape.push_back(hidden_shape.at(0));
  // shape.push_back(matmul_weight_shape.at(1) * 8);

  return {hidden_shape};
}

PD_BUILD_OP(llama_lmhead)
    .Inputs({"Hidden",
             "NormWeight"})
            //  "MatmulWeight"})
    .Outputs({"Out"})
    .Attrs({"rmsNormEps: float"})
    .SetKernelFn(PD_KERNEL(LlaMaLmHeadOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        LlaMaLmHeadOpInferShape)); // neccessary if the op has muti_inputs
#endif
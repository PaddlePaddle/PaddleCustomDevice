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
#include "atb_transdata.h"
#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

std::shared_ptr<PpAtbLlamaTransdataNzParallelOp> g_atbTransdataOp;

PpAtbLlamaTransdataNzParallelOp::PpAtbLlamaTransdataNzParallelOp(
    const std::string &modelName) : PpAscendAtbOpBase(modelName) {}

PpAtbLlamaTransdataNzParallelOp::~PpAtbLlamaTransdataNzParallelOp() {}


void PpAtbLlamaTransdataNzParallelOp::BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                                              std::vector<const phi::DenseTensor *> &outTensors)
{
  variantPacks_.inTensors.resize(inTensors.size());
  for (size_t i = 0; i < inTensors.size(); i++) {
    variantPacks_.inTensors.at(i) = ConvertDenseTensorToAtbTensor(*(inTensors.at(i)));
    if (variantPacks_.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.inTensors.at(i).desc.format = ACL_FORMAT_ND;
    }
  }

  variantPacks_.outTensors.resize(outTensors.size());
  for (size_t i = 0; i < outTensors.size(); i++) {
    variantPacks_.outTensors.at(i) = ConvertDenseTensorToAtbTensor(*(outTensors.at(i)));
    if (variantPacks_.outTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.outTensors.at(i).desc.format = ACL_FORMAT_FRACTAL_NZ;
    }
  }
}

void PerpareTransdataInputs(
    const paddle::Tensor &x,
    std::vector<const phi::DenseTensor *> &inputs) {
  auto x_tensor = static_cast<const phi::DenseTensor *>(x.impl().get());
  inputs.push_back(x_tensor);
}

std::vector<std::vector<int64_t>> AtbTransdataInferShape(
    const std::vector<int64_t> &x_shape) {
  std::vector<std::vector<int64_t>> output_shape;
  output_shape.push_back({1, x_shape[1] / 16, x_shape[0], 16});
  return output_shape;
}

std::vector<paddle::Tensor> AtbTransdataOp(const paddle::Tensor &x) {
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));

  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  std::shared_ptr<phi::DenseTensor> out_tensor =
      std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim({1, x.shape()[1] / 16, x.shape()[0], 16}));
  dev_ctx->Alloc(out_tensor.get(), 
      static_cast<const phi::DenseTensor *>(x.impl().get())->dtype());  

  if (!g_atbTransdataOp) {
    std::cout << "Run In AtbTransdataOp " << std::endl;
    g_atbTransdataOp.reset(new PpAtbLlamaTransdataNzParallelOp("AtbTransdataOp"));

    atb::Operation *op = nullptr;
    atb::infer::TransdataParam transdataParam;
    transdataParam.transdataType = atb::infer::TransdataParam::ND_TO_FRACTAL_NZ;
    atb::CreateOperation(transdataParam, &op);

    g_atbTransdataOp->operation_.reset(op);
  }
  std::vector<const phi::DenseTensor *> inputs;
  PerpareTransdataInputs(x, inputs);

  std::vector<const phi::DenseTensor *> outputs;
  outputs.push_back(out_tensor.get());

  g_atbTransdataOp->Execute(stream, inputs, outputs);

  return {paddle::Tensor(out_tensor)};
}

PD_BUILD_OP(atb_transdata)
    .Inputs({"X"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AtbTransdataOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        AtbTransdataInferShape)); // neccessary if the op has muti_inputs
#endif

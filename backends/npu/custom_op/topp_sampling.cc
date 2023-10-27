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
#include "llama_layer/topp_sampling_operation.h"

#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

std::shared_ptr<PpAscendAtbOpBase> g_toppSamplingOp;

std::vector<paddle::Tensor> TopPSampling(const paddle::Tensor& x,
                                         const paddle::Tensor& top_ps,
                                         int random_seed) {
  // TODO:当前版本不支设置参数random_seed，不支持返回probs
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(x.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  std::vector<int64_t> x_shape = x.shape();
  std::vector<int64_t> out_probs_shape = {x_shape[0], 1};
  std::vector<int64_t> out_ids_shape = {x_shape[0], 1};

  std::shared_ptr<phi::DenseTensor> out_ids_tensor = std::make_shared<phi::DenseTensor>();
  out_ids_tensor->Resize(phi::make_ddim(out_ids_shape));
  dev_ctx->Alloc(out_ids_tensor.get(), paddle::DataType::INT64);

  std::shared_ptr<phi::DenseTensor> out_probs_tensor = std::make_shared<phi::DenseTensor>();
  out_probs_tensor->Resize(phi::make_ddim(out_probs_shape));
  dev_ctx->Alloc(out_probs_tensor.get(), paddle::DataType::FLOAT16);

  std::vector<const phi::DenseTensor *> outputs;
  outputs.push_back(out_ids_tensor.get());
  outputs.push_back(out_probs_tensor.get());

  std::vector<const phi::DenseTensor *> inputs;
  inputs.push_back(static_cast<const phi::DenseTensor *>(x.impl().get()));
  inputs.push_back(static_cast<const phi::DenseTensor *>(top_ps.impl().get()));

  if (!g_toppSamplingOp) {
    std::cout << "Run In AtbTopPSamplingOp: " << std::endl;
    g_toppSamplingOp.reset(new PpAscendAtbOpBase("AtbTopPSamplingOp"));

    atb::Operation *op = nullptr;
    ToppSamplingParam param = {(uint32_t)random_seed, 50}; // topk 50
    CreateToppSamplingOperation(param, &op);
    g_toppSamplingOp->operation_.reset(op);
  }

  g_toppSamplingOp->Execute(stream, inputs, outputs);

  return {paddle::Tensor(out_probs_tensor), paddle::Tensor(out_ids_tensor)};
}

std::vector<std::vector<int64_t>> TopPSamplingInferShape(const std::vector<int64_t>& x_shape,
                                                         const std::vector<int64_t>& top_ps_shape) {
    std::vector<int64_t> out_probs_shape = {x_shape[0], 1};                                                          
    std::vector<int64_t> out_ids_shape = {x_shape[0], 1};
    return {out_probs_shape, out_ids_shape};
}

std::vector<paddle::DataType> TopPSamplingInferDtype(const paddle::DataType& x_dtype,
                                                     const paddle::DataType& top_ps_dtype) {
    return {x_dtype, paddle::DataType::INT64};
}


PD_BUILD_OP(top_p_sampling)
    .Inputs({"x", "top_ps"})
    .Outputs({"topp_probs", "topp_ids"})
    .Attrs({"random_seed: int"})
    .SetKernelFn(PD_KERNEL(TopPSampling))
    .SetInferShapeFn(PD_INFER_SHAPE(TopPSamplingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopPSamplingInferDtype));
#endif

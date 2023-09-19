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
#include "llama_layer_parallel_op.h"
#include "llama_layer/llama_fusion_parallel_operation.h"
#include "llama_layer/llama_encoder_parallel_operation.h"
#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

std::shared_ptr<PpAtbLlaMaEncoderLayerParallelOp> g_llaMaEncoderLayerParallelOp;
static int32_t g_llamaEncoderLayerId = 0;
static uint64_t executeCount_ = 0;

void PerpareLlaMaEncoderLayerInputs(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &qkv_mix_weight,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &mlp_gate_up_weight,
    const paddle::Tensor &mlp_down_weight,
    const paddle::Tensor &positionIDs,
    const paddle::Tensor &cos_sin_table,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &cache_key_value,
    const paddle::Tensor &kv_seq_len,
    const paddle::Tensor &q_seq_len,
    phi::DenseTensor &layer_id_dense,
    std::vector<const phi::DenseTensor *> &inputs) {

  auto hidden_tensor = static_cast<const phi::DenseTensor *>(hidden.impl().get());
  auto norm_weight_tensor = static_cast<const phi::DenseTensor *>(norm_weight.impl().get());
  auto qkv_mix_weight_tensor = static_cast<const phi::DenseTensor *>(qkv_mix_weight.impl().get());
  auto self_out_linear_weight_tensor = static_cast<const phi::DenseTensor *>(self_out_linear_weight.impl().get());
  auto self_out_norm_weight_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_weight.impl().get());
  auto mlp_gate_up_weight_tensor = static_cast<const phi::DenseTensor *>(mlp_gate_up_weight.impl().get());
  auto mlp_down_weight_tensor = static_cast<const phi::DenseTensor *>(mlp_down_weight.impl().get());
  auto positionIDs_tensor = static_cast<const phi::DenseTensor *>(positionIDs.impl().get());
  auto cos_sin_table_tensor = static_cast<const phi::DenseTensor *>(cos_sin_table.impl().get());
  auto attention_mask_tensor = static_cast<const phi::DenseTensor *>(attention_mask.impl().get());
  auto cache_key_value_tensor = static_cast<const phi::DenseTensor *>(cache_key_value.impl().get());
  auto kv_seq_len_tensor = static_cast<const phi::DenseTensor *>(kv_seq_len.impl().get());
  auto q_seq_len_tensor = static_cast<const phi::DenseTensor *>(q_seq_len.impl().get());

  inputs.push_back(hidden_tensor);
  inputs.push_back(norm_weight_tensor);
  inputs.push_back(qkv_mix_weight_tensor);
  inputs.push_back(self_out_linear_weight_tensor);
  inputs.push_back(self_out_norm_weight_tensor);
  inputs.push_back(mlp_gate_up_weight_tensor);
  inputs.push_back(mlp_down_weight_tensor);
  inputs.push_back(positionIDs_tensor);
  inputs.push_back(cos_sin_table_tensor);
  inputs.push_back(attention_mask_tensor);
  inputs.push_back(cache_key_value_tensor);
  inputs.push_back(kv_seq_len_tensor);
  inputs.push_back(q_seq_len_tensor);
  inputs.push_back(&layer_id_dense);
}

PpAtbLlaMaEncoderLayerParallelOp::PpAtbLlaMaEncoderLayerParallelOp(
    const std::string &modelName, int32_t layerNum) : PpAscendAtbOpBase(modelName) {
  layerNum_ = layerNum;
}

PpAtbLlaMaEncoderLayerParallelOp::~PpAtbLlaMaEncoderLayerParallelOp() {}

std::vector<paddle::Tensor> LlaMaEncoderLayerParallelOp(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &qkv_mix_weight,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &mlp_gate_up_weight,
    const paddle::Tensor &mlp_down_weight,
    const paddle::Tensor &positionIDs,
    const paddle::Tensor &cos_sin_table,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &cache_key_value,
    const paddle::Tensor &kv_seq_len,
    float rmsNormEps,
    int headDim,
    int headNum) {

  // int32_t layer_num = 80; /* TODO:65B，写死8卡 */
  int32_t layer_num = 32; /* TODO:7B，写死8卡 */
  int32_t head_num = headNum;
  int32_t head_dim = headDim;

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(hidden.place()));

  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  auto comm = reinterpret_cast<HcclComm>(phi::detail::GetCCLComm(hidden.place(), 0));

  std::vector<int64_t> hidden_shape = hidden.shape();
  std::vector<int64_t> key_shape;
  std::vector<int64_t> value_shape;

  key_shape.push_back(hidden_shape.at(0));
  key_shape.push_back(hidden_shape.at(1));
  key_shape.push_back(head_num);
  key_shape.push_back(head_dim);

  value_shape.push_back(hidden_shape.at(0));
  value_shape.push_back(hidden_shape.at(1));
  value_shape.push_back(head_num);
  value_shape.push_back(head_dim);

  auto data_type = static_cast<const phi::DenseTensor *>(hidden.impl().get())->dtype();

  std::shared_ptr<phi::DenseTensor> layerout_tensor = std::make_shared<phi::DenseTensor>();
  layerout_tensor->Resize(phi::make_ddim(hidden.shape()));
  dev_ctx->Alloc(layerout_tensor.get(), data_type);

  std::vector<const phi::DenseTensor *> outputs;
  outputs.push_back(layerout_tensor.get());

  if (!g_llaMaEncoderLayerParallelOp) {
    std::cout << "Run In Encoder Parallel layernum: " << layer_num << " head_num: " << head_num << "head_dim: " << head_dim << std::endl;
    g_llaMaEncoderLayerParallelOp.reset(new PpAtbLlaMaEncoderLayerParallelOp("LlaMaEncoderLayerParallelOp", layer_num));

    atb::Operation *op = nullptr;
    LlamaLayerFusionParallelParam param = {rmsNormEps,
                                           head_num,
                                           head_dim,
                                           0,
                                           0,
                                           0,
                                           2,
                                           true,
                                           {0},
                                           {0},
                                           comm};
    LlamaLayerFusionParallelOperation(param, &op);
    g_llaMaEncoderLayerParallelOp->operation_.reset(op);
    std::vector<int32_t> layer_id_vec(1, 0);
    custom_kernel::TensorFromVector(*dev_ctx, layer_id_vec,
                                    *dev_ctx, &(g_llaMaEncoderLayerParallelOp->layerIdTensor));
  }

  std::vector<const phi::DenseTensor *> inputs;
  PerpareLlaMaEncoderLayerInputs(hidden,
                                 norm_weight,
                                 qkv_mix_weight,
                                 self_out_linear_weight,
                                 self_out_norm_weight,
                                 mlp_gate_up_weight,
                                 mlp_down_weight,
                                 positionIDs,
                                 cos_sin_table,
                                 attention_mask,
                                 cache_key_value,
                                 kv_seq_len, // token offset即kv_seq_len
                                 kv_seq_len, // 增量q_seq_len，始终为1
                                 g_llaMaEncoderLayerParallelOp->layerIdTensor,
                                 inputs);

  g_llaMaEncoderLayerParallelOp->Execute(stream, inputs, outputs);

  executeCount_++;
  if ((executeCount_) % layer_num == 0) {
    int ret = aclrtSynchronizeStream(stream);
  }

  return {paddle::Tensor(layerout_tensor),
          paddle::Tensor(cache_key_value)};
}

std::vector<std::vector<int64_t>> LlaMaEncoderLayerOpInferShape(
    const std::vector<int64_t> &hidden_shape,
    const std::vector<int64_t> &norm_weight_shape,
    const std::vector<int64_t> &qkv_mix_weight_shape,
    const std::vector<int64_t> &self_out_linear_weight_shape,
    const std::vector<int64_t> &self_out_norm_weight_shape,
    const std::vector<int64_t> &mlp_gate_up_weight_shape,
    const std::vector<int64_t> &mlp_down_weight_shape,
    const std::vector<int64_t> &positionIDs_shape,
    const std::vector<int64_t> &cos_sin_table_shape,
    const std::vector<int64_t> &attention_mask_shape,
    const std::vector<int64_t> &cacheKV_shape,
    const std::vector<int64_t> &seq_len_shape,
    float rmsNormEps,
    int headDim,
    int headNum) {

  int32_t head_num = headNum; /* TODO:64个，写死8卡 */
  int32_t head_dim = headDim;

  std::vector<int64_t> key_shape; /* [bs, seq_len, hidden_size] */
  std::vector<int64_t> value_shape; /* [bs, seq_len, hidden_size] */
  key_shape.push_back(hidden_shape.at(0));
  key_shape.push_back(hidden_shape.at(1));
  key_shape.push_back(head_num);  /* TODO:当前写死了6.7B模型  */
  key_shape.push_back(head_dim);

  value_shape.push_back(hidden_shape.at(0));
  value_shape.push_back(hidden_shape.at(1));
  value_shape.push_back(head_num);
  value_shape.push_back(head_dim);
  return {hidden_shape, cacheKV_shape};
}

PD_BUILD_OP(llama_encoder_layer_parallel)
    .Inputs({"Hidden",
             "NormWeight",
             "QKVMixWeight",
             "SelfOutLinearWeight",
             "SelfOutNormWeight",
             "MlpGateUpWeight",
             "MlpDownWeight",
             "PositionIDs",
             "CosSinTable",
             "AttentionMask",
             "Cache_KV",
             "SeqLength"})
    .Outputs({"Out", "PresentKV"})
    .Attrs({"rmsNormEps: float",
            "headDim: int",
            "headNum: int"})
            // "headnum: float",
            // "shape: std::vector<int>",
            // "scale: float"})
    .SetKernelFn(PD_KERNEL(LlaMaEncoderLayerParallelOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        LlaMaEncoderLayerOpInferShape)); // neccessary if the op has muti_inputs
#endif
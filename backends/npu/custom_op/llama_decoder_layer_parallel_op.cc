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
#include "llama_layer_parallel_op.h"
#include "llama_layer/llama_fusion_parallel_operation.h"
#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

std::shared_ptr<PpAtbLlaMaDecoderLayerParallelOp> g_llaMaDecoderLayerParallelOp;
static int32_t g_llamadecoderLayerId = 0;
static uint64_t executeCount_ = 0;

struct PpAtbSeqLen {
  std::vector<int32_t> kv_seq_len_vec;
  std::vector<int32_t> q_seq_len_vec;
  atb::SVector<int32_t> kv_seq_len_param;
  atb::SVector<int32_t> q_seq_len_param;
  phi::DenseTensor kv_seq_len_tensor;
  phi::DenseTensor q_seq_len_tensor;
};

static PpAtbSeqLen g_atbSeqLen;
static atb::Tensor g_llama_cachek;
static atb::Tensor g_llama_cachev;
static atb::Tensor g_llama_attenmask;

void PerpareLlaMaDecoderLayerInputs(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &q_mix_weight,
    const paddle::Tensor &k_mix_weight,
    const paddle::Tensor &v_mix_weight,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &mlp_gate_weight,
    const paddle::Tensor &mlp_down_weight,
    const paddle::Tensor &mlp_up_weight,
    const paddle::Tensor &positionIDs,
    const paddle::Tensor &cos_table,
    const paddle::Tensor &sin_table,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &past_key,
    const paddle::Tensor &past_value,
    const paddle::Tensor &kv_seq_len,
    phi::DenseTensor &q_seq_len_dense,
    phi::DenseTensor &layer_id_dense,
    std::vector<const phi::DenseTensor *> &inputs) {

  auto hidden_tensor = static_cast<const phi::DenseTensor *>(hidden.impl().get());
  auto norm_weight_tensor = static_cast<const phi::DenseTensor *>(norm_weight.impl().get());
  auto q_mix_weight_tensor = static_cast<const phi::DenseTensor *>(q_mix_weight.impl().get());
  auto k_mix_weight_tensor = static_cast<const phi::DenseTensor *>(k_mix_weight.impl().get());
  auto v_mix_weight_tensor = static_cast<const phi::DenseTensor *>(v_mix_weight.impl().get());
  auto self_out_linear_weight_tensor = static_cast<const phi::DenseTensor *>(self_out_linear_weight.impl().get());
  auto self_out_norm_weight_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_weight.impl().get());
  auto mlp_gate_weight_tensor = static_cast<const phi::DenseTensor *>(mlp_gate_weight.impl().get());
  auto mlp_down_weight_tensor = static_cast<const phi::DenseTensor *>(mlp_down_weight.impl().get());
  auto mlp_up_weight_tensor = static_cast<const phi::DenseTensor *>(mlp_up_weight.impl().get());
  auto positionIDs_tensor = static_cast<const phi::DenseTensor *>(positionIDs.impl().get());
  auto cos_table_tensor = static_cast<const phi::DenseTensor *>(cos_table.impl().get());
  auto sin_table_tensor = static_cast<const phi::DenseTensor *>(sin_table.impl().get());
  auto attention_mask_tensor = static_cast<const phi::DenseTensor *>(attention_mask.impl().get());
  auto past_key_tensor = static_cast<const phi::DenseTensor *>(past_key.impl().get());
  auto past_value_tensor = static_cast<const phi::DenseTensor *>(past_value.impl().get());
  auto kv_seq_len_tensor = static_cast<const phi::DenseTensor *>(kv_seq_len.impl().get());

  inputs.push_back(hidden_tensor);
  inputs.push_back(norm_weight_tensor);
  inputs.push_back(q_mix_weight_tensor);
  inputs.push_back(k_mix_weight_tensor);
  inputs.push_back(v_mix_weight_tensor);
  inputs.push_back(self_out_linear_weight_tensor);
  inputs.push_back(self_out_norm_weight_tensor);
  inputs.push_back(mlp_gate_weight_tensor);
  inputs.push_back(mlp_down_weight_tensor);
  inputs.push_back(mlp_up_weight_tensor);
  inputs.push_back(positionIDs_tensor);
  inputs.push_back(cos_table_tensor);
  inputs.push_back(sin_table_tensor);
  inputs.push_back(attention_mask_tensor);
  inputs.push_back(past_key_tensor);
  inputs.push_back(past_value_tensor);
  inputs.push_back(kv_seq_len_tensor);
  inputs.push_back(&q_seq_len_dense);
  inputs.push_back(&layer_id_dense);
}

void LlamaLayerFusionParallelOpUpdateParam(atb::VariantPack &variantPack)
{
  const uint32_t tokenOffsetTensorId = LlamaLayerFusionParallelTensorId::IN_TOKENOFFSET;
  const uint32_t seqLenTensorId = LlamaLayerFusionParallelTensorId::IN_SEQLEN;
  const uint32_t layerIdTensorId = LlamaLayerFusionParallelTensorId::IN_LAYERID;

  // TODO: 如何更新kv_seq_len, q_seq_len
  variantPack.inTensors.at(tokenOffsetTensorId).hostData = g_atbSeqLen.kv_seq_len_param.data();
  variantPack.inTensors.at(seqLenTensorId).hostData = g_atbSeqLen.q_seq_len_param.data();
  variantPack.inTensors.at(layerIdTensorId).hostData = &g_llamadecoderLayerId;
}

void PpAtbLlaMaDecoderLayerParallelOp::BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                                                 std::vector<const phi::DenseTensor *> &outTensors)
{
  variantPacks_.inTensors.resize(inTensors.size() + 3);
  for (size_t i = 0; i < inTensors.size(); i++) {
    variantPacks_.inTensors.at(i) = ConvertDenseTensorToAtbTensor(*(inTensors.at(i)));
    if (variantPacks_.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.inTensors.at(i).desc.format = ACL_FORMAT_ND;
    }
  }

  variantPacks_.inTensors.at(inTensors.size()) = g_llama_attenmask;
  variantPacks_.inTensors.at(inTensors.size() + 1) = g_llama_cachek;
  variantPacks_.inTensors.at(inTensors.size() + 2) = g_llama_cachev;

  variantPacks_.outTensors.resize(outTensors.size());
  for (size_t i = 0; i < outTensors.size(); i++) {
    variantPacks_.outTensors.at(i) = ConvertDenseTensorToAtbTensor(*(outTensors.at(i)));
    if (variantPacks_.outTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.outTensors.at(i).desc.format = ACL_FORMAT_ND;
    }
  }
  // param需要更新，依赖这种方式
  LlamaLayerFusionParallelOpUpdateParam(variantPacks_);
}

PpAtbLlaMaDecoderLayerParallelOp::PpAtbLlaMaDecoderLayerParallelOp(
    const std::string &modelName, int32_t layerNum) : PpAscendAtbOpBase(modelName) {
    layerNum_ = layerNum;
}

PpAtbLlaMaDecoderLayerParallelOp::~PpAtbLlaMaDecoderLayerParallelOp() {}

std::vector<paddle::Tensor> LlaMaDecoderLayerParallelOp(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &q_mix_weight,
    const paddle::Tensor &k_mix_weight,
    const paddle::Tensor &v_mix_weight,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &mlp_gate_weight,
    const paddle::Tensor &mlp_down_weight,
    const paddle::Tensor &mlp_up_weight,
    const paddle::Tensor &positionIDs,
    const paddle::Tensor &cos_table,
    const paddle::Tensor &sin_table,
    const paddle::Tensor &attention_mask, // TODO:待确认attention mask是否符合加速库
    const paddle::Tensor &kv_seq_len,
    const paddle::Tensor &past_key,
    const paddle::Tensor &past_value,
    float rmsNormEps,
    std::vector<int32_t> shape) {

  int32_t batch_size = past_key.shape().at(0);
  int32_t org_seq_len = past_key.shape().at(1);
  // int32_t layer_num = 80; /* TODO:65B，写死8卡 */
  // int32_t head_num = 8;
  // int32_t head_dim = 128;
  int32_t layer_num = 32; /* TODO:7B，写死8卡 */
  // int32_t head_num = 4;
  // int32_t head_dim = 128;
  int32_t head_num = shape[2];
  int32_t head_dim = shape[3];

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(hidden.place()));

  if (g_llamadecoderLayerId % layer_num == 0) {
    g_llamadecoderLayerId = 0;
    if (!g_llaMaDecoderLayerParallelOp) {
      /* qLen增量阶段始终为1 */
      g_atbSeqLen.q_seq_len_vec.clear();
      g_atbSeqLen.q_seq_len_vec.resize(batch_size, 1);
      atb::SVector<int32_t> q_seq_len(batch_size, 1);
      g_atbSeqLen.q_seq_len_param = q_seq_len;
      custom_kernel::TensorFromVector(*dev_ctx, g_atbSeqLen.q_seq_len_vec,
                                      *dev_ctx, &(g_atbSeqLen.q_seq_len_tensor));
    }
  }

  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  auto comm = reinterpret_cast<HcclComm>(phi::detail::GetCCLComm(hidden.place(), 0));

  std::shared_ptr<phi::DenseTensor> layerout_tensor = std::make_shared<phi::DenseTensor>();
  layerout_tensor->Resize(phi::make_ddim(hidden.shape()));
  dev_ctx->Alloc(layerout_tensor.get(),
      static_cast<const phi::DenseTensor *>(hidden.impl().get())->dtype());

  if (!g_llaMaDecoderLayerParallelOp) {
    std::cout << "Run In DDDDecoder Parallel layernum: " << layer_num << " head_num: " << head_num << "head_dim: " << head_dim << std::endl;

    g_llaMaDecoderLayerParallelOp.reset(new PpAtbLlaMaDecoderLayerParallelOp("LlaMaDecoderLayerParallelOp", layer_num));

    atb::Operation *op = nullptr;
    LlamaLayerFusionParallelParam param = {rmsNormEps,
                                           head_num,
                                           head_dim,
                                           0,
                                           0,
                                           "llama13b",
                                           g_llamadecoderLayerId,
                                           2,
                                           true,
                                           g_atbSeqLen.kv_seq_len_param,
                                           g_atbSeqLen.q_seq_len_param,
                                           comm};
    LlamaLayerFusionParallelOperation(param, &op);
    g_llaMaDecoderLayerParallelOp->operation_.reset(op);

    // 加速库支持layer之间的cache是连续，通过layerid进行偏移。
    // 当前传入cache都是layer的首地址，则layerid设零即可。
    std::vector<int32_t> layer_id_vec(1, 0);
    custom_kernel::TensorFromVector(*dev_ctx, layer_id_vec,
                                    *dev_ctx, &(g_llaMaDecoderLayerParallelOp->layerIdTensor));
  }

  g_llamadecoderLayerId++;

  std::vector<const phi::DenseTensor *> inputs;
  PerpareLlaMaDecoderLayerInputs(hidden,
                                 norm_weight,
                                 q_mix_weight,
                                 k_mix_weight,
                                 v_mix_weight,
                                 self_out_linear_weight,
                                 self_out_norm_weight,
                                 mlp_gate_weight,
                                 mlp_down_weight,
                                 mlp_up_weight,
                                 positionIDs,
                                 cos_table,
                                 sin_table,
                                 attention_mask,
                                 past_key,
                                 past_value,
                                 kv_seq_len, // token offset即kv_seq_len
                                 g_atbSeqLen.q_seq_len_tensor, // 增量q_seq_len，始终为1
                                 g_llaMaDecoderLayerParallelOp->layerIdTensor,
                                 inputs);
  std::vector<const phi::DenseTensor *> outputs = {layerout_tensor.get()};
  g_llaMaDecoderLayerParallelOp->Execute(stream, inputs, outputs);

  executeCount_++;
  if ((executeCount_) % layer_num == 0) {
    int ret = aclrtSynchronizeStream(stream);
  }

  return {paddle::Tensor(layerout_tensor), past_key, past_value}; // TODO:待确认past_key返回
}

std::vector<std::vector<int64_t>> LlaMaDecoderLayerOpInferShape(
    const std::vector<int64_t> &hidden_shape,
    const std::vector<int64_t> &norm_weight_shape,
    const std::vector<int64_t> &q_mix_weight_shape,
    const std::vector<int64_t> &k_mix_weight_shape,
    const std::vector<int64_t> &v_mix_weight_shape,
    const std::vector<int64_t> &self_out_linear_weight_shape,
    const std::vector<int64_t> &self_out_norm_weight_shape,
    const std::vector<int64_t> &mlp_gate_weight_shape,
    const std::vector<int64_t> &mlp_down_weight_shape,
    const std::vector<int64_t> &mlp_up_weight_shape,
    const std::vector<int64_t> &positionIDs_shape,
    const std::vector<int64_t> &cos_table_shape,
    const std::vector<int64_t> &sin_table_shape,
    const std::vector<int64_t> &attention_mask_shape,
    const std::vector<int64_t> &past_key_shape,
    const std::vector<int64_t> &past_value_shape) {

  return {hidden_shape, past_key_shape, past_value_shape};
}

PD_BUILD_OP(llama_decoder_layer_parallel)
    .Inputs({"Hidden",
             "NormWeight",
             "QMixWeight",
             "KMixWeight",
             "VMixWeight",
             "SelfOutLinearWeight",
             "SelfOutNormWeight",
             "MlpGateWeight",
             "MlpDownWeight",
             "MlpUpWeight",
             "PositionIDs",
             "CosTable",
             "SinTable",
             "AttentionMask",
             "SeqLength",
             "CacheK",
             "CacheV"})
    .Outputs({"Out", "PresentKey", "PresentValue"})
    .Attrs({"rmsNormEps: float",
            "shape: std::vector<int>"})
    .SetKernelFn(PD_KERNEL(LlaMaDecoderLayerParallelOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        LlaMaDecoderLayerOpInferShape)); // neccessary if the op has muti_inputs
#endif
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

std::shared_ptr<PpAtbLlamaEncoderLayerParallelOp> g_llamaEncoderLayerParallelOp;
static uint64_t executeCount = 0;

void PerpareLlamaEncoderLayerInputs(
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
  auto cache_key_value_tensor2 = static_cast<const phi::DenseTensor *>(cache_key_value.impl().get());
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
  inputs.push_back(cache_key_value_tensor2);
  inputs.push_back(kv_seq_len_tensor);
  inputs.push_back(q_seq_len_tensor);
  inputs.push_back(&layer_id_dense);
}

atb::Tensor PpAtbLlamaEncoderLayerParallelOp::CreateBatchStatusAtbHostTensor()
{
  atb::Tensor atbTensor;
  atbTensor.desc.format = ACL_FORMAT_ND;

  atbTensor.desc.shape.dimNum = 1;
  atbTensor.desc.shape.dims[0] = curBatchSize_;
  atbTensor.desc.dtype = ACL_UINT32;

  atbTensor.dataSize = atb::Utils::GetTensorSize(atbTensor);
  return atbTensor;
}

void PpAtbLlamaEncoderLayerParallelOp::BindHostTensorForUpdateParam(atb::VariantPack &variantPack)
{
  const uint32_t tokenOffsetTensorId = LlamaLayerFusionParallelTensorId::IN_TOKENOFFSET;
  const uint32_t seqLenTensorId = LlamaLayerFusionParallelTensorId::IN_SEQLEN;
  const uint32_t batchStatusTensorId = LlamaLayerFusionParallelTensorId::IN_BATCH_STATUS;

  variantPack.inTensors.at(tokenOffsetTensorId).hostData = kv_seq_len_param_.data();
  variantPack.inTensors.at(seqLenTensorId).hostData = q_seq_len_param_.data();
  variantPack.inTensors.at(batchStatusTensorId).hostData = batch_status_param_.data();
}

void PpAtbLlamaEncoderLayerParallelOp::BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                                                        std::vector<const phi::DenseTensor *> &outTensors)
{
  variantPacks_.inTensors.resize(inTensors.size() + 1); //补充
  for (size_t i = 0; i < inTensors.size(); i++) {
    if(i == 10) { // CACHE_K
      variantPacks_.inTensors.at(i) = ConvertDenseTensorToAtbTensorK(*(inTensors.at(i)));
      if (variantPacks_.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
        variantPacks_.inTensors.at(i).desc.format = ACL_FORMAT_ND;
      }
    }else if(i == 11) {// CACHE_V
      variantPacks_.inTensors.at(i) = ConvertDenseTensorToAtbTensorV(*(inTensors.at(i)));
      if (variantPacks_.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
        variantPacks_.inTensors.at(i).desc.format = ACL_FORMAT_ND;
      }    
    }else{
      variantPacks_.inTensors.at(i) = ConvertDenseTensorToAtbTensor(*(inTensors.at(i)));
      if (variantPacks_.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
        variantPacks_.inTensors.at(i).desc.format = ACL_FORMAT_ND;
      }
    }
  }
  variantPacks_.inTensors.at(inTensors.size()) = CreateBatchStatusAtbHostTensor();

  variantPacks_.outTensors.resize(outTensors.size());
  for (size_t i = 0; i < outTensors.size(); i++) {
    variantPacks_.outTensors.at(i) = ConvertDenseTensorToAtbTensor(*(outTensors.at(i)));
    if (variantPacks_.outTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.outTensors.at(i).desc.format = ACL_FORMAT_ND;
    }
  }
  // param需要更新，依赖这种方式
  BindHostTensorForUpdateParam(variantPacks_);
}

void PpAtbLlamaEncoderLayerParallelOp::UpdateInputTensorAndParam(const paddle::Tensor &seq_len)
{
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(seq_len.place()));

  int32_t batch_size = seq_len.shape().at(0);
  std::vector<int32_t> seq_len_vec;

  auto seq_len_tensor = const_cast<phi::DenseTensor *>(static_cast<const phi::DenseTensor *>(seq_len.impl().get()));
  custom_kernel::TensorToVector(*dev_ctx, *seq_len_tensor, *dev_ctx, &seq_len_vec);

  kv_seq_len_param_.clear();
  q_seq_len_param_.clear();
  batch_status_param_.clear();
  for(auto array: seq_len_vec) {
    int32_t status = array == 0 ? 0 : 1;
    kv_seq_len_param_.push_back(array);
    q_seq_len_param_.push_back(array);
    batch_status_param_.push_back(status);
  }
  for (int i = batch_size; i < maxBatchSize_; i++) {
    batch_status_param_.push_back(0);
  }
}

PpAtbLlamaEncoderLayerParallelOp::PpAtbLlamaEncoderLayerParallelOp(
    const std::string &modelName, int32_t layerNum, int32_t batchSize, int maxBatchSize) : PpAscendAtbOpBase(modelName) {
  layerNum_ = layerNum;
  curBatchSize_ = batchSize;
  maxBatchSize_ = maxBatchSize;
}

PpAtbLlamaEncoderLayerParallelOp::~PpAtbLlamaEncoderLayerParallelOp() {}

std::vector<paddle::Tensor> LlamaEncoderLayerParallelOp(
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

  int32_t batch_size = hidden.shape().at(0);
  // int32_t layer_num = 80; /* TODO:65B，写死8卡 */
  int32_t layer_num = 32; /* TODO:7B，写死8卡 */
  int32_t head_num = headNum;
  int32_t head_dim = headDim;
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(hidden.place()));

  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  auto comm = reinterpret_cast<HcclComm>(phi::detail::GetCCLComm(hidden.place(), 0));

  if (!g_llamaEncoderLayerParallelOp) {
    int32_t max_batch_size = attention_mask.shape().at(0);
    std::cout << "Run In Encoder Parallel layernum: " << layer_num << " head_num: " << head_num << " head_dim: " << head_dim << " max_batchsize: "<< max_batch_size << std::endl;
    g_llamaEncoderLayerParallelOp.reset(new PpAtbLlamaEncoderLayerParallelOp("LlamaEncoderLayerParallelOp", layer_num, batch_size, max_batch_size));

    atb::Operation *op = nullptr; 
    LlamaLayerFusionParallelParam param = {rmsNormEps,
                                           head_num,
                                           head_dim,
                                           0,
                                           0,
                                           1.0 / std::sqrt(head_dim), // qkScale
                                           2,
                                           true,
                                           comm,
                                           true}; // encoder also enable dynamic batch
    LlamaLayerFusionParallelOperation(param, &op);
    g_llamaEncoderLayerParallelOp->operation_.reset(op);
    std::vector<int32_t> layer_id_vec(1, 0);
    custom_kernel::TensorFromVector(*dev_ctx, layer_id_vec,
                                    *dev_ctx, &(g_llamaEncoderLayerParallelOp->layerIdTensor_));
  }

  if (executeCount % layer_num == 0) {
    g_llamaEncoderLayerParallelOp->UpdateInputTensorAndParam(kv_seq_len);
  }

  std::vector<const phi::DenseTensor *> inputs;
  PerpareLlamaEncoderLayerInputs(hidden,
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
                                 kv_seq_len, // 全量阶段
                                 g_llamaEncoderLayerParallelOp->layerIdTensor_,
                                 inputs);
  auto data_type = static_cast<const phi::DenseTensor *>(hidden.impl().get())->dtype();
  std::shared_ptr<phi::DenseTensor> layerout_tensor = std::make_shared<phi::DenseTensor>();
  layerout_tensor->Resize(phi::make_ddim(hidden.shape()));
  dev_ctx->Alloc(layerout_tensor.get(), data_type);

  std::vector<const phi::DenseTensor *> outputs;
  outputs.push_back(layerout_tensor.get());

  g_llamaEncoderLayerParallelOp->Execute(stream, inputs, outputs);

  executeCount++; // Lmhead阶段sync

  return {paddle::Tensor(layerout_tensor), cache_key_value};
}

std::vector<std::vector<int64_t>> LlamaEncoderLayerOpInferShape(
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
    .SetKernelFn(PD_KERNEL(LlamaEncoderLayerParallelOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        LlamaEncoderLayerOpInferShape)); // neccessary if the op has muti_inputs
#endif
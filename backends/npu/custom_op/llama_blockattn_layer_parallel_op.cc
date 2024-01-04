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
#include "llama_layer/llama_blockattn_parallel_operation.h"
#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

int32_t layer_num = 80; /* TODO:65B，写死8卡 */
std::shared_ptr<PpAtbLlamaBlockAttnLayerParallelOp> g_llamaBlockAttnEncoderOp;
std::shared_ptr<PpAtbLlamaBlockAttnLayerParallelOp> g_llamaBlockAttnDecoderOp;
static uint64_t executeCount = 0;
static bool g_isEncoder = true;
static phi::DenseTensor slot_mapping_tensor; // 保存每个token在Cache中存储偏移，shape为[sum(seq_len_pre_batch)]

static bool first_run = true;
paddle::Tensor qkv_deq_blank_bias;
paddle::Tensor self_out_linear_deq_blank_bias; 
paddle::Tensor mlp_deq_blank_bias; 
paddle::Tensor mlp_down_deq_blank_bias;
paddle::Tensor norm_blank_bias; 
paddle::Tensor self_out_norm_blank_bias;
paddle::Tensor empty_offset;

void PerpareLlamaBlockAttnEncoderInputs(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &norm_blank_bias,
    const paddle::Tensor &qkv_mix_weight,
    const paddle::Tensor &qkv_deq_scale,
    const paddle::Tensor &qkv_deq_blank_bias,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_linear_shift,
    const paddle::Tensor &self_out_linear_smooth,
    const paddle::Tensor &self_out_linear_deq_scale,
    const paddle::Tensor &self_out_linear_deq_blank_bias,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &self_out_norm_blank_bias,
    const paddle::Tensor &mlp_gate_up_weight,
    const paddle::Tensor &mlp_deq_scale,
    const paddle::Tensor &mlp_deq_blank_bias,
    const paddle::Tensor &mlp_down_weight,
    const paddle::Tensor &mlp_down_shift,
    const paddle::Tensor &mlp_down_smooth,
    const paddle::Tensor &mlp_down_deq_scale,
    const paddle::Tensor &mlp_down_deq_blank_bias,
    const paddle::Tensor &cos_table,
    const paddle::Tensor &sin_table,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &cache_key,
    const paddle::Tensor &cache_value,
    const paddle::Tensor &cache_k_quant_scales,
    const paddle::Tensor &cache_k_dequant_scales,
    const paddle::Tensor &cache_v_quant_scales,
    const paddle::Tensor &cache_v_dequant_scales,
    const paddle::Tensor &empty_offset,
    const paddle::Tensor &seq_len,
    const paddle::Tensor &block_tables,
    const phi::DenseTensor &slot_mapping_tensor,
    std::vector<const phi::DenseTensor *> &inputs) {

  auto hidden_tensor = static_cast<const phi::DenseTensor *>(hidden.impl().get());
  auto norm_weight_tensor = static_cast<const phi::DenseTensor *>(norm_weight.impl().get());
  auto norm_blank_bias_tensor = static_cast<const phi::DenseTensor *>(norm_blank_bias.impl().get());
  auto qkv_mix_weight_tensor = static_cast<const phi::DenseTensor *>(qkv_mix_weight.impl().get());
  auto qkv_deq_scale_tensor = static_cast<const phi::DenseTensor *>(qkv_deq_scale.impl().get());
  auto qkv_deq_blank_bias_tensor = static_cast<const phi::DenseTensor *>(qkv_deq_blank_bias.impl().get());
  auto self_out_linear_weight_tensor = static_cast<phi::DenseTensor *>(self_out_linear_weight.impl().get());
  auto self_out_linear_shift_tensor = static_cast<phi::DenseTensor *>(self_out_linear_shift.impl().get());
  auto self_out_linear_smooth_tensor = static_cast<phi::DenseTensor *>(self_out_linear_smooth.impl().get());
  auto self_out_linear_deq_scale_tensor = static_cast<const phi::DenseTensor *>(self_out_linear_deq_scale.impl().get());
  auto self_out_linear_deq_blank_bias_tensor = static_cast<const phi::DenseTensor *>(self_out_linear_deq_blank_bias.impl().get());
  auto self_out_norm_weight_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_weight.impl().get());
  auto self_out_norm_blank_bias_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_blank_bias.impl().get());
  auto mlp_gate_up_weight_tensor = static_cast<phi::DenseTensor *>(mlp_gate_up_weight.impl().get());
  auto mlp_deq_scale_tensor = static_cast<phi::DenseTensor *>(mlp_deq_scale.impl().get());
  auto mlp_deq_blank_bias_tensor = static_cast<phi::DenseTensor *>(mlp_deq_blank_bias.impl().get());
  auto mlp_down_weight_tensor = static_cast<phi::DenseTensor *>(mlp_down_weight.impl().get());
  auto mlp_down_shift_tensor = static_cast<phi::DenseTensor *>(mlp_down_shift.impl().get());
  auto mlp_down_smooth_tensor = static_cast<phi::DenseTensor *>(mlp_down_smooth.impl().get());
  auto mlp_down_deq_scale_tensor = static_cast<phi::DenseTensor *>(mlp_down_deq_scale.impl().get());
  auto mlp_down_deq_blank_bias_tensor = static_cast<phi::DenseTensor *>(mlp_down_deq_blank_bias.impl().get());
  auto cos_table_tensor = static_cast<const phi::DenseTensor *>(cos_table.impl().get());
  auto sin_table_tensor = static_cast<const phi::DenseTensor *>(sin_table.impl().get());
  auto attention_mask_tensor = static_cast<const phi::DenseTensor *>(attention_mask.impl().get());
  auto cache_key_tensor = static_cast<const phi::DenseTensor *>(cache_key.impl().get());
  auto cache_value_tensor = static_cast<const phi::DenseTensor *>(cache_value.impl().get());
  auto cache_k_quant_scales_tensor = static_cast<const phi::DenseTensor *>(cache_k_quant_scales.impl().get());
  auto cache_k_dequant_scales_tensor = static_cast<const phi::DenseTensor *>(cache_k_dequant_scales.impl().get());
  auto cache_v_quant_scales_tensor = static_cast<const phi::DenseTensor *>(cache_v_quant_scales.impl().get());
  auto cache_v_dequant_scales_tensor = static_cast<const phi::DenseTensor *>(cache_v_dequant_scales.impl().get());
  auto empty_offset_tensor = static_cast<const phi::DenseTensor *>(empty_offset.impl().get());
  auto seq_len_tensor = static_cast<const phi::DenseTensor *>(seq_len.impl().get());
  auto block_tables_tensor = static_cast<const phi::DenseTensor *>(block_tables.impl().get());

  inputs.push_back(hidden_tensor);
  inputs.push_back(norm_weight_tensor);
  inputs.push_back(norm_blank_bias_tensor);
  inputs.push_back(qkv_mix_weight_tensor);
  inputs.push_back(qkv_deq_scale_tensor);
  inputs.push_back(qkv_deq_blank_bias_tensor);
  inputs.push_back(self_out_linear_weight_tensor);
  inputs.push_back(self_out_linear_shift_tensor);
  inputs.push_back(self_out_linear_smooth_tensor);
  inputs.push_back(self_out_linear_deq_scale_tensor);
  inputs.push_back(self_out_linear_deq_blank_bias_tensor);
  inputs.push_back(self_out_norm_weight_tensor);
  inputs.push_back(self_out_norm_blank_bias_tensor);
  inputs.push_back(mlp_gate_up_weight_tensor);
  inputs.push_back(mlp_deq_scale_tensor);
  inputs.push_back(mlp_deq_blank_bias_tensor);
  inputs.push_back(mlp_down_weight_tensor);
  inputs.push_back(mlp_down_shift_tensor);
  inputs.push_back(mlp_down_smooth_tensor);
  inputs.push_back(mlp_down_deq_scale_tensor);
  inputs.push_back(mlp_down_deq_blank_bias_tensor);
  inputs.push_back(cos_table_tensor);
  inputs.push_back(sin_table_tensor);
  inputs.push_back(attention_mask_tensor);
  inputs.push_back(cache_key_tensor);
  inputs.push_back(cache_value_tensor);
  inputs.push_back(cache_k_quant_scales_tensor);
  inputs.push_back(cache_k_dequant_scales_tensor);
  inputs.push_back(cache_v_quant_scales_tensor);
  inputs.push_back(cache_v_dequant_scales_tensor);
  inputs.push_back(empty_offset_tensor);
  inputs.push_back(seq_len_tensor);
  inputs.push_back(block_tables_tensor);
  inputs.push_back(&slot_mapping_tensor);
}

void PerpareLlamaBlockAttnDecoderInputs(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &norm_blank_bias,
    const paddle::Tensor &qkv_mix_weight,
    const paddle::Tensor &qkv_deq_scale,
    const paddle::Tensor &qkv_deq_blank_bias,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_linear_shift,
    const paddle::Tensor &self_out_linear_smooth,
    const paddle::Tensor &self_out_linear_deq_scale,
    const paddle::Tensor &self_out_linear_deq_blank_bias,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &self_out_norm_blank_bias,
    const paddle::Tensor &mlp_gate_up_weight,
    const paddle::Tensor &mlp_deq_scale,
    const paddle::Tensor &mlp_deq_blank_bias,
    const paddle::Tensor &mlp_down_weight,
    const paddle::Tensor &mlp_down_shift,
    const paddle::Tensor &mlp_down_smooth,
    const paddle::Tensor &mlp_down_deq_scale,
    const paddle::Tensor &mlp_down_deq_blank_bias,
    const paddle::Tensor &cos_table,
    const paddle::Tensor &sin_table,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &cache_key,
    const paddle::Tensor &cache_value,
    const paddle::Tensor &cache_k_quant_scales,
    const paddle::Tensor &cache_k_dequant_scales,
    const paddle::Tensor &cache_v_quant_scales,
    const paddle::Tensor &cache_v_dequant_scales,
    const paddle::Tensor &empty_offset,
    const phi::DenseTensor &seq_len_tensor,
    const paddle::Tensor &block_tables,
    const phi::DenseTensor &slot_mapping_tensor,
    std::vector<const phi::DenseTensor *> &inputs) {

  auto hidden_tensor = static_cast<const phi::DenseTensor *>(hidden.impl().get());
  auto norm_weight_tensor = static_cast<const phi::DenseTensor *>(norm_weight.impl().get());
  auto norm_blank_bias_tensor = static_cast<const phi::DenseTensor *>(norm_blank_bias.impl().get());
  auto qkv_mix_weight_tensor = static_cast<const phi::DenseTensor *>(qkv_mix_weight.impl().get());
  auto qkv_deq_scale_tensor = static_cast<const phi::DenseTensor *>(qkv_deq_scale.impl().get());
  auto qkv_deq_blank_bias_tensor = static_cast<const phi::DenseTensor *>(qkv_deq_blank_bias.impl().get());
  auto self_out_linear_weight_tensor = static_cast<phi::DenseTensor *>(self_out_linear_weight.impl().get());
  auto self_out_linear_shift_tensor = static_cast<phi::DenseTensor *>(self_out_linear_shift.impl().get());
  auto self_out_linear_smooth_tensor = static_cast<phi::DenseTensor *>(self_out_linear_smooth.impl().get());
  auto self_out_linear_deq_scale_tensor = static_cast<const phi::DenseTensor *>(self_out_linear_deq_scale.impl().get());
  auto self_out_linear_deq_blank_bias_tensor = static_cast<const phi::DenseTensor *>(self_out_linear_deq_blank_bias.impl().get());
  auto self_out_norm_weight_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_weight.impl().get());
  auto self_out_norm_blank_bias_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_blank_bias.impl().get());
  auto mlp_gate_up_weight_tensor = static_cast<phi::DenseTensor *>(mlp_gate_up_weight.impl().get());
  auto mlp_deq_scale_tensor = static_cast<phi::DenseTensor *>(mlp_deq_scale.impl().get());
  auto mlp_deq_blank_bias_tensor = static_cast<phi::DenseTensor *>(mlp_deq_blank_bias.impl().get());
  auto mlp_down_weight_tensor = static_cast<phi::DenseTensor *>(mlp_down_weight.impl().get());
  auto mlp_down_shift_tensor = static_cast<phi::DenseTensor *>(mlp_down_shift.impl().get());
  auto mlp_down_smooth_tensor = static_cast<phi::DenseTensor *>(mlp_down_smooth.impl().get());
  auto mlp_down_deq_scale_tensor = static_cast<phi::DenseTensor *>(mlp_down_deq_scale.impl().get());
  auto mlp_down_deq_blank_bias_tensor = static_cast<phi::DenseTensor *>(mlp_down_deq_blank_bias.impl().get());
  auto cos_table_tensor = static_cast<const phi::DenseTensor *>(cos_table.impl().get());
  auto sin_table_tensor = static_cast<const phi::DenseTensor *>(sin_table.impl().get());
  auto attention_mask_tensor = static_cast<const phi::DenseTensor *>(attention_mask.impl().get());
  auto cache_key_tensor = static_cast<const phi::DenseTensor *>(cache_key.impl().get());
  auto cache_value_tensor = static_cast<const phi::DenseTensor *>(cache_value.impl().get());
  auto cache_k_quant_scales_tensor = static_cast<const phi::DenseTensor *>(cache_k_quant_scales.impl().get());
  auto cache_k_dequant_scales_tensor = static_cast<const phi::DenseTensor *>(cache_k_dequant_scales.impl().get());
  auto cache_v_quant_scales_tensor = static_cast<const phi::DenseTensor *>(cache_v_quant_scales.impl().get());
  auto cache_v_dequant_scales_tensor = static_cast<const phi::DenseTensor *>(cache_v_dequant_scales.impl().get());  
  auto empty_offset_tensor = static_cast<const phi::DenseTensor *>(empty_offset.impl().get());
  // auto seq_len_tensor = static_cast<const phi::DenseTensor *>(seq_len.impl().get());
  auto block_tables_tensor = static_cast<const phi::DenseTensor *>(block_tables.impl().get());

  inputs.push_back(hidden_tensor);
  inputs.push_back(norm_weight_tensor);
  inputs.push_back(norm_blank_bias_tensor);
  inputs.push_back(qkv_mix_weight_tensor);
  inputs.push_back(qkv_deq_scale_tensor);
  inputs.push_back(qkv_deq_blank_bias_tensor);
  inputs.push_back(self_out_linear_weight_tensor);
  inputs.push_back(self_out_linear_shift_tensor);
  inputs.push_back(self_out_linear_smooth_tensor);
  inputs.push_back(self_out_linear_deq_scale_tensor);
  inputs.push_back(self_out_linear_deq_blank_bias_tensor);
  inputs.push_back(self_out_norm_weight_tensor);
  inputs.push_back(self_out_norm_blank_bias_tensor);
  inputs.push_back(mlp_gate_up_weight_tensor);
  inputs.push_back(mlp_deq_scale_tensor);
  inputs.push_back(mlp_deq_blank_bias_tensor);
  inputs.push_back(mlp_down_weight_tensor);
  inputs.push_back(mlp_down_shift_tensor);
  inputs.push_back(mlp_down_smooth_tensor);
  inputs.push_back(mlp_down_deq_scale_tensor);
  inputs.push_back(mlp_down_deq_blank_bias_tensor);
  inputs.push_back(cos_table_tensor);
  inputs.push_back(sin_table_tensor);
  inputs.push_back(attention_mask_tensor);
  inputs.push_back(cache_key_tensor);
  inputs.push_back(cache_value_tensor);
  inputs.push_back(cache_k_quant_scales_tensor);
  inputs.push_back(cache_k_dequant_scales_tensor);
  inputs.push_back(cache_v_quant_scales_tensor);
  inputs.push_back(cache_v_dequant_scales_tensor);
  inputs.push_back(empty_offset_tensor);  
  inputs.push_back(&seq_len_tensor);
  inputs.push_back(block_tables_tensor);
  inputs.push_back(&slot_mapping_tensor);
}

void PpAtbLlamaBlockAttnLayerParallelOp::BindHostTensorForUpdateParam(atb::VariantPack &variantPack)
{
  if (g_isEncoder) { // 只有Encoder阶段需要这个param。
    const uint32_t seqLenTensorId = LlamaBlockAttnParallelTensorId::IN_SEQLEN;
    variantPack.inTensors.at(seqLenTensorId).hostData = seq_len_param_.data();
  }
}

void PpAtbLlamaBlockAttnLayerParallelOp::BuildVariantPack(std::vector<const phi::DenseTensor *> &inTensors,
                                                        std::vector<const phi::DenseTensor *> &outTensors)
{
  variantPacks_.inTensors.resize(inTensors.size());
  for (size_t i = 0; i < inTensors.size(); i++) {
    variantPacks_.inTensors.at(i) = ConvertDenseTensorToAtbTensor(*(inTensors.at(i)));
    if (variantPacks_.inTensors.at(i).desc.format == ACL_FORMAT_NCHW) {
      variantPacks_.inTensors.at(i).desc.format = ACL_FORMAT_ND;
    }
    if (i == LlamaBlockAttnParallelTensorId::IN_CACHE_K || i == LlamaBlockAttnParallelTensorId::IN_CACHE_V){
      variantPacks_.inTensors.at(i).desc.dtype = ACL_INT8;
    }
  }

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

void PpAtbLlamaBlockAttnLayerParallelOp::UpdateInputTensorAndParam(const paddle::Tensor &block_tables, const paddle::Tensor &seq_len, int32_t block_size)
{
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(seq_len.place()));

  int32_t batch_size = seq_len.shape().at(0);
  std::vector<int32_t> seq_len_vec;

  auto seq_len_tensor = const_cast<phi::DenseTensor *>(static_cast<const phi::DenseTensor *>(seq_len.impl().get()));
  custom_kernel::TensorToVector(*dev_ctx, *seq_len_tensor, *dev_ctx, &seq_len_vec);

  std::vector<int32_t> block_tables_vec;
  auto block_tables_tensor = const_cast<phi::DenseTensor *>(static_cast<const phi::DenseTensor *>(block_tables.impl().get()));
  custom_kernel::TensorToVector(*dev_ctx, *block_tables_tensor, *dev_ctx, &block_tables_vec);

  std::vector<int32_t> slot_mapping_vec; // slot mapping保存的是block table中以0开始偏移的slot id，TODO待优化为模型传入
  int32_t pre_max_block_num = block_tables.shape().at(1);
  int32_t block_offset = 0;
  if (g_isEncoder) {
    int32_t total_seq_len = std::accumulate(seq_len_vec.begin(), seq_len_vec.end(), 0);
    slot_mapping_vec.reserve(total_seq_len);
    seq_len_param_.clear(); // 只有encoder需要这个param

    for (int32_t i = 0; i < batch_size; i++) {
      int32_t len = seq_len_vec[i];
      int32_t need_block_num = (len + block_size - 1) / block_size;
      int32_t mod_len = len;
      int32_t slot_offset;

      seq_len_param_.push_back(len);
      for (int32_t j = 0; j < need_block_num - 1; j++) {
        slot_offset = block_tables_vec[block_offset + j] * block_size;
        for (int32_t k = 0; k < block_size; k++) {
          slot_mapping_vec.push_back(slot_offset + k);
        }
        mod_len -= block_size;
      }
      slot_offset = block_tables_vec[block_offset + need_block_num - 1] * block_size;
      for (int32_t k = 0; k < mod_len; k++) {
        slot_mapping_vec.push_back(slot_offset + k);
      }
      block_offset += pre_max_block_num;
    }
  } else {
    std::vector<int32_t> token_offset;
    token_offset.resize(batch_size, 1); // decoder_seqlen不包含增量当前的长度

    slot_mapping_vec.reserve(batch_size); // 增量阶段，slotmapping只关注增量token
    for (int32_t i = 0; i < batch_size; i++) {
      int32_t len = seq_len_vec[i];
      int32_t need_block_num = (len + 1 + block_size - 1) / block_size;
      int32_t slot_id = block_tables_vec[block_offset + need_block_num - 1] * block_size + (len % block_size);

      slot_mapping_vec.push_back(slot_id);
      block_offset += pre_max_block_num;
      token_offset[i] += seq_len_vec[i];
    }
    custom_kernel::TensorFromVector(*dev_ctx, token_offset,
                                  *dev_ctx, &token_offset_tensor_);
  }

  custom_kernel::TensorFromVector(*dev_ctx, slot_mapping_vec,
                                  *dev_ctx, &slot_mapping_tensor);
}

PpAtbLlamaBlockAttnLayerParallelOp::PpAtbLlamaBlockAttnLayerParallelOp(
    const std::string &modelName, int layer_num) : PpAscendAtbOpBase(modelName) {
    operations_.resize(layer_num);
}

PpAtbLlamaBlockAttnLayerParallelOp::~PpAtbLlamaBlockAttnLayerParallelOp() {}

bool isEncoderToken(const paddle::Tensor &encoder_seq_len)
{
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(encoder_seq_len.place()));
  std::vector<int32_t> seq_len_vec;
  auto seq_len_tensor = const_cast<phi::DenseTensor *>(static_cast<const phi::DenseTensor *>(encoder_seq_len.impl().get()));
  custom_kernel::TensorToVector(*dev_ctx, *seq_len_tensor, *dev_ctx, &seq_len_vec);

  for(auto array: seq_len_vec) {
    if (array > 0) {
      return true; // 只要encoder非零，即认为是prefill阶段
    }
  }

  return false;
}

void InitAtbLlamaBlockAttnLayerOp(std::shared_ptr<PpAtbLlamaBlockAttnLayerParallelOp> &block_op,
                                  float rmsNormEps, int32_t head_num, int32_t head_dim, HcclComm comm,
                                  float inputRmsNormScale, float selfRmsNormScale, float selfQuantScale, float mlpQuantScale, int layerid)
{
    if(!block_op){
      std::cout << "Run In Block Attention Parallel isPrefill:" << g_isEncoder <<
      " head_num: " << head_num << " head_dim: " << head_dim << std::endl;
      block_op.reset(new PpAtbLlamaBlockAttnLayerParallelOp("LlamaBlockAttnLayerParallelOp", layer_num));
    }

    std::string device_id_str = getenv("FLAGS_selected_npus");
    int device_id = stoi(device_id_str);
    int nranks = 8;

    atb::Operation *op = nullptr; 
    LlamaBlockAttnParallelParam param = {rmsNormEps,
                                         head_num,
                                         head_dim,
                                         device_id,
                                         nranks,
                                         1.0 / std::sqrt(head_dim), // qkScale
                                         2, // rotaryCoeff
                                         true,
                                         comm,
                                         g_isEncoder, // isPrefill
                                         selfRmsNormScale,
                                         0,
                                         selfQuantScale,
                                         0,
                                         inputRmsNormScale,
                                         0,
                                         mlpQuantScale,
                                         0}; 
    LlamaBlockAttnParallelOperation(param, &op);
    block_op->operations_.at(layerid).reset(op);
}

std::vector<paddle::Tensor> LlamaBlockAttnLayerParallelOp(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &qkv_mix_weight,
    const paddle::Tensor &qkv_deq_scale,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_linear_shift,
    const paddle::Tensor &self_out_linear_smooth,
    const paddle::Tensor &self_out_linear_deq_scale,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &mlp_gate_up_weight,
    const paddle::Tensor &mlp_deq_scale,
    const paddle::Tensor &mlp_down_weight,
    const paddle::Tensor &mlp_down_shift,
    const paddle::Tensor &mlp_down_smooth,
    const paddle::Tensor &mlp_down_deq_scale,
    const paddle::Tensor &cos_table,
    const paddle::Tensor &sin_table,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &cache_key,
    const paddle::Tensor &cache_value,
    const paddle::Tensor &decoder_seq_len,
    const paddle::Tensor &encoder_seq_len,
    const paddle::Tensor &block_tables,
    const paddle::Tensor &cache_k_quant_scales,
    const paddle::Tensor &cache_k_dequant_scales,
    const paddle::Tensor &cache_v_quant_scales,
    const paddle::Tensor &cache_v_dequant_scales,
    int32_t block_size,
    float rmsNormEps,
    float inputRmsNormScale,
    float selfRmsNormScale,
    float selfQuantScale,
    float mlpQuantScale) {

  int32_t batch_size = hidden.shape().at(0);
  int32_t head_num = cache_key.shape().at(1);
  int32_t head_dim = cache_key.shape().at(3);
  int32_t max_batch_size = attention_mask.shape().at(0);

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(hidden.place()));

  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  auto comm = reinterpret_cast<HcclComm>(phi::detail::GetCCLComm(hidden.place(), 0));

  int layer_id = executeCount % layer_num;
  if (layer_id == 0) {
    g_isEncoder = isEncoderToken(encoder_seq_len);
  }

  inputRmsNormScale *= 127;
  selfRmsNormScale *= 127;
  selfQuantScale *= 127;
  mlpQuantScale *= 127;
  if (g_isEncoder && (!g_llamaBlockAttnEncoderOp || !g_llamaBlockAttnEncoderOp->operations_.at(layer_id))) {
    InitAtbLlamaBlockAttnLayerOp(g_llamaBlockAttnEncoderOp, rmsNormEps, head_num, head_dim, comm, inputRmsNormScale, selfRmsNormScale, selfQuantScale, mlpQuantScale, layer_id);
  } else if (!g_isEncoder && (!g_llamaBlockAttnDecoderOp || !g_llamaBlockAttnDecoderOp->operations_.at(layer_id))) {
    InitAtbLlamaBlockAttnLayerOp(g_llamaBlockAttnDecoderOp, rmsNormEps, head_num, head_dim, comm, inputRmsNormScale, selfRmsNormScale, selfQuantScale, mlpQuantScale, layer_id);
  }

  if (layer_id == 0) {
    if (g_isEncoder) {
      g_llamaBlockAttnEncoderOp->output_->Resize(phi::make_ddim(hidden.shape()));
      dev_ctx->Alloc(g_llamaBlockAttnEncoderOp->output_.get(), 
        static_cast<const phi::DenseTensor *>(hidden.impl().get())->dtype());
      g_llamaBlockAttnEncoderOp->UpdateInputTensorAndParam(block_tables, encoder_seq_len, block_size);
    } else {
      g_llamaBlockAttnDecoderOp->output_->Resize(phi::make_ddim(hidden.shape()));
      g_llamaBlockAttnDecoderOp->UpdateInputTensorAndParam(block_tables, decoder_seq_len, block_size);
    }
    executeCount = 0;
  }
  std::vector<const phi::DenseTensor *> inputs;

  executeCount++;
  if (first_run) {
      qkv_deq_blank_bias = paddle::full({qkv_mix_weight.shape()[0]}, 0, paddle::DataType::INT32, hidden.place()); 
      self_out_linear_deq_blank_bias = paddle::full({self_out_linear_weight.shape()[0]}, 0, paddle::DataType::INT32, hidden.place()); 
      mlp_deq_blank_bias = paddle::full({mlp_gate_up_weight.shape()[0]}, 0, paddle::DataType::INT32, hidden.place()); 
      mlp_down_deq_blank_bias = paddle::full({mlp_down_weight.shape()[0]}, 0, paddle::DataType::INT32, hidden.place()); 
      norm_blank_bias = paddle::full(norm_weight.shape(), 0, paddle::DataType::FLOAT16, hidden.place()); 
      self_out_norm_blank_bias = paddle::full(self_out_norm_weight.shape(), 0, paddle::DataType::FLOAT16, hidden.place()); 
      empty_offset = paddle::full({}, 0, paddle::DataType::INT8, hidden.place());
      first_run = false;
  }

  if (g_isEncoder) {
    PerpareLlamaBlockAttnEncoderInputs(hidden,
                                       norm_weight,
                                       norm_blank_bias,
                                       qkv_mix_weight,
                                       qkv_deq_scale,
                                       qkv_deq_blank_bias,
                                       self_out_linear_weight,
                                       self_out_linear_shift,
                                       self_out_linear_smooth,
                                       self_out_linear_deq_scale,
                                       self_out_linear_deq_blank_bias,
                                       self_out_norm_weight,
                                       self_out_norm_blank_bias,
                                       mlp_gate_up_weight,
                                       mlp_deq_scale,
                                       mlp_deq_blank_bias,
                                       mlp_down_weight,
                                       mlp_down_shift,
                                       mlp_down_smooth,
                                       mlp_down_deq_scale,
                                       mlp_down_deq_blank_bias,
                                       cos_table,
                                       sin_table,
                                       attention_mask,
                                       cache_key,
                                       cache_value,
                                       cache_k_quant_scales,
                                       cache_k_dequant_scales,
                                       cache_v_quant_scales,
                                       cache_v_dequant_scales,
                                       empty_offset,
                                       encoder_seq_len,
                                       block_tables,
                                       slot_mapping_tensor,
                                       inputs);
    std::vector<const phi::DenseTensor *> outputs = {g_llamaBlockAttnEncoderOp->output_.get()};
    g_llamaBlockAttnEncoderOp->Execute(stream, inputs, outputs, layer_id);
    return {paddle::Tensor(g_llamaBlockAttnEncoderOp->output_)};
  }
  PerpareLlamaBlockAttnDecoderInputs(hidden,
                                     norm_weight,
                                     norm_blank_bias,
                                     qkv_mix_weight,
                                     qkv_deq_scale,
                                     qkv_deq_blank_bias,
                                     self_out_linear_weight,
                                     self_out_linear_shift,
                                     self_out_linear_smooth,
                                     self_out_linear_deq_scale,
                                     self_out_linear_deq_blank_bias,
                                     self_out_norm_weight,
                                     self_out_norm_blank_bias,
                                     mlp_gate_up_weight,
                                     mlp_deq_scale,
                                     mlp_deq_blank_bias,
                                     mlp_down_weight,
                                     mlp_down_shift,
                                     mlp_down_smooth,
                                     mlp_down_deq_scale,
                                     mlp_down_deq_blank_bias,
                                     cos_table,
                                     sin_table,
                                     attention_mask,
                                     cache_key,
                                     cache_value,
                                     cache_k_quant_scales,
                                     cache_k_dequant_scales,
                                     cache_v_quant_scales,
                                     cache_v_dequant_scales,
                                     empty_offset,
                                     g_llamaBlockAttnDecoderOp->token_offset_tensor_,
                                     block_tables,
                                     slot_mapping_tensor,
                                     inputs);
  std::vector<const phi::DenseTensor *> outputs = {g_llamaBlockAttnDecoderOp->output_.get()};
  g_llamaBlockAttnDecoderOp->Execute(stream, inputs, outputs, layer_id);
  return {paddle::Tensor(g_llamaBlockAttnDecoderOp->output_)};
}

std::vector<std::vector<int64_t>> LlamaBlockAttnLayerOpInferShape(
    const std::vector<int64_t> &hidden_shape,
    const std::vector<int64_t> &norm_weight_shape,
    const std::vector<int64_t> &qkv_mix_weight_shape,
    const std::vector<int64_t> &qkv_deq_Scale_shape,
    const std::vector<int64_t> &self_out_linear_weight_shape,
    const std::vector<int64_t> &self_out_linear_shift_shape,
    const std::vector<int64_t> &self_out_linear_smooth_shape,
    const std::vector<int64_t> &self_out_linear_deq_scale_shape,
    const std::vector<int64_t> &self_out_norm_weight_shape,
    const std::vector<int64_t> &mlp_gate_up_weight_shape,
    const std::vector<int64_t> &mlp_deq_scale_shape,
    const std::vector<int64_t> &mlp_down_weight_shape,
    const std::vector<int64_t> &mlp_down_shift_shape,
    const std::vector<int64_t> &mlp_down_smooth_shape,
    const std::vector<int64_t> &mlp_down_deq_scale_shape,
    const std::vector<int64_t> &cos_table_shape,
    const std::vector<int64_t> &sin_table_shape,
    const std::vector<int64_t> &attention_mask_shape,
    const std::vector<int64_t> &cacheK_shape,
    const std::vector<int64_t> &cacheV_shape,
    const std::vector<int64_t> &decoder_seq_len_shape,
    const std::vector<int64_t> &encoder_seq_len_shape,
    const std::vector<int64_t> &block_tables_shape,
    const std::vector<int64_t> &cache_k_quant_scales_shape,
    const std::vector<int64_t> &cache_k_dequant_scales_shape,
    const std::vector<int64_t> &cache_v_quant_scales_shape,
    const std::vector<int64_t> &cache_v_dequant_scales_shape,
    int32_t block_size,
    float rmsNormEps,
    float inputRmsNormScale,
    float selfRmsNormScale,
    float selfQuantScale,
    float mlpQuantScale) {

  return {hidden_shape};
}

PD_BUILD_OP(llama_blockattn_layer_parallel)
    .Inputs({"Hidden",
             "NormWeight",
             "QKVMixWeight",
             "QKVDeqScale",
             "SelfOutLinearWeight",
             "SelfOutLinearShift",
             "SelfOutLinearSmooth",
             "SelfOutLinearDeqScale",
             "SelfOutNormWeight",
             "MlpGateUpWeight",
             "MlpDeqScale",
             "MlpDownWeight",
             "MlpDownShift",
             "MlpDownSmooth",
             "MlpDownDeqScale",
             "CosTable",
             "SinTable",
             "AttentionMask",
             "Cache_K",
             "Cache_V",
             "DecoderSeqLength",
             "EncoderSeqLength",
             "BlockTables",
             "CacheKQuantScales",
             "CacheKDequantScales",
             "CacheVQuantScales",
             "CacheVDequantScales"})
    .Outputs({"Out"})
    .Attrs({"block_size: int", "rmsNormEps: float", "inputRmsNormScale: float", "selfRmsNormScale: float", "selfQuantScale: float", "mlpQuantScale: float"})
    .SetKernelFn(PD_KERNEL(LlamaBlockAttnLayerParallelOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        LlamaBlockAttnLayerOpInferShape)); // neccessary if the op has muti_inputs
#endif
